import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from utils import get_chebyshev_laplacian, load_ems_data
from model import make_model
from evaluate import evaluate
from profiling import print_profiling_report


if __name__ == "__main__":

    DIST_FILE = 'data/adj_matrix.npy'
    DATA_FILE = 'data/All_Features_3.csv'
    N_NODES = 197
    M_HISTORY = 24
    KS = 3
    # For KT we use a fixed (1,3) temporal conv
    EPOCHS = 15

    Z_SCORE_COLS = [
        'call_count', 'lag_1', 'lag_24', 'lag_168',
        'roll_mean_24', 'roll_std_24', 'Population', 'INITIAL_SEVERITY_LEVEL_CODE'
    ]

    PASS_THROUGH_COLS = [
        'hour_sin', 'hour_cos',
        'dayofweek_sin', 'dayofweek_cos',
        'month_sin', 'month_cos',
        'weekofyear_sin', 'weekofyear_cos',
        'is_Covid', 'SPECIAL_EVENT_INDICATOR'
    ]

    NUM_FEATURES = len(Z_SCORE_COLS) + len(PASS_THROUGH_COLS)
    print(f"Total Features Configured: {NUM_FEATURES}", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on {device}", flush=True)

    # get_chebyshev_laplacian returns a torch tensor and we need the numpy version for make_model
    L_tilde_tensor = get_chebyshev_laplacian(DIST_FILE, N_NODES, KS)
    L_tilde_np = L_tilde_tensor.numpy()

    print("Loading EMS data...")
    (X_train, Y_train), (X_val, Y_val), (X_test, Y_test), (mean_z, std_z) = load_ems_data(
        csv_path=DATA_FILE,
        z_score_cols=Z_SCORE_COLS,
        pass_through_cols=PASS_THROUGH_COLS,
        history_window=M_HISTORY,
        prediction_horizon=1,
        train_ratio=0.6,
        val_ratio=0.2
    )

    # X shape should be (B, N, F, T) to match model input so rearranging the dimensions here
    X_train = X_train.permute(0, 3, 1, 2)   # (B, N, F, T)
    X_val   = X_val.permute(0, 3, 1, 2)
    X_test  = X_test.permute(0, 3, 1, 2)

    test_dataset = TensorDataset(X_test, Y_test)
    test_loader = DataLoader(test_dataset, batch_size=50, shuffle=False)

    print("\n" + "="*60, flush=True)
    print("TRAINING ASTGCN (RECENT COMPONENT ONLY)", flush=True)
    print("="*60, flush=True)

    best_params = {
        'lr': 0.0005,
        'batch_size': 32,
        'nb_chev_filter': 64,
        'nb_time_filter': 32,
        'optimizer': 'AdamW',
        'step_size': 7
    }

    best_train_loader = DataLoader(TensorDataset(X_train, Y_train), batch_size=best_params["batch_size"], shuffle=True)
    best_val_loader = DataLoader(TensorDataset(X_val, Y_val), batch_size=best_params["batch_size"], shuffle=False)

    print('loaders created', flush = True)
    model = make_model(
        DEVICE=device,
        nb_block=2,
        in_channels=NUM_FEATURES,
        K=KS,
        nb_chev_filter=best_params["nb_chev_filter"],
        nb_time_filter=best_params["nb_time_filter"],
        time_strides=1,
        L_tilde_np=L_tilde_np,
        num_for_predict=1,
        len_input=M_HISTORY,
        num_of_vertices=N_NODES,
    )

    print('model created', flush = True)

    final_optimizer = getattr(optim, best_params["optimizer"])(model.parameters(), lr=best_params["lr"])
    final_scheduler = optim.lr_scheduler.StepLR(final_optimizer, step_size=best_params["step_size"], gamma=0.7)
    final_loss_fn = nn.MSELoss()

    best_final_val_loss = float('inf')
    train_losses, val_losses = [], []

    use_cuda = device.type == "cuda"
    if use_cuda:
        torch.cuda.reset_peak_memory_stats(device)

    for epoch in range(EPOCHS):
        model.train()
        train_loss_sum = 0.0
        train_count = 0
        for batch_x, batch_y in best_train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            final_optimizer.zero_grad()
            pred = model(batch_x)          # (B, N, T_out)
            loss = final_loss_fn(pred.squeeze(-1), batch_y.squeeze(1))
            loss.backward()
            final_optimizer.step()
            train_loss_sum += loss.item() * batch_x.size(0)
            train_count += batch_x.size(0)

        epoch_train_loss = train_loss_sum / train_count
        train_losses.append(epoch_train_loss)

        model.eval()
        val_loss_sum = 0.0
        val_count = 0
        with torch.no_grad():
            for batch_x, batch_y in best_val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                pred = model(batch_x)      # (B, N, T_out)
                loss = final_loss_fn(pred.squeeze(-1), batch_y.squeeze(1))
                val_loss_sum += loss.item() * batch_x.size(0)
                val_count += batch_x.size(0)

        epoch_val_loss = val_loss_sum / val_count
        val_losses.append(epoch_val_loss)

        final_scheduler.step()

        print(f"ASTGCN Epoch {epoch+1}/{EPOCHS} | Train: {epoch_train_loss:.4f} | Val: {epoch_val_loss:.4f}", flush=True)

    peak_memory_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)

    # Profiling report 
    print_profiling_report(model, test_loader, device, peak_memory_mb)

    target_mean = mean_z[0, 0, 0]
    target_std = std_z[0, 0, 0]

    mae, rmse, r2, preds_denorm, targs_denorm = evaluate(
        model, test_loader, device, target_mean, target_std
    )

    print("Final Test Evaluation Results", flush=True)
    print(f"MAE:  {mae:.4f}", flush=True)
    print(f"RMSE: {rmse:.4f}", flush=True)
    print(f"MSE:  {rmse**2:.4f}", flush=True)
    print(f"R²:   {r2:.4f}", flush=True)
