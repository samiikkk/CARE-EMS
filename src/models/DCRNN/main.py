import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from utils import get_adjacency_matrix, load_ems_data
from model import DCRNNModel
from evaluate import evaluate
from profiling import print_profiling_report


if __name__ == "__main__":

    DIST_FILE  = 'data/adj_matrix_assym.npy'
    DATA_FILE  = 'data/All_Features_3.csv'
    N_NODES    = 197
    M_HISTORY  = 24       # encoder look-back
    HORIZON    = 1        # steps ahead to predict
    EPOCHS     = 1

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
    print(f"Total Features: {NUM_FEATURES}", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on {device}", flush=True)

    # Graph 
    adj_mx = get_adjacency_matrix(DIST_FILE, N_NODES, epsilon=0.5)

    # Data 
    print("Loading EMS data...", flush=True)
    (X_train, Y_train), (X_val, Y_val), (X_test, Y_test), (mean_z, std_z) = load_ems_data(
        csv_path          = DATA_FILE,
        z_score_cols      = Z_SCORE_COLS,
        pass_through_cols = PASS_THROUGH_COLS,
        history_window    = M_HISTORY,
        prediction_horizon= HORIZON,
        train_ratio       = 0.6,
        val_ratio         = 0.2,
    )
    # X shape: (B, T, N, F)   Y shape: (B, horizon, N)
    print(f"  X_train: {X_train.shape}  Y_train: {Y_train.shape}", flush=True)

    #Hyperparameters 
    best_params = {
        'lr'           : 0.001,
        'batch_size'   : 32,
        'num_units'    : 64,          # RNN hidden size
        'num_layers'   : 2,
        'max_diff_step': 2,
        'filter_type'  : 'dual_random_walk',
        'optimizer'    : 'Adam',
        'step_size'    : 7,
    }

    train_loader = DataLoader(
        TensorDataset(X_train, Y_train), batch_size=best_params["batch_size"], shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(X_val, Y_val), batch_size=best_params["batch_size"], shuffle=False
    )
    test_loader = DataLoader(
        TensorDataset(X_test, Y_test), batch_size=50, shuffle=False
    )

    # Model
    model = DCRNNModel(
        adj_mx              = adj_mx,
        input_dim           = NUM_FEATURES,
        output_dim          = 1,
        num_units           = best_params["num_units"],
        num_rnn_layers      = best_params["num_layers"],
        max_diffusion_step  = best_params["max_diff_step"],
        horizon             = HORIZON,
        filter_type         = best_params["filter_type"],
        cl_decay_steps      = 1000,
        use_curriculum_learning = True,
    ).to(device)

    optimizer = getattr(optim, best_params["optimizer"])(
        model.parameters(), lr=best_params["lr"]
    )
    scheduler  = optim.lr_scheduler.StepLR(
        optimizer, step_size=best_params["step_size"], gamma=0.7
    )
    loss_fn    = nn.MSELoss()

    # Training loop
    print("\n" + "="*60, flush=True)
    print("TRAINING DCRNN", flush=True)
    print("="*60, flush=True)

    best_val_loss = float('inf')
    train_losses, val_losses = [], []
    batches_seen = 0

    print("Starting training...", flush=True)
    for epoch in range(EPOCHS):
        model.train()
        train_loss_sum, train_count = 0.0, 0
        last_printed = 0

        # Reset peak memory stats before training starts
        use_cuda = device.type == "cuda"
        if use_cuda:
            torch.cuda.reset_peak_memory_stats(device)

        print('batch loading', flush=True)
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)     # (B, T, N, F)
            batch_y = batch_y.to(device)     # (B, horizon, N)

            optimizer.zero_grad()
            pred = model(batch_x, labels=batch_y, batches_seen=batches_seen)
            # pred: (B, horizon, N)
            loss = loss_fn(pred, batch_y)
            loss.backward()
            # gradient clipping 
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss_sum += loss.item() * batch_x.size(0)
            train_count    += batch_x.size(0)
            batches_seen   += 1

        epoch_train_loss = train_loss_sum / train_count
        train_losses.append(epoch_train_loss)

        # Validation
        model.eval()
        val_loss_sum, val_count = 0.0, 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                pred = model(batch_x)
                val_loss_sum += loss_fn(pred, batch_y).item() * batch_x.size(0)
                val_count    += batch_x.size(0)

        epoch_val_loss = val_loss_sum / val_count
        val_losses.append(epoch_val_loss)
        scheduler.step()

        print(f"DCRNN Epoch {epoch+1}/{EPOCHS} | "
              f"Train: {epoch_train_loss:.4f} | Val: {epoch_val_loss:.4f}", flush=True)

    # Capture peak memory after training completes

    peak_train_memory_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)

    # Profiling report  (parameters + inference time + memory)
    print_profiling_report(model, test_loader, device, peak_train_memory_mb)

    #Test evaluatio
    target_mean = float(mean_z[0, 0, 0])
    target_std  = float(std_z[0, 0, 0])

    mae, rmse, r2, preds_d, targs_d = evaluate(
        model, test_loader, device, target_mean, target_std
    )

    print("\nDCRNN FINAL RESULTS")
    print("="*60)
    print(f"MAE:  {mae:.4f}",      flush=True)
    print(f"RMSE: {rmse:.4f}",     flush=True)
    print(f"MSE:  {rmse**2:.4f}",  flush=True)
    print(f"R²:   {r2:.4f}",       flush=True)
