


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt

# ==========================================
# PART 1: GRAPH PROCESSING
# ==========================================

def get_chebyshev_laplacian(distance_file, n_nodes, k_cheb):
    """
    Loads distance matrix and converts to Scaled Laplacian for Chebyshev Conv.
    Includes Gaussian Kernel and Normalized Laplacian steps.
    """
    print(f"Loading graph from {distance_file}...")
    try:
        dist_matrix = np.load(distance_file)
    except FileNotFoundError:
        print(f"ERROR: File {distance_file} not found. Using random fallback.")
        dist_matrix = np.random.rand(n_nodes, n_nodes)

    if dist_matrix.shape[0] != n_nodes:
        raise ValueError(f"Distance matrix shape {dist_matrix.shape} does not match N_NODES={n_nodes}")

    # 1. Gaussian Kernel (Distance -> Weight)
    valid_dists = dist_matrix[dist_matrix > 0]
    sigma = np.std(valid_dists) if len(valid_dists) > 0 else 1.0
    epsilon = 0.5

    W = np.exp(- (dist_matrix**2) / (sigma**2))
    W[W < epsilon] = 0
    np.fill_diagonal(W, 0)

    # 2. Normalized Laplacian
    D = np.array(np.sum(W, axis=1))
    D_inv_sqrt = np.power(D, -0.5)
    D_inv_sqrt[np.isinf(D_inv_sqrt)] = 0.
    D_mat_inv_sqrt = np.diag(D_inv_sqrt)

    L = np.eye(n_nodes) - np.dot(np.dot(D_mat_inv_sqrt, W), D_mat_inv_sqrt)

    # 3. Scaled Laplacian (for Chebyshev stability)
    lambda_max = 2.0
    L_tilde = (2 * L) / lambda_max - np.eye(n_nodes)

    return torch.from_numpy(L_tilde.astype(np.float32))

# ==========================================
# PART 2: DATA LOADING (AUTO-GENERATES MISSING FEATURES)
# ==========================================

def load_ems_data(csv_path, z_score_cols, pass_through_cols, history_window, prediction_horizon=1, train_ratio=0.6, val_ratio=0.2):
    """
    Loads EMS data. 
    AUTOMATICALLY GENERATES missing cyclical time features (sin/cos) from 'datetime'.
    Normalizes Z-Score features ONLY. Leaves Pass-Through features untouched.
    """
    # 1. Load Data
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"❌ Error: The file '{csv_path}' was not found.")

    if 'datetime' in df.columns:
        df['datetime'] = pd.to_datetime(df['datetime'])
        df = df.sort_values(['datetime', 'ZIPCODE'])
    else:
        raise ValueError("DataFrame must contain 'datetime' column")

    # ==========================================
    # ✅ FIX: AUTO-GENERATE CYCLICAL FEATURES
    # ==========================================
    # This block creates dayofweek_sin, month_cos, etc. from the date
    # so they don't need to exist in the CSV file.
    dt_ref = df['datetime'].dt
    
    time_feats_config = {
        'hour': 24,
        'dayofweek': 7,
        'month': 12,
        'weekofyear': 52
    }

    print("Checking for missing time features...")
    for name, period in time_feats_config.items():
        sin_name = f"{name}_sin"
        cos_name = f"{name}_cos"
        
        # Only generate if requested in pass_through_cols AND missing from DF
        if (sin_name in pass_through_cols) and (sin_name not in df.columns):
            print(f"  -> Generating {sin_name} & {cos_name}...")
            
            # Extract raw integer (e.g. 0-23 for hour)
            if name == 'weekofyear':
                # weekofyear is specific to isocalendar in newer pandas
                raw_val = dt_ref.isocalendar().week.astype(int)
            else:
                raw_val = getattr(dt_ref, name)

            # Apply Sin/Cos transform
            df[sin_name] = np.sin(2 * np.pi * raw_val / period)
            df[cos_name] = np.cos(2 * np.pi * raw_val / period)

    # ==========================================
    # 2. Verify Columns Again
    # ==========================================
    feature_cols = z_score_cols + pass_through_cols
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"❌ Missing columns in CSV (could not auto-generate): {missing}")

    # ==========================================
    # 3. Pivot & Process
    # ==========================================
    data_list = []
    min_time_steps = float('inf')

    # Pass 1: Find min length
    for feat in feature_cols:
        pivot_df = df.pivot_table(index='datetime', columns='ZIPCODE', values=feat, aggfunc='mean')
        min_time_steps = min(min_time_steps, len(pivot_df))

    # Pass 2: Pivot, Fill, Trim
    for feat in feature_cols:
        pivot_df = df.pivot_table(index='datetime', columns='ZIPCODE', values=feat, aggfunc='mean')
        pivot_df = pivot_df.interpolate(method='linear', axis=0).fillna(0)
        pivot_df = pivot_df.reindex(sorted(pivot_df.columns), axis=1)
        pivot_df = pivot_df.iloc[:min_time_steps, :] 
        data_list.append(pivot_df.values)

    # Stack: (Time, Nodes, Total_Features)
    data_raw = np.stack(data_list, axis=-1) 

    # 4. Split Raw Data
    n_total = data_raw.shape[0]
    train_end = int(n_total * train_ratio)
    val_end = int(n_total * (train_ratio + val_ratio))

    train_data = data_raw[:train_end]
    val_data = data_raw[train_end:val_end]
    test_data = data_raw[val_end:]

    # 5. Hybrid Normalization
    num_z_cols = len(z_score_cols)
    
    # Calculate stats ONLY on Train, ONLY on Z-Score columns
    # axis=(0, 1) means Global Scaling (across Time and Nodes)
    train_z = train_data[..., :num_z_cols]
    
    mean_z = np.mean(train_z, axis=(0, 1), keepdims=True)
    std_z = np.std(train_z, axis=(0, 1), keepdims=True)
    std_z[std_z < 1e-5] = 1.0

    def normalize_split(data_arr):
        part_z = data_arr[..., :num_z_cols]
        part_p = data_arr[..., num_z_cols:]
        part_z_norm = (part_z - mean_z) / std_z
        return np.concatenate([part_z_norm, part_p], axis=-1)

    train_norm = normalize_split(train_data)
    val_norm = normalize_split(val_data)
    test_norm = normalize_split(test_data)

    print(f"\n✓ Normalization Logic Applied:")
    print(f"  - Z-Scored {num_z_cols} features")
    print(f"  - Passed Through {len(pass_through_cols)} features")
    print(f"  - Target (call_count) Mean: {mean_z[0,0,0]:.4f}, Std: {std_z[0,0,0]:.4f}")

    # 6. Sliding Windows
    def create_windows(data, history, horizon):
        X, Y = [], []
        num_samples = len(data) - history - horizon + 1
        
        if num_samples <= 0:
            return np.empty((0, history, data.shape[1])), np.empty((0, horizon))

        for i in range(num_samples):
            # Input: (Time, Nodes, Feat) -> Transpose to (Feat, Time, Nodes)
            x_inst = data[i : i + history].transpose(2, 0, 1)
            # Target: call_count is at index 0
            y_inst = data[i + history : i + history + horizon, :, 0]
            X.append(x_inst)
            Y.append(y_inst)
        return np.array(X), np.array(Y)

    print(f"\n✓ Creating sliding windows...")
    X_train, Y_train = create_windows(train_norm, history_window, prediction_horizon)
    X_val, Y_val = create_windows(val_norm, history_window, prediction_horizon)
    X_test, Y_test = create_windows(test_norm, history_window, prediction_horizon)

    # Convert to Tensors
    X_train_t, Y_train_t = torch.Tensor(X_train), torch.Tensor(Y_train)
    X_val_t, Y_val_t     = torch.Tensor(X_val), torch.Tensor(Y_val)
    X_test_t, Y_test_t   = torch.Tensor(X_test), torch.Tensor(Y_test)

    return (X_train_t, Y_train_t), (X_val_t, Y_val_t), (X_test_t, Y_test_t), (mean_z, std_z)


# ==========================================
# PART 3: MODEL ARCHITECTURE
# ==========================================

class TemporalConvLayer(nn.Module):
    def __init__(self, Kt, c_in, c_out):
        super(TemporalConvLayer, self).__init__()
        self.Kt = Kt
        self.c_out = c_out
        self.align = nn.Conv2d(c_in, c_out, kernel_size=(1, 1))
        self.conv = nn.Conv2d(c_in, 2 * c_out, kernel_size=(Kt, 1))
        self.layer_norm = nn.LayerNorm(c_out)

    def forward(self, x):
        x_in = self.align(x)
        # Causal padding
        x_padded = F.pad(x, (0, 0, self.Kt - 1, 0))
        out = self.conv(x_padded)
        P, Q = torch.chunk(out, 2, dim=1)
        out = (P + x_in) * torch.sigmoid(Q) # Gated TCN
        out = out.permute(0, 2, 3, 1)
        out = self.layer_norm(out)
        out = out.permute(0, 3, 1, 2)
        return out

class ChebGraphConv(nn.Module):
    def __init__(self, c_in, c_out, Ks, L_tilde):
        super(ChebGraphConv, self).__init__()
        self.Ks = Ks
        self.L_tilde = L_tilde
        self.Theta = nn.Parameter(torch.FloatTensor(Ks, c_in, c_out))
        self.bias = nn.Parameter(torch.FloatTensor(c_out))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.Theta, a=np.sqrt(5))
        nn.init.zeros_(self.bias)

    def forward(self, x):
        B, C, T, N = x.shape
        x_flat = x.permute(0, 2, 3, 1).contiguous().view(B * T, N, C)
        x_0 = x_flat
        x_1 = torch.matmul(self.L_tilde.to(x.device), x_flat)
        cheb_list = [x_0, x_1]
        for k in range(2, self.Ks):
            x_k = 2 * torch.matmul(self.L_tilde.to(x.device), cheb_list[-1]) - cheb_list[-2]
            cheb_list.append(x_k)
        cheb_stacked = torch.stack(cheb_list, dim=0)
        out = torch.einsum('kbni,kio->bno', cheb_stacked, self.Theta) + self.bias
        return F.relu(out.view(B, T, N, -1).permute(0, 3, 1, 2))

class STGCN(nn.Module):
    def __init__(self, Ks, Kt, n_nodes, blocks, L_tilde):
        super(STGCN, self).__init__()
        self.layers = nn.ModuleList()
        # Sandwich: TCN -> GCN -> TCN
        for channels in blocks:
            c_in, c_h, c_out = channels
            self.layers.append(nn.Sequential(
                TemporalConvLayer(Kt, c_in, c_h),
                ChebGraphConv(c_h, c_h, Ks, L_tilde),
                TemporalConvLayer(Kt, c_h, c_out)
            ))
        last_c = blocks[-1][-1]
        self.output = nn.Sequential(
            TemporalConvLayer(Kt, last_c, last_c),
            nn.Conv2d(last_c, 1, kernel_size=(1, 1))
        )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.output(x)[:, :, -1, :]

# ==========================================
# PART 4: EVALUATION & PLOTTING
# ==========================================

def evaluate(model, loader, device, mean, std):
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            pred = model(batch_x)
            all_preds.append(pred.squeeze(1).cpu().numpy())
            all_targets.append(batch_y.squeeze(1).cpu().numpy())
            
    preds = np.concatenate(all_preds, axis=0)
    targs = np.concatenate(all_targets, axis=0)
    
    # Denormalize
    preds_denorm = preds * std + mean
    targs_denorm = targs * std + mean
    
    mae = np.mean(np.abs(preds_denorm - targs_denorm))
    rmse = np.sqrt(np.mean((preds_denorm - targs_denorm) ** 2))
    
    non_zero_mask = targs_denorm != 0
    mape = np.mean(np.abs((preds_denorm[non_zero_mask] - targs_denorm[non_zero_mask]) / targs_denorm[non_zero_mask])) * 100
    
    ss_res = np.sum((targs_denorm - preds_denorm) ** 2)
    ss_tot = np.sum((targs_denorm - np.mean(targs_denorm)) ** 2)
    r2 = 1 - (ss_res / (ss_tot + 1e-8))
    
    return mae, rmse, mape, r2, preds_denorm, targs_denorm

def plot_training_curves(train_losses, val_losses):
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label='Train Loss', linewidth=2)
    plt.plot(val_losses, label='Val Loss', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('STGCN Training & Validation Loss')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig('stgcn_train_val_loss_STGCN_14.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_predictions_samples(preds_denorm, targs_denorm, n_samples=4):
    num_samples = preds_denorm.shape[0]
    n_samples = min(n_samples, num_samples)
    idxs = np.random.choice(num_samples, n_samples, replace=False)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()
    for ax, idx in zip(axes, idxs):
        pred = preds_denorm[idx, :]
        true = targs_denorm[idx, :]
        nodes = np.arange(len(pred))
        ax.plot(nodes, true, 'k-', label='True', linewidth=1.5)
        ax.plot(nodes, pred, 'r--', label='Pred', linewidth=1.5)
        ax.set_title(f'Sample {idx}')
        ax.set_xlabel('Node index')
        ax.set_ylabel('Call count')
        ax.grid(True, alpha=0.3)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right')
    plt.tight_layout()
    plt.savefig('stgcn_pred_samples_STGCN_14.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_error_histogram(preds_denorm, targs_denorm):
    errors = np.abs(preds_denorm - targs_denorm).flatten()
    plt.figure(figsize=(8, 5))
    plt.hist(errors, bins=50, color='steelblue', edgecolor='black', alpha=0.8)
    plt.xlabel('Absolute Error (call count)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Prediction Errors')
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig('stgcn_error_hist_STGCN_14.png', dpi=300, bbox_inches='tight')
    plt.close()

# ==========================================
# PART 5: EXECUTION 
# ==========================================

if __name__ == "__main__":
    # --- STATIC CONFIG ---
    DIST_FILE = 'distance_matrix (1).npy'
    DATA_FILE = 'All_Features_3.csv'
    N_NODES = 197
    M_HISTORY = 24 
    KS = 3          
    KT = 4
    EPOCHS = 50 # Keep at 50 for the final train; Optuna will use this but prune bad trials early.

    # --- FEATURE CONFIG ---
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
    print(f"Total Features Configured: {NUM_FEATURES}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on {device}")

    # 1. Process Graph
    L_tilde = get_chebyshev_laplacian(DIST_FILE, N_NODES, KS).to(device)

    # 2. Load Data (Done ONCE outside the search loop to save massive amounts of time)
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

    test_dataset = TensorDataset(X_test, Y_test)
    # Batch size for test loader doesn't affect training, so we fix it
    test_loader = DataLoader(test_dataset, batch_size=50, shuffle=False)

    # ==========================================
    # RETRAINING FINAL MODEL WITH BEST PARAMS
    # ==========================================
    print("\n" + "="*60)
    print("RETRAINING FINAL MODEL WITH OPTIMIZED PARAMETERS")
    print("="*60)
    
    best_params = {
        'lr': 0.0020861235735065513, 
        'batch_size': 16, 
        'hidden_dim': 64, 
        'out_dim': 32, 
        'optimizer': 'AdamW', 
        'step_size': 7
    }
    
    best_train_loader = DataLoader(TensorDataset(X_train, Y_train), batch_size=best_params["batch_size"], shuffle=True)
    best_val_loader = DataLoader(TensorDataset(X_val, Y_val), batch_size=best_params["batch_size"], shuffle=False)
    
    final_blocks = [
        [NUM_FEATURES, best_params["hidden_dim"], best_params["out_dim"]], 
        [best_params["out_dim"], best_params["hidden_dim"], best_params["out_dim"]],
        [best_params["out_dim"], best_params["hidden_dim"], best_params["out_dim"]]
    ]
    final_model = STGCN(KS, KT, N_NODES, final_blocks, L_tilde).to(device)
    
    final_optimizer = getattr(optim, best_params["optimizer"])(final_model.parameters(), lr=best_params["lr"])
    final_scheduler = optim.lr_scheduler.StepLR(final_optimizer, step_size=best_params["step_size"], gamma=0.7)
    final_loss_fn = nn.MSELoss()

    best_final_val_loss = float('inf')
    train_losses, val_losses = [], []

    for epoch in range(EPOCHS):
        final_model.train()
        train_loss_sum = 0.0
        train_count = 0
        for batch_x, batch_y in best_train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            final_optimizer.zero_grad()
            pred = final_model(batch_x)
            loss = final_loss_fn(pred.squeeze(1), batch_y.squeeze(1))
            loss.backward()
            final_optimizer.step()
            train_loss_sum += loss.item() * batch_x.size(0)
            train_count += batch_x.size(0)
            
        epoch_train_loss = train_loss_sum / train_count
        train_losses.append(epoch_train_loss)

        final_model.eval()
        val_loss_sum = 0.0
        val_count = 0
        with torch.no_grad():
            for batch_x, batch_y in best_val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                pred = final_model(batch_x)
                loss = final_loss_fn(pred.squeeze(1), batch_y.squeeze(1))
                val_loss_sum += loss.item() * batch_x.size(0)
                val_count += batch_x.size(0)
                
        epoch_val_loss = val_loss_sum / val_count
        val_losses.append(epoch_val_loss)
        
        final_scheduler.step()
        
        if epoch_val_loss < best_final_val_loss:
            best_final_val_loss = epoch_val_loss
            torch.save(final_model.state_dict(), "stgcn_ems_best_optuna14.pth")
            
        if (epoch + 1) % 10 == 0:
            print(f"Final Retrain Epoch {epoch+1}/{EPOCHS} | Train: {epoch_train_loss:.4f} | Val: {epoch_val_loss:.4f}")

    # ==========================================
    # FINAL EVALUATION & PLOTTING
    # ==========================================
    print("\n" + "="*60)
    print("EVALUATING ON TEST SET")
    print("="*60)
    final_model.load_state_dict(torch.load("stgcn_ems_best_optuna14.pth", map_location=device))
    target_mean = mean_z[0, 0, 0]
    target_std = std_z[0, 0, 0]
    
    mae, rmse, mape, r2, preds_denorm, targs_denorm = evaluate(
        final_model, test_loader, device, target_mean, target_std
    )
    
    print("\nFINAL RESULTS (OPTIMIZED)")
    print("="*60)
    print(f"MAE:  {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MSE:  {rmse**2:.4f}")
    print(f"MAPE: {mape:.2f}%")
    print(f"R²:   {r2:.4f}")

    plot_training_curves(train_losses, val_losses)
    plot_predictions_samples(preds_denorm, targs_denorm, n_samples=4)
    plot_error_histogram(preds_denorm, targs_denorm)
    print("\nGenerated plots: stgcn_train_val_loss_STGCN_14.png, stgcn_pred_samples_STGCN_14.png, stgcn_error_hist_STGCN_14.png")
