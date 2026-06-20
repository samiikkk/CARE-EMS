import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
import time
import gc
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt

# ==========================================
# ABLATION STUDY: PURE SPATIAL GCN BASELINE
# ==========================================
# Purpose  : Evaluate the predictive power of a purely spatial graph.
# Method   : Receives a 24-hour sequence but isolates ONLY the most recent 
#            time step (t-1) to predict (t). This entirely removes sequential 
#            temporal tracking, acting as a strict spatial-only baseline.
# ==========================================


# ==========================================
# PART 0: VERIFICATION UTILITIES
# ==========================================

def verify_rolling_features(csv_path, check_rows=5):
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print("[verify_rolling] File not found — skipping check.")
        return

    required = {'datetime', 'ZIPCODE', 'call_count', 'roll_mean_24'}

    if not required.issubset(df.columns):
        print("[verify_rolling] Missing required columns — skipping check.")
        return

    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.sort_values(['ZIPCODE', 'datetime'])

    sample_zip = df['ZIPCODE'].unique()[0]
    sub = df[df['ZIPCODE'] == sample_zip].copy().reset_index(drop=True)

    recomputed = sub['call_count'].rolling(
        window=24,
        min_periods=1
    ).mean().shift(1)

    compare = pd.DataFrame({
        'csv_value': sub['roll_mean_24'].values,
        'trailing_window': recomputed.values
    }).dropna().head(check_rows)

    max_diff = np.abs(compare['csv_value'] - compare['trailing_window']).max()

    if max_diff < 1e-3:
        print(f"✓ roll_mean_24 verified as TRAILING window (ZIP {sample_zip}).")
    else:
        print(f"⚠️  roll_mean_24 mismatch for ZIP {sample_zip}. Max diff: {max_diff:.4f}")


def verify_node_order(saved_order_path, dist_matrix_path, n_nodes):
    try:
        node_order = np.load(saved_order_path, allow_pickle=True)
        print(
            f"✓ Node order loaded: {len(node_order)} ZIPs, "
            f"first={node_order[0]}, last={node_order[-1]}"
        )
    except FileNotFoundError:
        print(f"[verify_node_order] {saved_order_path} not found.")
        return

    try:
        dist = np.load(dist_matrix_path)

        if dist.shape[0] != n_nodes:
            print(f"⚠️  Distance matrix {dist.shape} != N_NODES={n_nodes}.")
        else:
            print(f"✓ Distance matrix shape {dist.shape} matches N_NODES={n_nodes}.")
    except FileNotFoundError:
        print("[verify_node_order] Distance matrix not found — skipping shape check.")


# ==========================================
# PART 0B: MODEL-ONLY INFERENCE HELPERS
# ==========================================

def sync_if_cuda(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    param_size_mb = total_params * 4 / (1024 ** 2)

    return total_params, trainable_params, param_size_mb


def measure_model_only_inference_time(model, loader, device, warmup_steps=20, repeats=10):
    """
    Measures model-only forward-pass inference time.

    Excludes:
    - CSV loading
    - preprocessing
    - DataLoader iteration during timing
    - CPU-to-GPU transfer during timing
    - metric computation
    - plotting

    Measures only:
    - pred = model(batch_x)
    """

    model.eval()

    # Move all test batches to device BEFORE timing.
    test_batches_device = []

    with torch.no_grad():
        for batch_x, _ in loader:
            test_batches_device.append(batch_x.to(device))

    if len(test_batches_device) == 0:
        raise ValueError("Test loader is empty. Cannot measure inference time.")

    # Warmup
    with torch.no_grad():
        for i in range(warmup_steps):
            batch_x = test_batches_device[i % len(test_batches_device)]
            _ = model(batch_x)

    sync_if_cuda(device)

    start = time.perf_counter()

    with torch.no_grad():
        for _ in range(repeats):
            for batch_x in test_batches_device:
                _ = model(batch_x)

    sync_if_cuda(device)

    end = time.perf_counter()

    total_time = end - start
    total_batches = len(test_batches_device) * repeats
    total_samples = sum(batch_x.size(0) for batch_x in test_batches_device) * repeats

    ms_per_batch = (total_time / total_batches) * 1000
    ms_per_sample = (total_time / total_samples) * 1000
    sec_per_full_test_pass = total_time / repeats

    del test_batches_device
    gc.collect()

    if device.type == "cuda":
        torch.cuda.empty_cache()

    return ms_per_batch, ms_per_sample, sec_per_full_test_pass


def measure_model_only_peak_memory(model, loader, device):
    """
    Measures model-only peak memory during inference.

    For CUDA, the reported memory includes:
    - model parameters
    - graph/Laplacian buffer
    - one input batch
    - forward-pass intermediate tensors

    It excludes:
    - optimizer state, after optimizer is deleted before calling this function
    - training/backpropagation graph
    """

    model.eval()

    if device.type == "cuda":
        peak_mem_mb = 0.0

        with torch.no_grad():
            for batch_x, _ in loader:
                batch_x = batch_x.to(device)

                sync_if_cuda(device)
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(device)

                pred = model(batch_x)

                sync_if_cuda(device)

                batch_peak_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
                peak_mem_mb = max(peak_mem_mb, batch_peak_mb)

                del pred
                del batch_x

                gc.collect()
                torch.cuda.empty_cache()

        return peak_mem_mb

    else:
        import tracemalloc

        peak_mem_mb = 0.0

        with torch.no_grad():
            for batch_x, _ in loader:
                batch_x = batch_x.to(device)

                tracemalloc.start()

                pred = model(batch_x)

                _, peak = tracemalloc.get_traced_memory()
                tracemalloc.stop()

                batch_peak_mb = peak / (1024 ** 2)
                peak_mem_mb = max(peak_mem_mb, batch_peak_mb)

                del pred
                del batch_x

                gc.collect()

        return peak_mem_mb


# ==========================================
# PART 1: GRAPH PROCESSING
# ==========================================

def get_chebyshev_laplacian(distance_file, n_nodes, k_cheb):
    print(f"Loading graph from {distance_file}...")

    try:
        dist_matrix = np.load(distance_file)
    except FileNotFoundError:
        print(f"WARNING: {distance_file} not found. Using symmetric random fallback.")
        _rand = np.random.rand(n_nodes, n_nodes)
        dist_matrix = (_rand + _rand.T) / 2
        np.fill_diagonal(dist_matrix, 0)

    if dist_matrix.shape[0] != n_nodes:
        raise ValueError(f"Distance matrix shape {dist_matrix.shape} != N_NODES={n_nodes}")

    valid_dists = dist_matrix[dist_matrix > 0]
    sigma = np.std(valid_dists) if len(valid_dists) > 0 else 1.0

    W = np.exp(-(dist_matrix ** 2) / (sigma ** 2))
    W[W < 0.5] = 0
    np.fill_diagonal(W, 0)

    D = np.array(np.sum(W, axis=1))
    D_inv_sqrt = np.power(D, -0.5)
    D_inv_sqrt[np.isinf(D_inv_sqrt)] = 0.0

    D_mat = np.diag(D_inv_sqrt)
    L = np.eye(n_nodes) - D_mat @ W @ D_mat

    lambda_max = 2.0
    L_tilde = (2.0 / lambda_max) * L - np.eye(n_nodes)

    return torch.from_numpy(L_tilde.astype(np.float32))


# ==========================================
# PART 2: DATA LOADING
# ==========================================

def _ffill_numpy(arr):
    out = arr.copy()

    for node in range(arr.shape[1]):
        for feat in range(arr.shape[2]):
            col = out[:, node, feat]
            mask = np.isnan(col)

            if not mask.any():
                continue

            idx = np.where(~mask, np.arange(len(col)), 0)
            np.maximum.accumulate(idx, out=idx)
            col[mask] = col[idx[mask]]
            out[:, node, feat] = col

    return np.where(np.isnan(out), 0.0, out)


def load_ems_data(
    csv_path,
    z_score_cols,
    pass_through_cols,
    history_window,
    prediction_horizon=1,
    train_ratio=0.6,
    val_ratio=0.2,
    node_order_save_path='gcn_fair_node_order.npy'
):
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"File '{csv_path}' not found.")

    if 'datetime' not in df.columns:
        raise ValueError("DataFrame must contain 'datetime' column.")

    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.sort_values(['datetime', 'ZIPCODE'])

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

        if (sin_name in pass_through_cols or cos_name in pass_through_cols) and (
            sin_name not in df.columns or cos_name not in df.columns
        ):
            raw_val = (
                dt_ref.isocalendar().week.astype(int)
                if name == 'weekofyear'
                else getattr(dt_ref, name)
            )

            df[sin_name] = np.sin(2 * np.pi * raw_val / period)
            df[cos_name] = np.cos(2 * np.pi * raw_val / period)

    feature_cols = z_score_cols + pass_through_cols

    missing = [c for c in feature_cols if c not in df.columns]

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    print("Pivoting features (single pass)...")

    pivots = []
    min_time_steps = float('inf')
    zip_order = None

    for feat in feature_cols:
        pv = df.pivot_table(
            index='datetime',
            columns='ZIPCODE',
            values=feat,
            aggfunc='mean'
        )

        pv = pv.reindex(sorted(pv.columns), axis=1)

        if zip_order is None:
            zip_order = np.array(sorted(pv.columns))

        pivots.append(pv)
        min_time_steps = min(min_time_steps, len(pv))

    np.save(node_order_save_path, zip_order)

    data_list = [pv.iloc[:min_time_steps, :].values for pv in pivots]
    data_raw = np.stack(data_list, axis=-1)

    n_total = data_raw.shape[0]
    train_end = int(n_total * train_ratio)
    val_end = int(n_total * (train_ratio + val_ratio))

    train_data = data_raw[:train_end].copy()
    val_data = data_raw[train_end:val_end].copy()
    test_data = data_raw[val_end:].copy()

    train_data = _ffill_numpy(train_data)
    val_data = _ffill_numpy(val_data)
    test_data = _ffill_numpy(test_data)

    num_z = len(z_score_cols)

    mean_z = np.mean(train_data[..., :num_z], axis=(0, 1), keepdims=True)
    std_z = np.std(train_data[..., :num_z], axis=(0, 1), keepdims=True)
    std_z[std_z < 1e-5] = 1.0

    def normalize(arr):
        z = (arr[..., :num_z] - mean_z) / std_z
        return np.concatenate([z, arr[..., num_z:]], axis=-1)

    train_norm = normalize(train_data)
    val_norm = normalize(val_data)
    test_norm = normalize(test_data)

    print("\nNormalization Logic Applied:")
    print(f"  - Z-scored features: {num_z}")
    print(f"  - Pass-through features: {len(pass_through_cols)}")
    print(f"  - Target call_count mean: {mean_z[0, 0, 0]:.4f}")
    print(f"  - Target call_count std:  {std_z[0, 0, 0]:.4f}")

    def create_windows(data, history, horizon):
        X, Y = [], []
        n_samps = len(data) - history - horizon + 1

        if n_samps <= 0:
            return np.empty((0,)), np.empty((0,))

        for i in range(n_samps):
            X.append(data[i:i + history].transpose(2, 0, 1))
            Y.append(data[i + history:i + history + horizon, :, 0])

        return np.array(X), np.array(Y)

    print("\nCreating sliding windows...")

    X_tr, Y_tr = create_windows(train_norm, history_window, prediction_horizon)
    X_va, Y_va = create_windows(val_norm, history_window, prediction_horizon)
    X_te, Y_te = create_windows(test_norm, history_window, prediction_horizon)

    print(f"Train X: {X_tr.shape}, Train Y: {Y_tr.shape}")
    print(f"Val   X: {X_va.shape}, Val   Y: {Y_va.shape}")
    print(f"Test  X: {X_te.shape}, Test  Y: {Y_te.shape}")

    to_t = lambda a: torch.tensor(a, dtype=torch.float32)

    return (
        (to_t(X_tr), to_t(Y_tr)),
        (to_t(X_va), to_t(Y_va)),
        (to_t(X_te), to_t(Y_te)),
        (mean_z, std_z)
    )


# ==========================================
# PART 3: PURE SPATIAL ARCHITECTURE
# ==========================================

class ChebGraphConv(nn.Module):
    def __init__(self, c_in, c_out, Ks, L_tilde):
        super().__init__()

        self.Ks = Ks

        self.register_buffer('L_tilde', L_tilde)

        self.Theta = nn.Parameter(torch.FloatTensor(Ks, c_in, c_out))
        self.bias = nn.Parameter(torch.zeros(c_out))

        nn.init.kaiming_uniform_(self.Theta, a=np.sqrt(5))

    def forward(self, x):
        L = self.L_tilde

        x0 = x
        x1 = torch.matmul(L, x)

        cheb = [x0, x1]

        for _ in range(2, self.Ks):
            cheb.append(2 * torch.matmul(L, cheb[-1]) - cheb[-2])

        cheb_stack = torch.stack(cheb, dim=0)

        out = torch.einsum('kbni,kio->bno', cheb_stack, self.Theta) + self.bias

        return F.relu(out)


class PureSpatialGCN(nn.Module):
    def __init__(self, num_features, hidden_dim, Ks, L_tilde, dropout=0.1):
        super().__init__()

        self.input_proj = nn.Sequential(
            nn.Linear(num_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.conv = ChebGraphConv(hidden_dim, hidden_dim, Ks, L_tilde)

        self.readout = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, x):
        # x shape: (B, F, T, N)
        # Isolate most recent timestep — spatial-only baseline
        h = x[:, :, -1, :]                  # (B, F, N)
        h = h.permute(0, 2, 1).contiguous() # (B, N, F)

        h = self.input_proj(h)              # (B, N, hidden_dim)
        h = self.conv(h)                    # (B, N, hidden_dim)

        return self.readout(h).squeeze(-1)  # (B, N)


# ==========================================
# PART 4: EVALUATION & PLOTTING
# ==========================================

def evaluate(model, loader, device, mean, std):
    model.eval()

    preds_list = []
    targs_list = []

    with torch.no_grad():
        for bx, by in loader:
            preds_list.append(model(bx.to(device)).cpu().numpy())
            targs_list.append(by.squeeze(1).cpu().numpy())

    preds = np.concatenate(preds_list)
    targs = np.concatenate(targs_list)

    preds_d = preds * std + mean
    targs_d = targs * std + mean

    mae = np.mean(np.abs(preds_d - targs_d))
    rmse = np.sqrt(np.mean((preds_d - targs_d) ** 2))

    nz = targs_d != 0

    if np.any(nz):
        mape = np.mean(np.abs((preds_d[nz] - targs_d[nz]) / targs_d[nz])) * 100
    else:
        mape = np.nan

    ss_res = np.sum((targs_d - preds_d) ** 2)
    ss_tot = np.sum((targs_d - np.mean(targs_d)) ** 2)
    r2 = 1 - ss_res / (ss_tot + 1e-8)

    return mae, rmse, mape, r2, preds_d, targs_d


def plot_training_curves(train_losses, val_losses):
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label='Train Loss', linewidth=2)
    plt.plot(val_losses, label='Val Loss', linewidth=2)

    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Pure Spatial GCN — Training & Validation Loss')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plt.savefig('pure_gcn_train_val_loss.png', dpi=300, bbox_inches='tight')
    plt.close()


# ==========================================
# PART 5: EXECUTION
# ==========================================

if __name__ == "__main__":

    DIST_FILE = 'distance_matrix (1).npy'
    DATA_FILE = 'All_Features_3.csv'

    N_NODES = 197
    M_HISTORY = 24
    EPOCHS = 15

    Z_SCORE_COLS = [
        'call_count',
        'lag_1',
        'lag_24',
        'lag_168',
        'roll_mean_24',
        'roll_std_24',
        'Population',
        'INITIAL_SEVERITY_LEVEL_CODE'
    ]

    PASS_THROUGH_COLS = [
        'hour_sin',
        'hour_cos',
        'dayofweek_sin',
        'dayofweek_cos',
        'month_sin',
        'month_cos',
        'weekofyear_sin',
        'weekofyear_cos',
        'is_Covid',
        'SPECIAL_EVENT_INDICATOR'
    ]

    NUM_FEATURES = len(Z_SCORE_COLS) + len(PASS_THROUGH_COLS)

    FAIR_GCN_PARAMS = {
        'lr': 0.002,
        'batch_size': 16,
        'hidden_dim': 64,
        'Ks': 3,
        'dropout': 0.1,
        'optimizer': 'AdamW',
        'step_size': 7,
        'gamma': 0.7
    }

    print("=" * 60)
    print("PURE SPATIAL GCN BASELINE (Current State Only)")
    print("=" * 60)
    print(f"Features      : {NUM_FEATURES}")
    print(f"Input dim     : {NUM_FEATURES} (Sliced from Sequence)")
    print(f"Hidden dim    : {FAIR_GCN_PARAMS['hidden_dim']}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device        : {device}")

    if device.type == 'cuda':
        print(f"GPU name      : {torch.cuda.get_device_name(0)}")
        print(f"CUDA version  : {torch.version.cuda}")
        torch.cuda.empty_cache()

    verify_rolling_features(DATA_FILE)

    # Build L_tilde on CPU first, then move to device cleanly
    L_tilde = get_chebyshev_laplacian(
        DIST_FILE,
        N_NODES,
        FAIR_GCN_PARAMS['Ks']
    )

    L_tilde = L_tilde.to(device)

    (X_tr, Y_tr), (X_va, Y_va), (X_te, Y_te), (mean_z, std_z) = load_ems_data(
        csv_path=DATA_FILE,
        z_score_cols=Z_SCORE_COLS,
        pass_through_cols=PASS_THROUGH_COLS,
        history_window=M_HISTORY,
        node_order_save_path='gcn_fair_node_order.npy'
    )

    train_loader = DataLoader(
        TensorDataset(X_tr, Y_tr),
        batch_size=FAIR_GCN_PARAMS['batch_size'],
        shuffle=True
    )

    val_loader = DataLoader(
        TensorDataset(X_va, Y_va),
        batch_size=FAIR_GCN_PARAMS['batch_size'],
        shuffle=False
    )

    TEST_BATCH_SIZE = 50

    test_loader = DataLoader(
        TensorDataset(X_te, Y_te),
        batch_size=TEST_BATCH_SIZE,
        shuffle=False
    )

    print(f"Test samples   : {len(test_loader.dataset)}")
    print(f"Test batches   : {len(test_loader)}")
    print(f"Test batch size: {TEST_BATCH_SIZE}")

    model = PureSpatialGCN(
        num_features=NUM_FEATURES,
        hidden_dim=FAIR_GCN_PARAMS['hidden_dim'],
        Ks=FAIR_GCN_PARAMS['Ks'],
        L_tilde=L_tilde,
        dropout=FAIR_GCN_PARAMS['dropout']
    ).to(device)

    optimizer = getattr(optim, FAIR_GCN_PARAMS['optimizer'])(
        model.parameters(),
        lr=FAIR_GCN_PARAMS['lr']
    )

    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=FAIR_GCN_PARAMS['step_size'],
        gamma=FAIR_GCN_PARAMS['gamma']
    )

    loss_fn = nn.MSELoss()

    total_params, trainable_params, param_size_mb = count_parameters(model)

    print(f"Total Parameters:     {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")
    print(f"Parameter Size:       {param_size_mb:.4f} MB assuming float32")

    print("\nTRAINING PURE SPATIAL GCN...")

    best_val_loss = float('inf')
    train_losses = []
    val_losses = []

    CHECKPOINT = 'pure_gcn_baseline_best.pth'

    for epoch in range(EPOCHS):
        model.train()

        tr_loss = 0.0
        tr_n = 0

        for bx, by in train_loader:
            bx = bx.to(device)
            by = by.to(device)

            optimizer.zero_grad(set_to_none=True)

            pred = model(bx)
            loss = loss_fn(pred, by.squeeze(1))

            loss.backward()
            optimizer.step()

            tr_loss += loss.item() * bx.size(0)
            tr_n += bx.size(0)

        epoch_tr = tr_loss / tr_n
        train_losses.append(epoch_tr)

        model.eval()

        va_loss = 0.0
        va_n = 0

        with torch.no_grad():
            for bx, by in val_loader:
                bx = bx.to(device)
                by = by.to(device)

                pred = model(bx)
                loss = loss_fn(pred, by.squeeze(1))

                va_loss += loss.item() * bx.size(0)
                va_n += bx.size(0)

        epoch_va = va_loss / va_n
        val_losses.append(epoch_va)

        scheduler.step()

        if epoch_va < best_val_loss:
            best_val_loss = epoch_va
            torch.save(model.state_dict(), CHECKPOINT)

        if (epoch + 1) % 10 == 0:
            print(
                f"Epoch {epoch + 1:3d}/{EPOCHS} | "
                f"Train: {epoch_tr:.4f} | "
                f"Val: {epoch_va:.4f}"
            )

    print("\n" + "=" * 60)
    print("EVALUATING ON TEST SET")
    print("=" * 60)

    model.load_state_dict(
        torch.load(CHECKPOINT, map_location=device)
    )

    model.eval()

    t_mean = float(mean_z[0, 0, 0])
    t_std = float(std_z[0, 0, 0])

    # Remove optimizer-related objects before inference memory measurement.
    # This prevents optimizer state from being counted in inference memory.
    try:
        del optimizer
        del scheduler
        del loss_fn
        del bx
        del by
        del pred
        del loss
    except NameError:
        pass

    gc.collect()

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    # ==========================================
    # MODEL-ONLY INFERENCE TIME
    # ==========================================
    model_only_ms_batch, model_only_ms_sample, sec_per_test_pass = measure_model_only_inference_time(
        model=model,
        loader=test_loader,
        device=device,
        warmup_steps=20,
        repeats=10
    )

    print("\nMODEL-ONLY INFERENCE TIME")
    print("-" * 60)
    print(f"Model-only Inference Time (ms per batch):  {model_only_ms_batch:.4f}")
    print(f"Model-only Inference Time (ms per sample): {model_only_ms_sample:.6f}")
    print(f"Seconds per full test pass:                {sec_per_test_pass:.4f}")

    # ==========================================
    # MODEL-ONLY INFERENCE MEMORY
    # ==========================================
    model_only_peak_mem_mb = measure_model_only_peak_memory(
        model=model,
        loader=test_loader,
        device=device
    )

    print("\nMODEL-ONLY INFERENCE MEMORY")
    print("-" * 60)

    if device.type == "cuda":
        print(f"Model-only Peak GPU Memory During Inference (MB): {model_only_peak_mem_mb:.2f}")
    else:
        print(f"Model-only Peak CPU Memory During Inference (MB): {model_only_peak_mem_mb:.2f}")

    # ==========================================
    # METRICS
    # ==========================================
    mae, rmse, mape, r2, preds_d, targs_d = evaluate(
        model,
        test_loader,
        device,
        t_mean,
        t_std
    )

    print("\nPURE SPATIAL GCN RESULTS")
    print("=" * 60)
    print(f"MAE  : {mae:.4f}")
    print(f"RMSE : {rmse:.4f}")
    print(f"MSE  : {rmse ** 2:.4f}")
    print(f"MAPE : {mape:.2f}%")
    print(f"R²   : {r2:.4f}")

    print("\nCOMPLEXITY SUMMARY")
    print("=" * 60)
    print(f"Total Parameters:        {total_params:,}")
    print(f"Trainable Parameters:    {trainable_params:,}")
    print(f"Parameter Size MB:       {param_size_mb:.4f}")
    print(f"Inference ms/batch:      {model_only_ms_batch:.4f}")
    print(f"Inference ms/sample:     {model_only_ms_sample:.6f}")
    print(f"Peak Inference Memory:   {model_only_peak_mem_mb:.2f} MB")

    plot_training_curves(train_losses, val_losses)

    print("\nRun complete. Plots saved.")
