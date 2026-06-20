import os
import gc
import random
import time
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader

import matplotlib.pyplot as plt


# ==========================================
# PART 0: REPRODUCIBILITY
# ==========================================

SEED = 42

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def sync_if_cuda(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


set_seed(SEED)

RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
print("=" * 70, flush=True)
print(f"STGCN 2-BLOCK RUN STARTED | RUN_ID: {RUN_ID} | SEED: {SEED}", flush=True)
print("=" * 70, flush=True)


# ==========================================
# PART 1: GRAPH PROCESSING
# ==========================================

def get_chebyshev_laplacian(distance_file, n_nodes, k_cheb):
    print(f"Loading graph from {distance_file}...", flush=True)

    try:
        dist_matrix = np.load(distance_file)
    except FileNotFoundError:
        print(f"ERROR: File {distance_file} not found. Using random fallback.", flush=True)
        dist_matrix = np.random.rand(n_nodes, n_nodes)

    if dist_matrix.shape[0] != n_nodes or dist_matrix.shape[1] != n_nodes:
        raise ValueError(
            f"Distance matrix shape {dist_matrix.shape} does not match N_NODES={n_nodes}"
        )

    valid_dists = dist_matrix[dist_matrix > 0]
    sigma = np.std(valid_dists) if len(valid_dists) > 0 else 1.0
    epsilon = 0.5

    W = np.exp(-(dist_matrix ** 2) / (sigma ** 2))
    W[W < epsilon] = 0
    np.fill_diagonal(W, 0)

    D = np.array(np.sum(W, axis=1))
    D_inv_sqrt = np.power(D, -0.5)
    D_inv_sqrt[np.isinf(D_inv_sqrt)] = 0.0
    D_mat_inv_sqrt = np.diag(D_inv_sqrt)

    L = np.eye(n_nodes) - np.dot(np.dot(D_mat_inv_sqrt, W), D_mat_inv_sqrt)

    lambda_max = 2.0
    L_tilde = (2 * L) / lambda_max - np.eye(n_nodes)

    return torch.from_numpy(L_tilde.astype(np.float32))


# ==========================================
# PART 2: DATA LOADING
# ==========================================

def load_ems_data(
    csv_path,
    z_score_cols,
    pass_through_cols,
    history_window,
    prediction_horizon=1,
    train_ratio=0.6,
    val_ratio=0.2
):
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"Error: The file '{csv_path}' was not found.")

    if "datetime" not in df.columns:
        raise ValueError("DataFrame must contain 'datetime' column.")

    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values(["datetime", "ZIPCODE"])

    # ==========================================
    # Auto-generate missing cyclical time features
    # ==========================================
    dt_ref = df["datetime"].dt

    time_feats_config = {
        "hour": 24,
        "dayofweek": 7,
        "month": 12,
        "weekofyear": 52
    }

    print("Checking for missing time features...", flush=True)

    for name, period in time_feats_config.items():
        sin_name = f"{name}_sin"
        cos_name = f"{name}_cos"

        if (sin_name in pass_through_cols or cos_name in pass_through_cols) and (
            sin_name not in df.columns or cos_name not in df.columns
        ):
            print(f"  -> Generating {sin_name} and {cos_name}...", flush=True)

            if name == "weekofyear":
                raw_val = dt_ref.isocalendar().week.astype(int)
            else:
                raw_val = getattr(dt_ref, name)

            df[sin_name] = np.sin(2 * np.pi * raw_val / period)
            df[cos_name] = np.cos(2 * np.pi * raw_val / period)

    feature_cols = z_score_cols + pass_through_cols
    missing = [c for c in feature_cols if c not in df.columns]

    if missing:
        raise ValueError(f"Missing columns in CSV and could not auto-generate them: {missing}")

    # ==========================================
    # Pivot features
    # ==========================================
    data_list = []
    min_time_steps = float("inf")

    for feat in feature_cols:
        pivot_df = df.pivot_table(
            index="datetime",
            columns="ZIPCODE",
            values=feat,
            aggfunc="mean"
        )
        min_time_steps = min(min_time_steps, len(pivot_df))

    zip_order = None

    for feat in feature_cols:
        pivot_df = df.pivot_table(
            index="datetime",
            columns="ZIPCODE",
            values=feat,
            aggfunc="mean"
        )

        pivot_df = pivot_df.interpolate(method="linear", axis=0).fillna(0)
        pivot_df = pivot_df.reindex(sorted(pivot_df.columns), axis=1)

        if zip_order is None:
            zip_order = list(pivot_df.columns)
        else:
            pivot_df = pivot_df.reindex(columns=zip_order).fillna(0)

        pivot_df = pivot_df.iloc[:min_time_steps, :]
        data_list.append(pivot_df.values)

    data_raw = np.stack(data_list, axis=-1).astype(np.float32)

    print(f"\nRaw data shape: {data_raw.shape} = Time x Nodes x Features", flush=True)

    # ==========================================
    # Chronological split
    # ==========================================
    n_total = data_raw.shape[0]
    train_end = int(n_total * train_ratio)
    val_end = int(n_total * (train_ratio + val_ratio))

    train_data = data_raw[:train_end]
    val_data = data_raw[train_end:val_end]
    test_data = data_raw[val_end:]

    # ==========================================
    # Hybrid normalization
    # ==========================================
    num_z_cols = len(z_score_cols)

    train_z = train_data[..., :num_z_cols]

    mean_z = np.mean(train_z, axis=(0, 1), keepdims=True)
    std_z = np.std(train_z, axis=(0, 1), keepdims=True)
    std_z[std_z < 1e-5] = 1.0

    def normalize_split(data_arr):
        part_z = data_arr[..., :num_z_cols]
        part_p = data_arr[..., num_z_cols:]

        part_z_norm = (part_z - mean_z) / std_z

        return np.concatenate([part_z_norm, part_p], axis=-1).astype(np.float32)

    train_norm = normalize_split(train_data)
    val_norm = normalize_split(val_data)
    test_norm = normalize_split(test_data)

    print("\nNormalization Logic Applied:", flush=True)
    print(f"  - Z-scored features: {num_z_cols}", flush=True)
    print(f"  - Pass-through features: {len(pass_through_cols)}", flush=True)
    print(f"  - Target call_count mean: {mean_z[0, 0, 0]:.4f}", flush=True)
    print(f"  - Target call_count std:  {std_z[0, 0, 0]:.4f}", flush=True)

    # ==========================================
    # Sliding windows
    # ==========================================
    def create_windows(data, history, horizon):
        X, Y = [], []
        num_samples = len(data) - history - horizon + 1

        if num_samples <= 0:
            empty_x = np.empty(
                (0, data.shape[-1], history, data.shape[1]),
                dtype=np.float32
            )
            empty_y = np.empty(
                (0, horizon, data.shape[1]),
                dtype=np.float32
            )
            return empty_x, empty_y

        for i in range(num_samples):
            x_inst = data[i:i + history].transpose(2, 0, 1)
            y_inst = data[i + history:i + history + horizon, :, 0]

            X.append(x_inst)
            Y.append(y_inst)

        return np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32)

    print("\nCreating sliding windows...", flush=True)

    X_train, Y_train = create_windows(train_norm, history_window, prediction_horizon)
    X_val, Y_val = create_windows(val_norm, history_window, prediction_horizon)
    X_test, Y_test = create_windows(test_norm, history_window, prediction_horizon)

    print(f"Train X: {X_train.shape}, Train Y: {Y_train.shape}", flush=True)
    print(f"Val   X: {X_val.shape}, Val   Y: {Y_val.shape}", flush=True)
    print(f"Test  X: {X_test.shape}, Test  Y: {Y_test.shape}", flush=True)

    X_train_t = torch.from_numpy(X_train)
    Y_train_t = torch.from_numpy(Y_train)
    X_val_t = torch.from_numpy(X_val)
    Y_val_t = torch.from_numpy(Y_val)
    X_test_t = torch.from_numpy(X_test)
    Y_test_t = torch.from_numpy(Y_test)

    return (
        (X_train_t, Y_train_t),
        (X_val_t, Y_val_t),
        (X_test_t, Y_test_t),
        (mean_z, std_z)
    )


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

        x_padded = F.pad(x, (0, 0, self.Kt - 1, 0))
        out = self.conv(x_padded)

        P, Q = torch.chunk(out, 2, dim=1)

        out = (P + x_in) * torch.sigmoid(Q)

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

        L = self.L_tilde.to(x.device)

        x_0 = x_flat
        x_1 = torch.matmul(L, x_flat)

        cheb_list = [x_0, x_1]

        for k in range(2, self.Ks):
            x_k = 2 * torch.matmul(L, cheb_list[-1]) - cheb_list[-2]
            cheb_list.append(x_k)

        cheb_stacked = torch.stack(cheb_list, dim=0)

        out = torch.einsum("kbni,kio->bno", cheb_stacked, self.Theta) + self.bias

        out = out.view(B, T, N, -1).permute(0, 3, 1, 2)

        return F.relu(out)


class STGCN(nn.Module):
    def __init__(self, Ks, Kt, n_nodes, blocks, L_tilde):
        super(STGCN, self).__init__()

        self.layers = nn.ModuleList()

        for channels in blocks:
            c_in, c_h, c_out = channels

            self.layers.append(
                nn.Sequential(
                    TemporalConvLayer(Kt, c_in, c_h),
                    ChebGraphConv(c_h, c_h, Ks, L_tilde),
                    TemporalConvLayer(Kt, c_h, c_out)
                )
            )

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
# PART 4: EVALUATION
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

            all_preds.append(pred.squeeze(1).detach().cpu().numpy())
            all_targets.append(batch_y.squeeze(1).detach().cpu().numpy())

    preds = np.concatenate(all_preds, axis=0)
    targs = np.concatenate(all_targets, axis=0)

    preds_denorm = preds * std + mean
    targs_denorm = targs * std + mean

    mae = np.mean(np.abs(preds_denorm - targs_denorm))
    mse = np.mean((preds_denorm - targs_denorm) ** 2)
    rmse = np.sqrt(mse)

    non_zero_mask = targs_denorm != 0

    if np.any(non_zero_mask):
        mape = np.mean(
            np.abs(
                (preds_denorm[non_zero_mask] - targs_denorm[non_zero_mask])
                / targs_denorm[non_zero_mask]
            )
        ) * 100
    else:
        mape = np.nan

    ss_res = np.sum((targs_denorm - preds_denorm) ** 2)
    ss_tot = np.sum((targs_denorm - np.mean(targs_denorm)) ** 2)
    r2 = 1 - (ss_res / (ss_tot + 1e-8))

    return mae, rmse, mse, mape, r2, preds_denorm, targs_denorm


# ==========================================
# PART 5: MODEL-ONLY INFERENCE TIME / MEMORY
# ==========================================

def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    param_size_mb = total_params * 4 / (1024 ** 2)

    return total_params, trainable_params, param_size_mb


def measure_model_only_inference_time(model, loader, device, warmup_steps=20, repeats=10):
    """
    Measures model-only inference time.

    This excludes:
    - CSV loading
    - preprocessing
    - DataLoader iteration cost during timing
    - CPU-to-GPU transfer during timing
    - metric calculation
    - plotting

    It measures only:
    - trained model forward pass: pred = model(batch_x)
    """

    model.eval()

    # Move all test batches to device BEFORE timing.
    # This removes CPU-to-GPU transfer from the timed section.
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

    # Cleanup preloaded batches from GPU memory
    del test_batches_device
    gc.collect()

    if device.type == "cuda":
        torch.cuda.empty_cache()

    return ms_per_batch, ms_per_sample, sec_per_full_test_pass


def measure_model_only_peak_memory(model, loader, device):
    """
    Measures model-only peak memory during inference.

    For CUDA:
    - Moves one batch to GPU before resetting peak stats.
    - Resets peak memory immediately before forward pass.
    - Measures peak allocated memory during forward pass.
    - Does not include optimizer states because optimizer is deleted before this function is called.

    The reported value includes:
    - model parameters
    - one input batch
    - forward-pass activations/intermediate tensors
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

                current, peak = tracemalloc.get_traced_memory()
                tracemalloc.stop()

                batch_peak_mb = peak / (1024 ** 2)
                peak_mem_mb = max(peak_mem_mb, batch_peak_mb)

                del pred
                del batch_x

                gc.collect()

        return peak_mem_mb


# ==========================================
# PART 6: PLOTTING
# ==========================================

def plot_training_curves(train_losses, val_losses, output_file):
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label="Train Loss", linewidth=2)
    plt.plot(val_losses, label="Val Loss", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("STGCN 2-Block Training and Validation Loss")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()


def plot_predictions_samples(preds_denorm, targs_denorm, output_file, n_samples=4):
    num_samples = preds_denorm.shape[0]
    n_samples = min(n_samples, num_samples)

    np.random.seed(SEED)
    idxs = np.random.choice(num_samples, n_samples, replace=False)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()

    for ax, idx in zip(axes, idxs):
        pred = preds_denorm[idx, :]
        true = targs_denorm[idx, :]

        nodes = np.arange(len(pred))

        ax.plot(nodes, true, "k-", label="True", linewidth=1.5)
        ax.plot(nodes, pred, "r--", label="Pred", linewidth=1.5)

        ax.set_title(f"Sample {idx}")
        ax.set_xlabel("Node index")
        ax.set_ylabel("Call count")
        ax.grid(True, alpha=0.3)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right")

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()


def plot_error_histogram(preds_denorm, targs_denorm, output_file):
    errors = np.abs(preds_denorm - targs_denorm).flatten()

    plt.figure(figsize=(8, 5))
    plt.hist(errors, bins=50, color="steelblue", edgecolor="black", alpha=0.8)
    plt.xlabel("Absolute Error: call count")
    plt.ylabel("Frequency")
    plt.title("Distribution of Prediction Errors")
    plt.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()


# ==========================================
# PART 7: EXECUTION
# ==========================================

if __name__ == "__main__":

    # -----------------------------
    # Static configuration
    # -----------------------------
    DIST_FILE = "distance_matrix (1).npy"
    DATA_FILE = "All_Features_3.csv"

    N_NODES = 197
    M_HISTORY = 24

    KS = 3
    KT = 4

    EPOCHS = 50

    CHECKPOINT_PATH = "stgcn_ems_best_2blocks.pth"

    TRAIN_VAL_LOSS_PLOT = "stgcn_train_val_loss_STGCN_2blocks.png"
    PRED_SAMPLE_PLOT = "stgcn_pred_samples_STGCN_2blocks.png"
    ERROR_HIST_PLOT = "stgcn_error_hist_STGCN_2blocks.png"

    # -----------------------------
    # Feature configuration
    # -----------------------------
    Z_SCORE_COLS = [
        "call_count",
        "lag_1",
        "lag_24",
        "lag_168",
        "roll_mean_24",
        "roll_std_24",
        "Population",
        "INITIAL_SEVERITY_LEVEL_CODE"
    ]

    PASS_THROUGH_COLS = [
        "hour_sin",
        "hour_cos",
        "dayofweek_sin",
        "dayofweek_cos",
        "month_sin",
        "month_cos",
        "weekofyear_sin",
        "weekofyear_cos",
        "is_Covid",
        "SPECIAL_EVENT_INDICATOR"
    ]

    NUM_FEATURES = len(Z_SCORE_COLS) + len(PASS_THROUGH_COLS)

    print(f"Total Features Configured: {NUM_FEATURES}", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Running on device: {device}", flush=True)

    if device.type == "cuda":
        print(f"GPU name: {torch.cuda.get_device_name(0)}", flush=True)
        print(f"CUDA version used by PyTorch: {torch.version.cuda}", flush=True)

    # -----------------------------
    # Graph
    # -----------------------------
    L_tilde = get_chebyshev_laplacian(
        distance_file=DIST_FILE,
        n_nodes=N_NODES,
        k_cheb=KS
    ).to(device)

    # -----------------------------
    # Data
    # -----------------------------
    print("\nLoading EMS data...", flush=True)

    (X_train, Y_train), (X_val, Y_val), (X_test, Y_test), (mean_z, std_z) = load_ems_data(
        csv_path=DATA_FILE,
        z_score_cols=Z_SCORE_COLS,
        pass_through_cols=PASS_THROUGH_COLS,
        history_window=M_HISTORY,
        prediction_horizon=1,
        train_ratio=0.6,
        val_ratio=0.2
    )

    if X_train.shape[-1] != N_NODES:
        raise ValueError(
            f"Data node count {X_train.shape[-1]} does not match N_NODES={N_NODES}."
        )

    # -----------------------------
    # DataLoaders
    # -----------------------------
    TEST_BATCH_SIZE = 50

    test_dataset = TensorDataset(X_test, Y_test)

    test_loader = DataLoader(
        test_dataset,
        batch_size=TEST_BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    print(f"Test samples: {len(test_loader.dataset)}", flush=True)
    print(f"Test batches: {len(test_loader)}", flush=True)
    print(f"Test batch size: {TEST_BATCH_SIZE}", flush=True)

    # ==========================================
    # RETRAIN FINAL 2-BLOCK MODEL
    # ==========================================
    print("\n" + "=" * 60, flush=True)
    print("RETRAINING FINAL 2-BLOCK STGCN MODEL", flush=True)
    print("=" * 60, flush=True)

    best_params = {
        "lr": 0.0020861235735065513,
        "batch_size": 16,
        "hidden_dim": 64,
        "out_dim": 32,
        "optimizer": "AdamW",
        "step_size": 7
    }

    best_train_loader = DataLoader(
        TensorDataset(X_train, Y_train),
        batch_size=best_params["batch_size"],
        shuffle=True,
        num_workers=0
    )

    best_val_loader = DataLoader(
        TensorDataset(X_val, Y_val),
        batch_size=best_params["batch_size"],
        shuffle=False,
        num_workers=0
    )

    # -----------------------------
    # 2 ST blocks
    # -----------------------------
    final_blocks = [
        [NUM_FEATURES, best_params["hidden_dim"], best_params["out_dim"]],
        [best_params["out_dim"], best_params["hidden_dim"], best_params["out_dim"]]
    ]

    final_model = STGCN(
        Ks=KS,
        Kt=KT,
        n_nodes=N_NODES,
        blocks=final_blocks,
        L_tilde=L_tilde
    ).to(device)

    # -----------------------------
    # Parameter count
    # -----------------------------
    total_params, trainable_params, param_size_mb = count_parameters(final_model)

    print(f"Total Parameters:     {total_params:,}", flush=True)
    print(f"Trainable Parameters: {trainable_params:,}", flush=True)
    print(f"Parameter Size:       {param_size_mb:.4f} MB assuming float32", flush=True)

    final_optimizer = getattr(optim, best_params["optimizer"])(
        final_model.parameters(),
        lr=best_params["lr"]
    )

    final_scheduler = optim.lr_scheduler.StepLR(
        final_optimizer,
        step_size=best_params["step_size"],
        gamma=0.7
    )

    final_loss_fn = nn.MSELoss()

    best_final_val_loss = float("inf")

    train_losses = []
    val_losses = []

    for epoch in range(EPOCHS):
        final_model.train()

        train_loss_sum = 0.0
        train_count = 0

        for batch_x, batch_y in best_train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            final_optimizer.zero_grad()

            pred = final_model(batch_x)

            loss = final_loss_fn(
                pred.squeeze(1),
                batch_y.squeeze(1)
            )

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
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)

                pred = final_model(batch_x)

                loss = final_loss_fn(
                    pred.squeeze(1),
                    batch_y.squeeze(1)
                )

                val_loss_sum += loss.item() * batch_x.size(0)
                val_count += batch_x.size(0)

        epoch_val_loss = val_loss_sum / val_count
        val_losses.append(epoch_val_loss)

        final_scheduler.step()

        if epoch_val_loss < best_final_val_loss:
            best_final_val_loss = epoch_val_loss
            torch.save(final_model.state_dict(), CHECKPOINT_PATH)

        if (epoch + 1) % 10 == 0:
            print(
                f"Final Retrain Epoch {epoch + 1}/{EPOCHS} | "
                f"Train: {epoch_train_loss:.4f} | "
                f"Val: {epoch_val_loss:.4f}",
                flush=True
            )

    # ==========================================
    # FINAL EVALUATION
    # ==========================================
    print("\n" + "=" * 60, flush=True)
    print("EVALUATING ON TEST SET", flush=True)
    print("=" * 60, flush=True)

    final_model.load_state_dict(
        torch.load(CHECKPOINT_PATH, map_location=device)
    )

    final_model.eval()

    # Delete optimizer/scheduler/loss before inference memory measurement.
    # This prevents optimizer states from being counted as inference memory.
    try:
        del final_optimizer
        del final_scheduler
        del final_loss_fn
        del batch_x
        del batch_y
        del pred
        del loss
    except NameError:
        pass

    gc.collect()

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    target_mean = mean_z[0, 0, 0]
    target_std = std_z[0, 0, 0]

    # -----------------------------
    # Model-only inference time
    # -----------------------------
    model_only_ms_batch, model_only_ms_sample, sec_per_test_pass = measure_model_only_inference_time(
        model=final_model,
        loader=test_loader,
        device=device,
        warmup_steps=20,
        repeats=10
    )

    print("\nMODEL-ONLY INFERENCE TIME", flush=True)
    print("-" * 60, flush=True)
    print(f"Model-only Inference Time (ms per batch):  {model_only_ms_batch:.4f}", flush=True)
    print(f"Model-only Inference Time (ms per sample): {model_only_ms_sample:.6f}", flush=True)
    print(f"Seconds per full test pass:                {sec_per_test_pass:.4f}", flush=True)

    # -----------------------------
    # Model-only peak memory
    # -----------------------------
    model_only_peak_mem_mb = measure_model_only_peak_memory(
        model=final_model,
        loader=test_loader,
        device=device
    )

    print("\nMODEL-ONLY INFERENCE MEMORY", flush=True)
    print("-" * 60, flush=True)

    if device.type == "cuda":
        print(f"Model-only Peak GPU Memory During Inference (MB): {model_only_peak_mem_mb:.2f}", flush=True)
    else:
        print(f"Model-only Peak CPU Memory During Inference (MB): {model_only_peak_mem_mb:.2f}", flush=True)

    # -----------------------------
    # Accuracy metrics
    # -----------------------------
    mae, rmse, mse, mape, r2, preds_denorm, targs_denorm = evaluate(
        model=final_model,
        loader=test_loader,
        device=device,
        mean=target_mean,
        std=target_std
    )

    print("\nFINAL TEST RESULTS", flush=True)
    print("=" * 60, flush=True)
    print(f"MAE:  {mae:.4f}", flush=True)
    print(f"RMSE: {rmse:.4f}", flush=True)
    print(f"MSE:  {mse:.4f}", flush=True)
    print(f"MAPE: {mape:.2f}%", flush=True)
    print(f"R²:   {r2:.4f}", flush=True)

    print("\nCOMPLEXITY SUMMARY", flush=True)
    print("=" * 60, flush=True)
    print(f"Total Parameters:        {total_params:,}", flush=True)
    print(f"Trainable Parameters:    {trainable_params:,}", flush=True)
    print(f"Parameter Size MB:       {param_size_mb:.4f}", flush=True)
    print(f"Inference ms/batch:      {model_only_ms_batch:.4f}", flush=True)
    print(f"Inference ms/sample:     {model_only_ms_sample:.6f}", flush=True)
    print(f"Peak Inference Memory:   {model_only_peak_mem_mb:.2f} MB", flush=True)

    # -----------------------------
    # Plots
    # -----------------------------
    plot_training_curves(
        train_losses=train_losses,
        val_losses=val_losses,
        output_file=TRAIN_VAL_LOSS_PLOT
    )

    plot_predictions_samples(
        preds_denorm=preds_denorm,
        targs_denorm=targs_denorm,
        output_file=PRED_SAMPLE_PLOT,
        n_samples=4
    )

    plot_error_histogram(
        preds_denorm=preds_denorm,
        targs_denorm=targs_denorm,
        output_file=ERROR_HIST_PLOT
    )

    print("\nGenerated plots:", flush=True)
    print(f"  - {TRAIN_VAL_LOSS_PLOT}", flush=True)
    print(f"  - {PRED_SAMPLE_PLOT}", flush=True)
    print(f"  - {ERROR_HIST_PLOT}", flush=True)

    print("\n" + "=" * 70, flush=True)
    print(f"STGCN 2-BLOCK RUN FINISHED | RUN_ID: {RUN_ID}", flush=True)
    print("=" * 70, flush=True)
