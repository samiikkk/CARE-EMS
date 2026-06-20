"""
ST-MGCN for ZIP-code-level EMS Call Demand Forecasting
======================================================

Updated version:
- Keeps your original ST-MGCN training and evaluation logic.
- Measures model-only inference time.
- Measures inference-only peak memory.
- Reports total parameters, trainable parameters, and parameter size.
"""

import gc
import random
import time
from datetime import datetime

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity


# =============================================================================
# SECTION 0: REPRODUCIBILITY
# =============================================================================

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
print(f"ST-MGCN RUN STARTED | RUN_ID: {RUN_ID} | SEED: {SEED}", flush=True)
print("=" * 70, flush=True)


# =============================================================================
# SECTION 1: GRAPH CONSTRUCTION
# =============================================================================

def build_proximity_graph(
    dist_matrix: np.ndarray,
    sigma_sq: float = None,
    threshold: float = 0.5,
) -> np.ndarray:
    N = dist_matrix.shape[0]

    if dist_matrix.shape != (N, N):
        raise ValueError("dist_matrix must be square.")

    if sigma_sq is None:
        mask = (dist_matrix > 0) & np.isfinite(dist_matrix)
        sigma_std = float(np.std(dist_matrix[mask])) if mask.sum() > 0 else 1.0
        sigma_sq = max(sigma_std ** 2, 1e-8)

    with np.errstate(divide="ignore", invalid="ignore"):
        adj = np.exp(-np.square(dist_matrix) / sigma_sq)

    np.fill_diagonal(adj, 0.0)
    adj[adj < threshold] = 0.0

    print(f"  [Proximity Graph] sigma_sq={sigma_sq:.4f}, threshold={threshold}", flush=True)
    print(f"  Non-zero edges: {int((adj > 0).sum())} / {N * (N - 1)}", flush=True)

    return adj.astype(np.float32)


def build_demand_corr_graph(
    df: pd.DataFrame,
    zip_col: str,
    time_col: str,
    call_col: str,
    zip_list: list,
    top_k: int = 10,
    corr_threshold: float = 0.3,
) -> np.ndarray:
    pivot = df.pivot_table(
        index=time_col,
        columns=zip_col,
        values=call_col,
        aggfunc="mean"
    )

    pivot = pivot.reindex(columns=zip_list).fillna(0.0)

    corr_mat = pivot.corr().values
    corr_mat = np.nan_to_num(corr_mat, nan=0.0)

    adj = corr_mat.copy()
    np.fill_diagonal(adj, 0.0)
    adj[adj < corr_threshold] = 0.0

    N = len(zip_list)

    if top_k > 0:
        for i in range(N):
            row = adj[i].copy()

            if row.max() == 0:
                continue

            top_idx = np.argsort(row)[::-1][:top_k]
            mask = np.zeros(N, dtype=bool)
            mask[top_idx] = True
            adj[i][~mask] = 0.0

    print(f"  [Demand-Corr Graph] corr_threshold={corr_threshold}, top_k={top_k}", flush=True)
    print(f"  Non-zero edges: {int((adj > 0).sum())} / {N * (N - 1)}", flush=True)

    return adj.astype(np.float32)


def build_context_graph(
    df: pd.DataFrame,
    zip_col: str,
    feature_cols: list,
    zip_list: list,
    sim_threshold: float = 0.85,
) -> np.ndarray:
    zip_feat = (
        df.groupby(zip_col)[feature_cols]
        .mean()
        .reindex(zip_list)
        .fillna(0.0)
    )

    feat_mat = zip_feat.values.astype(np.float64)
    feat_mat_std = StandardScaler().fit_transform(feat_mat)

    sim_mat = cosine_similarity(feat_mat_std)
    sim_mat = np.nan_to_num(sim_mat, nan=0.0)

    adj = sim_mat.copy()
    np.fill_diagonal(adj, 0.0)
    adj[adj < sim_threshold] = 0.0

    N = len(zip_list)

    print(f"  [Context Graph] sim_threshold={sim_threshold}", flush=True)
    print(f"  Non-zero edges: {int((adj > 0).sum())} / {N * (N - 1)}", flush=True)

    return adj.astype(np.float32)


# =============================================================================
# SECTION 2: ADJACENCY PREPROCESSOR
# =============================================================================

class Adj_Preprocessor:
    def __init__(self, kernel_type: str, K: int):
        self.kernel_type = kernel_type
        self.K = K if kernel_type != "localpool" else 1

    def process(self, adj: torch.Tensor) -> torch.Tensor:
        N = adj.shape[0]
        kernel_list = []

        if self.kernel_type in ["localpool", "chebyshev"]:
            adj_norm = self._symmetric_normalize(adj)

            if self.kernel_type == "localpool":
                kernel_list.append(torch.eye(N))
                kernel_list.append(adj_norm)
            else:
                L = torch.eye(N) - adj_norm
                L_rescaled = self._rescale_laplacian(L)
                kernel_list = self._chebyshev_polynomials(L_rescaled, kernel_list)

        elif self.kernel_type == "random_walk_diffusion":
            P = self._random_walk_normalize(adj)
            kernel_list = self._chebyshev_polynomials(P.T, kernel_list)

        else:
            raise ValueError(
                f"Invalid kernel_type '{self.kernel_type}'. "
                "Choose: chebyshev | localpool | random_walk_diffusion"
            )

        return torch.stack(kernel_list, dim=0)

    @staticmethod
    def _random_walk_normalize(A: torch.Tensor) -> torch.Tensor:
        d_inv = torch.pow(A.sum(dim=1), -1)
        d_inv[torch.isinf(d_inv)] = 0.0
        return torch.mm(torch.diag(d_inv), A)

    @staticmethod
    def _symmetric_normalize(A: torch.Tensor) -> torch.Tensor:
        d = A.sum(dim=1)
        d[d == 0] = 1e-8
        D = torch.diag(torch.pow(d, -0.5))
        return torch.mm(torch.mm(D, A), D)

    @staticmethod
    def _rescale_laplacian(L: torch.Tensor) -> torch.Tensor:
        try:
            lambda_max = float(torch.linalg.eigvalsh(L).max())
        except Exception:
            print("  [Warning] Eigenvalue computation failed; using lambda_max=2.", flush=True)
            lambda_max = 2.0

        lambda_max = max(lambda_max, 1e-6)

        return (2.0 / lambda_max) * L - torch.eye(L.shape[0])

    def _chebyshev_polynomials(self, x: torch.Tensor, T_k: list) -> list:
        for k in range(self.K + 1):
            if k == 0:
                T_k.append(torch.eye(x.shape[0]))
            elif k == 1:
                T_k.append(x.clone())
            else:
                T_k.append(2.0 * torch.mm(x, T_k[k - 1]) - T_k[k - 2])

        return T_k


# =============================================================================
# SECTION 3: MODEL MODULES
# =============================================================================

class GCN(nn.Module):
    def __init__(
        self,
        K: int,
        input_dim: int,
        hidden_dim: int,
        bias: bool = True,
        activation=nn.ReLU
    ):
        super().__init__()

        self.K = K
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.activation = activation() if activation is not None else None

        self.W = nn.Parameter(torch.empty(K * input_dim, hidden_dim))
        nn.init.xavier_normal_(self.W)

        self.b = nn.Parameter(torch.zeros(hidden_dim)) if bias else None

    def forward(self, A: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        assert A.shape[0] == self.K

        supports = [
            torch.einsum("ij,bjp->bip", A[k], x)
            for k in range(self.K)
        ]

        support_cat = torch.cat(supports, dim=-1)
        out = torch.einsum("bip,pq->biq", support_cat, self.W)

        if self.b is not None:
            out = out + self.b

        if self.activation is not None:
            out = self.activation(out)

        return out


class CG_LSTM(nn.Module):
    def __init__(
        self,
        seq_len: int,
        n_nodes: int,
        input_dim: int,
        lstm_hidden_dim: int,
        lstm_num_layers: int,
        K: int,
        gconv_use_bias: bool,
        gconv_activation=nn.ReLU
    ):
        super().__init__()

        self.seq_len = seq_len
        self.n_nodes = n_nodes
        self.input_dim = input_dim
        self.lstm_hidden_dim = lstm_hidden_dim
        self.lstm_num_layers = lstm_num_layers

        self.gconv_temporal = GCN(
            K=K,
            input_dim=seq_len,
            hidden_dim=seq_len,
            bias=gconv_use_bias,
            activation=gconv_activation,
        )

        self.fc_gate_1 = nn.Linear(seq_len, seq_len, bias=True)
        self.fc_gate_2 = nn.Linear(seq_len, seq_len, bias=True)

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=lstm_hidden_dim,
            num_layers=lstm_num_layers,
            batch_first=True,
        )

    def forward(
        self,
        adj: torch.Tensor,
        obs_seq: torch.Tensor,
        hidden: tuple
    ) -> tuple:
        B = obs_seq.shape[0]

        x_seq = obs_seq.sum(dim=-1).permute(0, 2, 1)
        x_gconv = self.gconv_temporal(adj, x_seq)

        x_hat = x_seq + x_gconv
        z_t = x_hat.mean(dim=1)

        s = torch.sigmoid(
            self.fc_gate_2(torch.relu(self.fc_gate_1(z_t)))
        )

        obs_rw = torch.einsum("btnf,bt->btnf", obs_seq, s)

        shared = obs_rw.permute(0, 2, 1, 3).reshape(
            B * self.n_nodes,
            self.seq_len,
            self.input_dim
        )

        lstm_out, hidden = self.lstm(shared, hidden)

        output = lstm_out[:, -1, :].reshape(
            B,
            self.n_nodes,
            self.lstm_hidden_dim
        )

        return output, hidden

    def init_hidden(self, batch_size: int) -> tuple:
        w = next(self.parameters()).data

        h = w.new_zeros(
            self.lstm_num_layers,
            batch_size * self.n_nodes,
            self.lstm_hidden_dim
        )

        c = w.new_zeros(
            self.lstm_num_layers,
            batch_size * self.n_nodes,
            self.lstm_hidden_dim
        )

        return h, c


class ST_MGCN(nn.Module):
    def __init__(
        self,
        M: int,
        seq_len: int,
        n_nodes: int,
        input_dim: int,
        lstm_hidden_dim: int,
        lstm_num_layers: int,
        gcn_hidden_dim: int,
        sta_kernel_config: dict,
        gconv_use_bias: bool = True,
        gconv_activation=nn.ReLU
    ):
        super().__init__()

        self.M = M
        self.sta_K = self._get_support_K(sta_kernel_config)

        self.rnn_list = nn.ModuleList()
        self.gcn_list = nn.ModuleList()

        for _ in range(M):
            self.rnn_list.append(
                CG_LSTM(
                    seq_len=seq_len,
                    n_nodes=n_nodes,
                    input_dim=input_dim,
                    lstm_hidden_dim=lstm_hidden_dim,
                    lstm_num_layers=lstm_num_layers,
                    K=self.sta_K,
                    gconv_use_bias=gconv_use_bias,
                    gconv_activation=gconv_activation,
                )
            )

            self.gcn_list.append(
                GCN(
                    K=self.sta_K,
                    input_dim=lstm_hidden_dim,
                    hidden_dim=gcn_hidden_dim,
                    bias=gconv_use_bias,
                    activation=gconv_activation,
                )
            )

        self.output_fc = nn.Linear(gcn_hidden_dim, 1, bias=True)

    @staticmethod
    def _get_support_K(config: dict) -> int:
        kernel_type = config["kernel_type"]

        if kernel_type == "localpool":
            return 2
        elif kernel_type == "chebyshev":
            return config["K"] + 1
        elif kernel_type == "random_walk_diffusion":
            return config["K"] + 1

        raise ValueError(f"Unknown kernel_type: {kernel_type!r}")

    def init_hidden_list(self, batch_size: int) -> list:
        return [
            self.rnn_list[m].init_hidden(batch_size)
            for m in range(self.M)
        ]

    def forward(
        self,
        obs_seq: torch.Tensor,
        sta_adj_list: list
    ) -> torch.Tensor:
        assert len(sta_adj_list) == self.M, (
            f"Expected {self.M} adjacency matrices, got {len(sta_adj_list)}."
        )

        B = obs_seq.shape[0]

        hidden_list = self.init_hidden_list(B)
        feat_list = []

        for m in range(self.M):
            rnn_out, hidden_list[m] = self.rnn_list[m](
                sta_adj_list[m],
                obs_seq,
                hidden_list[m]
            )

            gcn_out = self.gcn_list[m](
                sta_adj_list[m],
                rnn_out
            )

            feat_list.append(gcn_out)

        feat_fused = torch.stack(feat_list, dim=-1).sum(dim=-1)

        y_pred = self.output_fc(feat_fused)

        return y_pred


# =============================================================================
# SECTION 4: DATA LOADING
# =============================================================================

_TIME_CANDIDATES = [
    "datetime", "date", "timestamp", "time", "hour",
    "Datetime", "Date", "Timestamp", "Time", "Hour",
]

_ZIP_CANDIDATES = [
    "zip", "ZIP", "zipcode", "zip_code", "Zip", "ZipCode",
    "zip code", "postal_code", "PostalCode", "postalcode", "ZIPCODE",
]

_CALL_CANDIDATES = [
    "call_count", "call count", "CallCount", "calls",
    "Call_Count", "CALL_COUNT", "demand", "count",
]

_AUTO_GEN_SINCOS_COLS = {
    "hour_sin", "hour_cos",
    "dayofweek_sin", "dayofweek_cos",
    "month_sin", "month_cos",
    "weekofyear_sin", "weekofyear_cos",
}

_TIME_FEATS_CONFIG = {
    "hour": 24,
    "dayofweek": 7,
    "month": 12,
    "weekofyear": 52,
}


def detect_col(df: pd.DataFrame, candidates: list, label: str) -> str:
    for c in candidates:
        if c in df.columns:
            return c

    raise ValueError(
        f"Could not auto-detect '{label}' column.\n"
        f"Tried: {candidates}\n"
        f"Actual columns: {list(df.columns)}"
    )


def load_ems_data(
    csv_path: str,
    z_score_cols: list,
    pass_through_cols: list,
    history_window: int = 24,
    prediction_horizon: int = 1,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    zip_col: str = "zip",
    time_col: str = "datetime",
) -> tuple:
    df = pd.read_csv(csv_path)

    time_col = detect_col(df, [time_col] + _TIME_CANDIDATES, "time/datetime")
    zip_col = detect_col(df, [zip_col] + _ZIP_CANDIDATES, "zip/zipcode")

    print(f"  Detected columns -> time='{time_col}', zip='{zip_col}'", flush=True)

    df[time_col] = pd.to_datetime(df[time_col])
    df = df.sort_values([time_col, zip_col]).reset_index(drop=True)

    dt_ref = df[time_col].dt

    print("  Checking for missing cyclical time features...", flush=True)

    for raw_name, period in _TIME_FEATS_CONFIG.items():
        sin_name = f"{raw_name}_sin"
        cos_name = f"{raw_name}_cos"

        if (sin_name in pass_through_cols or cos_name in pass_through_cols) and (
            sin_name not in df.columns or cos_name not in df.columns
        ):
            print(
                f"    -> Auto-generating '{sin_name}' and '{cos_name}'",
                flush=True
            )

            if raw_name == "weekofyear":
                raw_val = dt_ref.isocalendar().week.astype(int)
            else:
                raw_val = getattr(dt_ref, raw_name)

            df[sin_name] = np.sin(2 * np.pi * raw_val / period)
            df[cos_name] = np.cos(2 * np.pi * raw_val / period)

    all_cols = z_score_cols + pass_through_cols

    missing = [c for c in all_cols if c not in df.columns]

    if missing:
        raise ValueError(
            f"Missing columns in CSV and could not auto-generate them: {missing}"
        )

    zip_list = sorted(df[zip_col].unique().tolist())
    N_nodes = len(zip_list)

    timestamps = sorted(df[time_col].unique())
    T_all = len(timestamps)
    F = len(all_cols)

    feature_arr = np.zeros((T_all, N_nodes, F), dtype=np.float32)

    for fi, col in enumerate(all_cols):
        pivot = df.pivot_table(
            index=time_col,
            columns=zip_col,
            values=col,
            aggfunc="mean"
        )

        pivot = pivot.reindex(
            index=timestamps,
            columns=zip_list
        ).fillna(0.0)

        feature_arr[:, :, fi] = pivot.values.astype(np.float32)

    n_train = int(T_all * train_ratio)
    n_val = int(T_all * val_ratio)

    train_arr = feature_arr[:n_train]
    val_arr = feature_arr[n_train:n_train + n_val]
    test_arr = feature_arr[n_train + n_val:]

    n_z = len(z_score_cols)

    train_z = train_arr[:, :, :n_z]

    mean_z = train_z.mean(axis=(0, 1), keepdims=True)
    std_z = train_z.std(axis=(0, 1), keepdims=True)
    std_z[std_z < 1e-5] = 1.0

    def normalize(arr: np.ndarray) -> np.ndarray:
        out = arr.copy()
        out[:, :, :n_z] = (out[:, :, :n_z] - mean_z) / std_z
        return out.astype(np.float32)

    train_arr = normalize(train_arr)
    val_arr = normalize(val_arr)
    test_arr = normalize(test_arr)

    mean_call = float(mean_z[0, 0, 0])
    std_call = float(std_z[0, 0, 0])

    print("\n  Normalization applied:", flush=True)
    print(f"    Z-scored features: {n_z}", flush=True)
    print(f"    Pass-through features: {len(pass_through_cols)}", flush=True)
    print(f"    call_count mean={mean_call:.4f}, std={std_call:.4f}", flush=True)

    def make_samples(arr: np.ndarray):
        Xs, Ys = [], []
        T = len(arr)

        for t in range(history_window, T - prediction_horizon + 1):
            Xs.append(arr[t - history_window:t])
            Ys.append(arr[t:t + prediction_horizon, :, 0])

        if not Xs:
            return (
                torch.FloatTensor(np.empty((0, history_window, N_nodes, F))),
                torch.FloatTensor(np.empty((0, prediction_horizon, N_nodes))),
            )

        return (
            torch.FloatTensor(np.stack(Xs)),
            torch.FloatTensor(np.stack(Ys)),
        )

    print("\n  Creating sliding windows...", flush=True)

    X_train, Y_train = make_samples(train_arr)
    X_val, Y_val = make_samples(val_arr)
    X_test, Y_test = make_samples(test_arr)

    print("\n  Data loaded successfully:", flush=True)
    print(f"    X_train: {X_train.shape} | Y_train: {Y_train.shape}", flush=True)
    print(f"    X_val:   {X_val.shape} | Y_val:   {Y_val.shape}", flush=True)
    print(f"    X_test:  {X_test.shape} | Y_test:  {Y_test.shape}", flush=True)
    print(f"    N_nodes={N_nodes} | NUM_FEATURES={F}", flush=True)

    return (
        (X_train, Y_train),
        (X_val, Y_val),
        (X_test, Y_test),
        (mean_call, std_call),
        zip_list,
    )


# =============================================================================
# SECTION 5: EVALUATION METRICS
# =============================================================================

def inverse_transform(y_scaled: np.ndarray, mean_z: float, std_z: float) -> np.ndarray:
    return y_scaled * std_z + mean_z


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = y_true.flatten().astype(np.float64)
    y_pred = y_pred.flatten().astype(np.float64)

    mae = np.mean(np.abs(y_true - y_pred))
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)

    non_zero_mask = y_true != 0

    if non_zero_mask.sum() > 0:
        mape = np.mean(
            np.abs(
                (y_true[non_zero_mask] - y_pred[non_zero_mask])
                / y_true[non_zero_mask]
            )
        ) * 100.0
    else:
        mape = float("nan")

    denom = np.maximum((np.abs(y_true) + np.abs(y_pred)) / 2.0, 1e-8)
    smape = np.mean(np.abs(y_true - y_pred) / denom) * 100.0

    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1.0 - (ss_res / (ss_tot + 1e-8))

    return {
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "MAPE": mape,
        "SMAPE": smape,
        "R2": r2
    }


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    adj_list: list,
    device: torch.device,
    mean_z: float,
    std_z: float,
) -> dict:
    model.eval()

    all_preds = []
    all_true = []

    for batch_x, batch_y in loader:
        batch_x = batch_x.to(device)

        pred = model(batch_x, adj_list)

        all_preds.append(pred.squeeze(-1).detach().cpu().numpy())
        all_true.append(batch_y.squeeze(1).detach().cpu().numpy())

    preds = np.concatenate(all_preds, axis=0)
    trues = np.concatenate(all_true, axis=0)

    preds_denorm = inverse_transform(preds, mean_z, std_z)
    trues_denorm = inverse_transform(trues, mean_z, std_z)

    return compute_metrics(trues_denorm, preds_denorm)


# =============================================================================
# SECTION 6: MODEL-ONLY INFERENCE TIME / MEMORY / PARAMETERS
# =============================================================================

def count_parameters(model: nn.Module):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    param_size_mb = total_params * 4 / (1024 ** 2)

    return total_params, trainable_params, param_size_mb


def measure_model_only_inference_time(
    model: nn.Module,
    loader: DataLoader,
    adj_list: list,
    device: torch.device,
    warmup_steps: int = 20,
    repeats: int = 10,
):
    """
    Measures model-only forward-pass inference time.

    Excludes:
    - CSV loading
    - preprocessing
    - DataLoader iteration during timing
    - CPU-to-GPU transfer during timing
    - metric calculation
    - plotting
    """

    model.eval()

    test_batches_device = []

    with torch.no_grad():
        for batch_x, _ in loader:
            test_batches_device.append(batch_x.to(device))

    if len(test_batches_device) == 0:
        raise ValueError("Test loader is empty. Cannot measure inference time.")

    with torch.no_grad():
        for i in range(warmup_steps):
            batch_x = test_batches_device[i % len(test_batches_device)]
            _ = model(batch_x, adj_list)

    sync_if_cuda(device)

    start = time.perf_counter()

    with torch.no_grad():
        for _ in range(repeats):
            for batch_x in test_batches_device:
                _ = model(batch_x, adj_list)

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


def measure_model_only_peak_memory(
    model: nn.Module,
    loader: DataLoader,
    adj_list: list,
    device: torch.device,
):
    """
    Measures model-only peak memory during inference.

    CUDA result includes:
    - model parameters
    - adjacency supports already on GPU
    - one input batch
    - forward-pass intermediate tensors
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

                pred = model(batch_x, adj_list)

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

                pred = model(batch_x, adj_list)

                _, peak = tracemalloc.get_traced_memory()
                tracemalloc.stop()

                batch_peak_mb = peak / (1024 ** 2)
                peak_mem_mb = max(peak_mem_mb, batch_peak_mb)

                del pred
                del batch_x

                gc.collect()

        return peak_mem_mb


# =============================================================================
# SECTION 7: MAIN
# =============================================================================

if __name__ == "__main__":

    # -----------------------------
    # Static configuration
    # -----------------------------
    DIST_FILE = "distance_matrix (1).npy"
    DATA_FILE = "All_Features_3.csv"

    N_NODES = 197
    M_HISTORY = 24
    M_GRAPHS = 3
    EPOCHS = 15

    KERNEL_TYPE = "chebyshev"
    K_ORDER = 2

    SAVE_PATH = "stmgcn_ems_best_model_only.pth"

    Z_SCORE_COLS = [
        "call_count",
        "lag_1",
        "lag_24",
        "lag_168",
        "roll_mean_24",
        "roll_std_24",
        "Population",
        "INITIAL_SEVERITY_LEVEL_CODE",
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
        "SPECIAL_EVENT_INDICATOR",
    ]

    NUM_FEATURES = len(Z_SCORE_COLS) + len(PASS_THROUGH_COLS)

    best_params = {
        "lr": 0.001,
        "batch_size": 16,
        "lstm_hidden": 64,
        "lstm_layers": 1,
        "gcn_hidden": 32,
        "optimizer": "AdamW",
        "weight_decay": 1e-4,
        "step_size": 7,
        "lr_gamma": 0.7,
    }

    GRAPH_PARAMS = {
        "prox_sigma_sq": None,
        "prox_threshold": 0.5,
        "corr_threshold": 0.3,
        "corr_top_k": 10,
        "ctx_features": ["Population", "INITIAL_SEVERITY_LEVEL_CODE"],
        "ctx_threshold": 0.85,
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 65, flush=True)
    print("ST-MGCN | ZIP-level EMS Call Demand Forecasting", flush=True)
    print("=" * 65, flush=True)
    print(f"Device: {device}", flush=True)

    if device.type == "cuda":
        print(f"GPU name: {torch.cuda.get_device_name(0)}", flush=True)
        print(f"CUDA version used by PyTorch: {torch.version.cuda}", flush=True)

    # -----------------------------
    # Resolve feature names
    # -----------------------------
    df_peek = pd.read_csv(DATA_FILE, nrows=1)
    actual_cols = list(df_peek.columns)

    def resolve_cols(user_list: list, actual_cols: list, skip_set: set = None) -> list:
        skip_set = skip_set or set()
        lower_map = {c.lower(): c for c in actual_cols}
        resolved = []

        for name in user_list:
            if name in skip_set:
                resolved.append(name)
            elif name in actual_cols:
                resolved.append(name)
            elif name.lower() in lower_map:
                resolved.append(lower_map[name.lower()])
            else:
                raise ValueError(
                    f"Feature column '{name}' not found in CSV and is not auto-generated.\n"
                    f"Actual columns: {actual_cols}"
                )

        return resolved

    Z_SCORE_COLS = resolve_cols(Z_SCORE_COLS, actual_cols)
    PASS_THROUGH_COLS = resolve_cols(
        PASS_THROUGH_COLS,
        actual_cols,
        skip_set=_AUTO_GEN_SINCOS_COLS
    )

    GRAPH_PARAMS["ctx_features"] = resolve_cols(
        GRAPH_PARAMS["ctx_features"],
        actual_cols
    )

    NUM_FEATURES = len(Z_SCORE_COLS) + len(PASS_THROUGH_COLS)

    print(f"Features total: {NUM_FEATURES}", flush=True)
    print(f"Nodes: {N_NODES}", flush=True)
    print(f"History: {M_HISTORY}", flush=True)
    print(f"Graphs: {M_GRAPHS}", flush=True)
    print(f"Kernel: {KERNEL_TYPE}, K={K_ORDER}", flush=True)

    # =========================================================================
    # STEP 1: LOAD DATA
    # =========================================================================
    print("\n" + "-" * 65, flush=True)
    print("STEP 1 | Loading EMS data", flush=True)
    print("-" * 65, flush=True)

    (X_train, Y_train), (X_val, Y_val), (X_test, Y_test), \
        (mean_call, std_call), zip_list = load_ems_data(
            csv_path=DATA_FILE,
            z_score_cols=Z_SCORE_COLS,
            pass_through_cols=PASS_THROUGH_COLS,
            history_window=M_HISTORY,
            prediction_horizon=1,
            train_ratio=0.6,
            val_ratio=0.2,
        )

    if len(zip_list) != N_NODES:
        raise ValueError(
            f"Detected {len(zip_list)} nodes, but N_NODES={N_NODES}."
        )

    # =========================================================================
    # STEP 2: BUILD GRAPHS
    # =========================================================================
    print("\n" + "-" * 65, flush=True)
    print("STEP 2 | Constructing multi-graphs", flush=True)
    print("-" * 65, flush=True)

    df_raw = pd.read_csv(DATA_FILE)

    time_col_name = detect_col(df_raw, _TIME_CANDIDATES, "time/datetime")
    zip_col_name = detect_col(df_raw, _ZIP_CANDIDATES, "zip/zipcode")

    df_raw[time_col_name] = pd.to_datetime(df_raw[time_col_name])
    df_raw = df_raw.sort_values([time_col_name, zip_col_name]).reset_index(drop=True)

    T_total = df_raw[time_col_name].nunique()
    n_train_steps = int(T_total * 0.6)

    train_timestamps = sorted(df_raw[time_col_name].unique())[:n_train_steps]
    df_train_only = df_raw[df_raw[time_col_name].isin(train_timestamps)]

    dist_matrix = np.load(DIST_FILE)

    adj_prox = build_proximity_graph(
        dist_matrix,
        sigma_sq=GRAPH_PARAMS["prox_sigma_sq"],
        threshold=GRAPH_PARAMS["prox_threshold"],
    )

    call_col_name = detect_col(df_raw, _CALL_CANDIDATES, "call_count")

    adj_corr = build_demand_corr_graph(
        df=df_train_only,
        zip_col=zip_col_name,
        time_col=time_col_name,
        call_col=call_col_name,
        zip_list=zip_list,
        top_k=GRAPH_PARAMS["corr_top_k"],
        corr_threshold=GRAPH_PARAMS["corr_threshold"],
    )

    adj_ctx = build_context_graph(
        df=df_raw,
        zip_col=zip_col_name,
        feature_cols=GRAPH_PARAMS["ctx_features"],
        zip_list=zip_list,
        sim_threshold=GRAPH_PARAMS["ctx_threshold"],
    )

    # =========================================================================
    # STEP 3: PREPROCESS ADJACENCY MATRICES
    # =========================================================================
    print("\n" + "-" * 65, flush=True)
    print("STEP 3 | Computing Chebyshev supports", flush=True)
    print("-" * 65, flush=True)

    preprocessor = Adj_Preprocessor(
        kernel_type=KERNEL_TYPE,
        K=K_ORDER
    )

    def prep_adj(adj_np: np.ndarray) -> torch.Tensor:
        return preprocessor.process(torch.FloatTensor(adj_np)).to(device)

    adj_prox_t = prep_adj(adj_prox)
    adj_corr_t = prep_adj(adj_corr)
    adj_ctx_t = prep_adj(adj_ctx)

    sta_adj_list = [adj_prox_t, adj_corr_t, adj_ctx_t]

    print(f"Adj tensor shape: {adj_prox_t.shape}", flush=True)

    # =========================================================================
    # STEP 4: BUILD MODEL
    # =========================================================================
    print("\n" + "-" * 65, flush=True)
    print("STEP 4 | Building ST-MGCN model", flush=True)
    print("-" * 65, flush=True)

    sta_kernel_config = {
        "kernel_type": KERNEL_TYPE,
        "K": K_ORDER
    }

    model = ST_MGCN(
        M=M_GRAPHS,
        seq_len=M_HISTORY,
        n_nodes=N_NODES,
        input_dim=NUM_FEATURES,
        lstm_hidden_dim=best_params["lstm_hidden"],
        lstm_num_layers=best_params["lstm_layers"],
        gcn_hidden_dim=best_params["gcn_hidden"],
        sta_kernel_config=sta_kernel_config,
        gconv_use_bias=True,
        gconv_activation=nn.ReLU,
    ).to(device)

    total_params, trainable_params, param_size_mb = count_parameters(model)

    print(f"Total Parameters:     {total_params:,}", flush=True)
    print(f"Trainable Parameters: {trainable_params:,}", flush=True)
    print(f"Parameter Size:       {param_size_mb:.4f} MB assuming float32", flush=True)

    # =========================================================================
    # STEP 5: DATA LOADERS
    # =========================================================================
    train_loader = DataLoader(
        TensorDataset(X_train, Y_train),
        batch_size=best_params["batch_size"],
        shuffle=True,
        drop_last=True,
        num_workers=0
    )

    val_loader = DataLoader(
        TensorDataset(X_val, Y_val),
        batch_size=best_params["batch_size"],
        shuffle=False,
        num_workers=0
    )

    TEST_BATCH_SIZE = 50

    test_loader = DataLoader(
        TensorDataset(X_test, Y_test),
        batch_size=TEST_BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    print(f"Test samples: {len(test_loader.dataset)}", flush=True)
    print(f"Test batches: {len(test_loader)}", flush=True)
    print(f"Test batch size: {TEST_BATCH_SIZE}", flush=True)

    # =========================================================================
    # STEP 6: TRAINING LOOP
    # =========================================================================
    print("\n" + "-" * 65, flush=True)
    print("STEP 6 | Training", flush=True)
    print("-" * 65, flush=True)

    optimizer = getattr(optim, best_params["optimizer"])(
        model.parameters(),
        lr=best_params["lr"],
        weight_decay=best_params["weight_decay"],
    )

    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=best_params["step_size"],
        gamma=best_params["lr_gamma"],
    )

    loss_fn = nn.MSELoss()

    best_val_loss = float("inf")
    train_losses = []
    val_losses = []

    for epoch in range(EPOCHS):
        model.train()

        train_loss_sum = 0.0
        train_count = 0

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()

            pred = model(batch_x, sta_adj_list)

            loss = loss_fn(
                pred.squeeze(-1),
                batch_y.squeeze(1)
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=5.0
            )

            optimizer.step()

            train_loss_sum += loss.item() * batch_x.size(0)
            train_count += batch_x.size(0)

        epoch_train_loss = train_loss_sum / train_count
        train_losses.append(epoch_train_loss)

        model.eval()

        val_loss_sum = 0.0
        val_count = 0

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)

                pred = model(batch_x, sta_adj_list)

                loss = loss_fn(
                    pred.squeeze(-1),
                    batch_y.squeeze(1)
                )

                val_loss_sum += loss.item() * batch_x.size(0)
                val_count += batch_x.size(0)

        epoch_val_loss = val_loss_sum / val_count
        val_losses.append(epoch_val_loss)

        scheduler.step()

        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(model.state_dict(), SAVE_PATH)

        if (epoch + 1) % 10 == 0:
            print(
                f"Epoch {epoch + 1:3d}/{EPOCHS} | "
                f"Train Loss: {epoch_train_loss:.4f} | "
                f"Val Loss: {epoch_val_loss:.4f} | "
                f"LR: {scheduler.get_last_lr()[0]:.2e}",
                flush=True
            )

    print(f"\nBest val MSE loss: {best_val_loss:.4f} -> saved to '{SAVE_PATH}'", flush=True)

    # =========================================================================
    # STEP 7: FINAL EVALUATION
    # =========================================================================
    print("\n" + "-" * 65, flush=True)
    print("STEP 7 | Final evaluation on test set", flush=True)
    print("-" * 65, flush=True)

    model.load_state_dict(
        torch.load(SAVE_PATH, map_location=device)
    )

    model.eval()

    # Remove optimizer objects before memory measurement.
    try:
        del optimizer
        del scheduler
        del loss_fn
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

    # -----------------------------
    # Model-only inference time
    # -----------------------------
    model_only_ms_batch, model_only_ms_sample, sec_per_test_pass = measure_model_only_inference_time(
        model=model,
        loader=test_loader,
        adj_list=sta_adj_list,
        device=device,
        warmup_steps=20,
        repeats=10
    )

    print("\nMODEL-ONLY INFERENCE TIME", flush=True)
    print("-" * 65, flush=True)
    print(f"Model-only Inference Time (ms per batch):  {model_only_ms_batch:.4f}", flush=True)
    print(f"Model-only Inference Time (ms per sample): {model_only_ms_sample:.6f}", flush=True)
    print(f"Seconds per full test pass:                {sec_per_test_pass:.4f}", flush=True)

    # -----------------------------
    # Model-only inference memory
    # -----------------------------
    model_only_peak_mem_mb = measure_model_only_peak_memory(
        model=model,
        loader=test_loader,
        adj_list=sta_adj_list,
        device=device
    )

    print("\nMODEL-ONLY INFERENCE MEMORY", flush=True)
    print("-" * 65, flush=True)

    if device.type == "cuda":
        print(
            f"Model-only Peak GPU Memory During Inference (MB): {model_only_peak_mem_mb:.2f}",
            flush=True
        )
    else:
        print(
            f"Model-only Peak CPU Memory During Inference (MB): {model_only_peak_mem_mb:.2f}",
            flush=True
        )

    # -----------------------------
    # Accuracy metrics
    # -----------------------------
    metrics = evaluate_model(
        model=model,
        loader=test_loader,
        adj_list=sta_adj_list,
        device=device,
        mean_z=mean_call,
        std_z=std_call
    )

    print("\nFINAL TEST RESULTS", flush=True)
    print("=" * 65, flush=True)
    print(f"MAE:   {metrics['MAE']:.4f}", flush=True)
    print(f"RMSE:  {metrics['RMSE']:.4f}", flush=True)
    print(f"MSE:   {metrics['MSE']:.4f}", flush=True)
    print(f"MAPE:  {metrics['MAPE']:.2f}%", flush=True)
    print(f"SMAPE: {metrics['SMAPE']:.2f}%", flush=True)
    print(f"R²:    {metrics['R2']:.4f}", flush=True)

    print("\nCOMPLEXITY SUMMARY", flush=True)
    print("=" * 65, flush=True)
    print(f"Total Parameters:        {total_params:,}", flush=True)
    print(f"Trainable Parameters:    {trainable_params:,}", flush=True)
    print(f"Parameter Size MB:       {param_size_mb:.4f}", flush=True)
    print(f"Inference ms/batch:      {model_only_ms_batch:.4f}", flush=True)
    print(f"Inference ms/sample:     {model_only_ms_sample:.6f}", flush=True)
    print(f"Peak Inference Memory:   {model_only_peak_mem_mb:.2f} MB", flush=True)

    print("\n" + "=" * 65, flush=True)
    print("ST-MGCN training and model-only evaluation complete.", flush=True)
    print(f"Best model saved to: {SAVE_PATH}", flush=True)
    print(f"RUN_ID: {RUN_ID}", flush=True)
    print("=" * 65, flush=True)
