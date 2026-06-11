import torch
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse import linalg


# GRAPH PROCESSING
def get_adjacency_matrix(distance_file, n_nodes, epsilon=0.5):

    #Loads distance matrix and converts to a weighted directed adjacency matrix using a thresholded Gaussian kernel
  
    print(f"Loading graph from {distance_file}...", flush=True)
    try:
        dist_matrix = np.load(distance_file).astype(np.float32)
    except FileNotFoundError:
        print(f"ERROR: File {distance_file} not found. Using random fallback.")
        dist_matrix = np.random.rand(n_nodes, n_nodes).astype(np.float32)

    if dist_matrix.shape[0] != n_nodes:
        raise ValueError(
            f"Distance matrix shape {dist_matrix.shape} does not match N_NODES={n_nodes}"
        )

    # sigma^2 from non-zero (off-diagonal) distances
    valid_dists = dist_matrix[dist_matrix > 0]
    sigma2 = np.var(valid_dists) if len(valid_dists) > 0 else 1.0

    # Gaussian kernel
    W = np.exp(-(dist_matrix ** 2) / sigma2)

    #entries below epsilon become 0
    W[W < epsilon] = 0.0

    # The diagonal is exp(0)=1 which remains here.

    print(f"  Adjacency matrix: {(W > 0).sum()} non-zero entries "
          f"(sparsity {1 - (W > 0).mean():.2%})", flush=True)
    return W          # shape (N, N), values in [0, 1]


#sparse transition matrices (used by DCGRUCell) 

def _calculate_random_walk_matrix(adj_mx):
    #Row-normalised transition matrix  D_out^{-1} W  (scipy sparse COO)
    adj_mx = sp.coo_matrix(adj_mx)
    d = np.array(adj_mx.sum(1)).flatten()
    d_inv = np.where(d != 0, 1.0 / d, 0.0)
    d_mat_inv = sp.diags(d_inv)
    return d_mat_inv.dot(adj_mx).tocoo().astype(np.float32)


def build_sparse_supports(adj_mx, filter_type="dual_random_walk"):

    #Returns a list of torch sparse tensors that will be used for graph diffusion.

    supports_scipy = []
    if filter_type == "random_walk":
        supports_scipy.append(_calculate_random_walk_matrix(adj_mx))
    elif filter_type == "dual_random_walk":
        supports_scipy.append(_calculate_random_walk_matrix(adj_mx))
        supports_scipy.append(_calculate_random_walk_matrix(adj_mx.T))
    else:
        raise ValueError(f"Unknown filter_type: {filter_type}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sparse_tensors = []
    for S in supports_scipy:
        S = S.tocoo()
        indices = np.column_stack((S.row, S.col))
        # row-major ordering 
        indices = indices[np.lexsort((indices[:, 0], indices[:, 1]))]
        t = torch.sparse_coo_tensor(
            indices.T, S.data.astype(np.float32), S.shape, device=device
        )
        sparse_tensors.append(t)
    return sparse_tensors


# DATA LOADING  
def load_ems_data(csv_path, z_score_cols, pass_through_cols,
                  history_window, prediction_horizon=1,
                  train_ratio=0.6, val_ratio=0.2):

    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"Error: '{csv_path}' not found.")

    if 'datetime' in df.columns:
        df['datetime'] = pd.to_datetime(df['datetime'])
        df = df.sort_values(['datetime', 'ZIPCODE'])
    else:
        raise ValueError("DataFrame must contain 'datetime' column")

    dt_ref = df['datetime'].dt

    time_feats_config = {
        'hour': 24, 'dayofweek': 7, 'month': 12, 'weekofyear': 52
    }
    print("Checking for missing time features...", flush=True)
    for name, period in time_feats_config.items():
        sin_name, cos_name = f"{name}_sin", f"{name}_cos"
        if (sin_name in pass_through_cols) and (sin_name not in df.columns):
            print(f"  -> Generating {sin_name} & {cos_name}...", flush=True)
            raw_val = (dt_ref.isocalendar().week.astype(int)
                       if name == 'weekofyear' else getattr(dt_ref, name))
            df[sin_name] = np.sin(2 * np.pi * raw_val / period)
            df[cos_name] = np.cos(2 * np.pi * raw_val / period)

    feature_cols = z_score_cols + pass_through_cols
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    data_list = []
    min_time_steps = float('inf')
    for feat in feature_cols:
        pv = df.pivot_table(index='datetime', columns='ZIPCODE', values=feat, aggfunc='mean')
        min_time_steps = min(min_time_steps, len(pv))
    for feat in feature_cols:
        pv = df.pivot_table(index='datetime', columns='ZIPCODE', values=feat, aggfunc='mean')
        pv = pv.interpolate(method='linear', axis=0).fillna(0)
        pv = pv.reindex(sorted(pv.columns), axis=1).iloc[:min_time_steps, :]
        data_list.append(pv.values)

    data_raw = np.stack(data_list, axis=-1)       # (T, N, F)
    n_total = data_raw.shape[0]
    train_end = int(n_total * train_ratio)
    val_end   = int(n_total * (train_ratio + val_ratio))

    train_data = data_raw[:train_end]
    val_data   = data_raw[train_end:val_end]
    test_data  = data_raw[val_end:]

    num_z = len(z_score_cols)
    mean_z = np.mean(train_data[..., :num_z], axis=(0, 1), keepdims=True)
    std_z  = np.std (train_data[..., :num_z], axis=(0, 1), keepdims=True)
    std_z[std_z < 1e-5] = 1.0

    def normalize(arr):
        z = (arr[..., :num_z] - mean_z) / std_z
        return np.concatenate([z, arr[..., num_z:]], axis=-1)

    train_norm = normalize(train_data)
    val_norm   = normalize(val_data)
    test_norm  = normalize(test_data)

    print(f"\n✓ Normalisation applied: z-scored {num_z} cols, "
          f"pass-through {len(pass_through_cols)} cols", flush=True)
    print(f"  Target mean: {mean_z[0,0,0]:.4f}  std: {std_z[0,0,0]:.4f}", flush=True)

    def create_windows(data, history, horizon):

        #Returns:
        #X : (samples, history, N, F) 
        #Y : (samples, horizon, N) 

        X, Y = [], []
        for i in range(len(data) - history - horizon + 1):
            X.append(data[i : i + history])                       # (H, N, F)
            Y.append(data[i + history : i + history + horizon, :, 0])  # (horizon, N)
        if not X:
            return (np.empty((0, history, data.shape[1], data.shape[2])),
                    np.empty((0, horizon, data.shape[1])))
        return np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32)

    print("\n Creating sliding windows", flush=True)
    X_train, Y_train = create_windows(train_norm, history_window, prediction_horizon)
    X_val,   Y_val   = create_windows(val_norm,   history_window, prediction_horizon)
    X_test,  Y_test  = create_windows(test_norm,  history_window, prediction_horizon)

    to_t = lambda a: torch.from_numpy(a)
    return (
        (to_t(X_train), to_t(Y_train)),
        (to_t(X_val),   to_t(Y_val)),
        (to_t(X_test),  to_t(Y_test)),
        (mean_z, std_z),
    )
