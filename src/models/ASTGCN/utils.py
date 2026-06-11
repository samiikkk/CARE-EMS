import torch
import numpy as np
import pandas as pd


# GRAPH PROCESSING
def get_chebyshev_laplacian(distance_file, n_nodes, k_cheb):
    """
    Loads distance matrix and converts to Scaled Laplacian for Chebyshev Conv.
    Includes Gaussian Kernel and Normalized Laplacian steps.
    """
    print(f"Loading graph from {distance_file}...", flush=True)
    try:
        dist_matrix = np.load(distance_file)
    except FileNotFoundError:
        print(f"ERROR: File {distance_file} not found. Using random fallback.", flush=True)
        dist_matrix = np.random.rand(n_nodes, n_nodes)

    if dist_matrix.shape[0] != n_nodes:
        raise ValueError(f"Distance matrix shape {dist_matrix.shape} does not match N_NODES={n_nodes}")

    # Gaussian Kernel (Distance -> Weight)
    valid_dists = dist_matrix[dist_matrix > 0]
    sigma = np.std(valid_dists) if len(valid_dists) > 0 else 1.0
    epsilon = 0.5

    W = np.exp(- (dist_matrix**2) / (sigma**2))
    W[W < epsilon] = 0
    np.fill_diagonal(W, 0)

    # Normalized Laplacian
    D = np.array(np.sum(W, axis=1))
    D_inv_sqrt = np.power(D, -0.5)
    D_inv_sqrt[np.isinf(D_inv_sqrt)] = 0.
    D_mat_inv_sqrt = np.diag(D_inv_sqrt)

    L = np.eye(n_nodes) - np.dot(np.dot(D_mat_inv_sqrt, W), D_mat_inv_sqrt)

    # Scaled Laplacian (for Chebyshev)
    lambda_max = 2.0
    L_tilde = (2 * L) / lambda_max - np.eye(n_nodes)

    return torch.from_numpy(L_tilde.astype(np.float32))


# DATA LOADING
def load_ems_data(csv_path, z_score_cols, pass_through_cols, history_window, prediction_horizon=1, train_ratio=0.6, val_ratio=0.2):

    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"❌ Error: The file '{csv_path}' was not found.")

    if 'datetime' in df.columns:
        df['datetime'] = pd.to_datetime(df['datetime'])
        df = df.sort_values(['datetime', 'ZIPCODE'])
    else:
        raise ValueError("DataFrame must contain 'datetime' column")

    dt_ref = df['datetime'].dt

    time_feats_config = {
        'hour': 24,
        'dayofweek': 7,
        'month': 12,
        'weekofyear': 52
    }

    print("Checking for missing time features...", flush=True)
    for name, period in time_feats_config.items():
        sin_name = f"{name}_sin"
        cos_name = f"{name}_cos"

        if (sin_name in pass_through_cols) and (sin_name not in df.columns):
            print(f"  -> Generating {sin_name} & {cos_name}...", flush=True)

            if name == 'weekofyear':
                raw_val = dt_ref.isocalendar().week.astype(int)
            else:
                raw_val = getattr(dt_ref, name)
            df[sin_name] = np.sin(2 * np.pi * raw_val / period)
            df[cos_name] = np.cos(2 * np.pi * raw_val / period)


    feature_cols = z_score_cols + pass_through_cols
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f" Missing columns in CSV (could not auto-generate): {missing}")

    data_list = []
    min_time_steps = float('inf')
    for feat in feature_cols:
        pivot_df = df.pivot_table(index='datetime', columns='ZIPCODE', values=feat, aggfunc='mean')
        min_time_steps = min(min_time_steps, len(pivot_df))
    for feat in feature_cols:
        pivot_df = df.pivot_table(index='datetime', columns='ZIPCODE', values=feat, aggfunc='mean')
        pivot_df = pivot_df.interpolate(method='linear', axis=0).fillna(0)
        pivot_df = pivot_df.reindex(sorted(pivot_df.columns), axis=1)
        pivot_df = pivot_df.iloc[:min_time_steps, :]
        data_list.append(pivot_df.values)

    data_raw = np.stack(data_list, axis=-1)
    n_total = data_raw.shape[0]
    train_end = int(n_total * train_ratio)
    val_end = int(n_total * (train_ratio + val_ratio))

    train_data = data_raw[:train_end]
    val_data = data_raw[train_end:val_end]
    test_data = data_raw[val_end:]
    num_z_cols = len(z_score_cols)
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

    print(f"\n✓ Normalization Logic Applied:", flush=True)
    print(f"  - Z-Scored {num_z_cols} features", flush=True)
    print(f"  - Passed Through {len(pass_through_cols)} features", flush=True)
    print(f"  - Target (call_count) Mean: {mean_z[0,0,0]:.4f}, Std: {std_z[0,0,0]:.4f}", flush=True)

    # Sliding Windows
    def create_windows(data, history, horizon):
        X, Y = [], []
        num_samples = len(data) - history - horizon + 1

        if num_samples <= 0:
            return np.empty((0, history, data.shape[1])), np.empty((0, horizon))

        for i in range(num_samples):
            x_inst = data[i : i + history].transpose(2, 0, 1)
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


# BASIS CHEBYSHEV POLYNOMIALS
def cheb_polynomial(L_tilde, K):
    """Return list of K Chebyshev basis matrices as numpy arrays."""
    N = L_tilde.shape[0]
    cheb_list = [np.eye(N), L_tilde.copy()]
    for i in range(2, K):
        cheb_list.append(2 * L_tilde @ cheb_list[-1] - cheb_list[-2])
    return cheb_list
