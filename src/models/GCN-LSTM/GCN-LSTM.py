import pandas as pd

import numpy as np
import time
import torch
from scipy.spatial.distance import cdist

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler

import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from torch_geometric.nn import GCNConv

W = np.load("data/adj_matrix_gaussian.npy")

def build_topk_edge_index_weight(W, k=10):

    num_nodes = W.shape[0]

    undirected_edges = {}

    for i in range(num_nodes):
        neighbors = np.argsort(W[i])[::-1]

        neighbors = neighbors[neighbors != i][:k]

        for j in neighbors:
            a, b = min(i, j), max(i, j)

            if (a, b) not in undirected_edges:
                undirected_edges[(a, b)] = W[i, j]

    edges = []
    weights = []

    for (i, j), w in undirected_edges.items():
        edges.append([i, j])
        edges.append([j, i])
        weights.append(w)
        weights.append(w)

    edge_index = torch.tensor(edges, dtype=torch.long).T
    edge_weight = torch.tensor(weights, dtype=torch.float32)

    return edge_index, edge_weight

edge_index, edge_weight = build_topk_edge_index_weight(W, k=10)

print("edge_index shape:", edge_index.shape, flush=True)
print("edge_weight shape:", edge_weight.shape, flush=True)

assert edge_index.dtype == torch.long
assert edge_weight.dtype == torch.float32
assert edge_index.shape[0] == 2
assert edge_index.shape[1] == edge_weight.shape[0]
assert edge_index.max().item() < 197

df = pd.read_csv('data/All_Features_3.csv')

df['datetime'] = pd.to_datetime(df['datetime'])

NULL_CHECK_COLS = ['lag_1', 'lag_24', 'lag_168', 'roll_mean_24', 'roll_std_24']
df = df.dropna(subset=NULL_CHECK_COLS).reset_index(drop=True)

# Generate cyclic time features
dt_ref = df['datetime'].dt
df['hour_sin']  = np.sin(2 * np.pi * dt_ref.hour / 24)
df['hour_cos']  = np.cos(2 * np.pi * dt_ref.hour / 24)
df['dow_sin']   = np.sin(2 * np.pi * dt_ref.dayofweek / 7)
df['dow_cos']   = np.cos(2 * np.pi * dt_ref.dayofweek / 7)
df['month_sin'] = np.sin(2 * np.pi * dt_ref.month / 12)
df['month_cos'] = np.cos(2 * np.pi * dt_ref.month / 12)
df['week_sin']  = np.sin(2 * np.pi * dt_ref.isocalendar().week.astype(int) / 52)
df['week_cos']  = np.cos(2 * np.pi * dt_ref.isocalendar().week.astype(int) / 52)

df = df.sort_values(['ZIPCODE', 'datetime']).reset_index(drop=True)

NUM_ZIPS = 197

# Z-scored features 
Z_SCORE_COLS = [
    'call_count',
    'lag_1', 'lag_24', 'lag_168',
    'roll_mean_24', 'roll_std_24',
    'Population', 'INITIAL_SEVERITY_LEVEL_CODE'
]

# Pass-through features 
PASS_THROUGH_COLS = [
    'hour_sin', 'hour_cos',
    'dow_sin', 'dow_cos',
    'month_sin', 'month_cos',
    'week_sin', 'week_cos',
    'is_Covid', 'SPECIAL_EVENT_INDICATOR'
]

FEATURE_COLS = Z_SCORE_COLS + PASS_THROUGH_COLS

TARGET_COL = 'call_count'

def prepare_df_2018(df):
    df = df.sort_values(['datetime', 'ZIPCODE']).copy()
    zips = sorted(df['ZIPCODE'].unique())
    zip2idx = {z: i for i, z in enumerate(zips)}
    df['zip_idx'] = df['ZIPCODE'].map(zip2idx)
    df = df.sort_values(['datetime', 'zip_idx']).reset_index(drop=True)

    assert len(df) % NUM_ZIPS == 0, "Each hour must contain all ZIPs"
    return df

df_2018 = prepare_df_2018(df)
timestamps = df_2018['datetime'].unique()

TRAIN_RATIO = 0.6
VAL_RATIO   = 0.2

unique_times = sorted(df_2018['datetime'].unique())
n_total      = len(unique_times)
train_end    = unique_times[int(n_total * TRAIN_RATIO) - 1]
val_end      = unique_times[int(n_total * (TRAIN_RATIO + VAL_RATIO)) - 1]

train_mask = df_2018['datetime'] <= train_end

# Z-score all Z_SCORE_COLS 
scaler = StandardScaler()
scaler.fit(df_2018.loc[train_mask, Z_SCORE_COLS].values)
df_2018.loc[:, Z_SCORE_COLS] = scaler.transform(df_2018[Z_SCORE_COLS].values)

class EMSWindowDataset(Dataset):
    def __init__(self, df, feature_cols, target_col,
                 num_zips=197, window=24, horizon=1):
        self.df = df
        self.feature_cols = feature_cols
        self.target_col = target_col
        self.num_zips = num_zips
        self.window = window
        self.horizon = horizon
        self.timestamps = df['datetime'].unique()
        self.length = len(self.timestamps) - window - horizon + 1

    def __len__(self):
        return self.length

    def _slice_hour(self, t):
        s = t * self.num_zips
        return self.df.iloc[s:s + self.num_zips]

    def __getitem__(self, idx):
        X = torch.stack([
            torch.tensor(
                self._slice_hour(idx + t)[self.feature_cols].values,
                dtype=torch.float32
            )
            for t in range(self.window)
        ])  # (L, N, F)

        y = torch.tensor(
            self._slice_hour(idx + self.window + self.horizon - 1)[self.target_col].values,
            dtype=torch.float32
        )  # (N,)

        return X, y
    
WINDOW = 24
HORIZON = 1

dataset = EMSWindowDataset(
    df_2018, FEATURE_COLS, TARGET_COL,
    num_zips=NUM_ZIPS, window=WINDOW, horizon=HORIZON
)

target_times = [
    timestamps[i + WINDOW + HORIZON - 1]
    for i in range(len(dataset))
]

train_idx, val_idx, test_idx = [], [], []

for i, t in enumerate(target_times):
    if t <= train_end:
        train_idx.append(i)
    elif t <= val_end:
        val_idx.append(i)
    else:
        test_idx.append(i)

train_ds = Subset(dataset, train_idx)
val_ds   = Subset(dataset, val_idx)
test_ds  = Subset(dataset, test_idx)


def collate_fn(batch):
    X, y = zip(*batch)
    return torch.stack(X), torch.stack(y)


train_loader = DataLoader(train_ds, batch_size=8, shuffle=True,
                          num_workers=4, pin_memory=True, collate_fn=collate_fn)

val_loader   = DataLoader(val_ds, batch_size=8, shuffle=False,
                          num_workers=2, pin_memory=True, collate_fn=collate_fn)

test_loader  = DataLoader(test_ds, batch_size=8, shuffle=False,
                          num_workers=2, pin_memory=True, collate_fn=collate_fn)

def batch_edge_index(edge_index, edge_weight, B, N, device):
    E = edge_index.size(1)
    offsets = torch.arange(B, device=device).repeat_interleave(E) * N
    be_idx = edge_index.repeat(1, B) + offsets
    be_w = edge_weight.repeat(B)
    return be_idx, be_w
    
class GCN_LSTM(nn.Module):
    def __init__(self, in_dim, gcn_dim=32, lstm_dim=64):
        super().__init__()
        self.gcn = GCNConv(in_dim, gcn_dim, normalize=True)
        self.lstm = nn.LSTM(gcn_dim, lstm_dim, batch_first=True)
        self.fc = nn.Linear(lstm_dim, 1)

    def forward(self, X, edge_index, edge_weight):
        B, L, N, F = X.shape
        device = X.device

        be_idx, be_w = batch_edge_index(edge_index, edge_weight, B, N, device)

        gcn_seq = []
        for t in range(L):
            xt = X[:, t].reshape(B * N, F)
            ht = self.gcn(xt, be_idx, be_w)
            gcn_seq.append(ht.view(B, N, -1))

        H = torch.stack(gcn_seq, dim=1)          # (B, L, N, gcn_dim)
        H = H.permute(0, 2, 1, 3).reshape(B*N, L, -1)

        out, _ = self.lstm(H)
        y = self.fc(out[:, -1]).squeeze(-1)
        return y.view(B, N)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

edge_index = edge_index.to(device)
edge_weight = edge_weight.to(device)

model = GCN_LSTM(len(FEATURE_COLS)).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.7)
loss_fn = nn.MSELoss()

for epoch in range(1):
    model.train()
    tr_loss = 0

    # Reset peak memory stats before training starts
    use_cuda = device.type == "cuda"
    if use_cuda:
        torch.cuda.reset_peak_memory_stats(device)

    for X, y in train_loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        y_hat = model(X, edge_index, edge_weight)
        loss = loss_fn(y_hat, y)
        loss.backward()
        optimizer.step()
        tr_loss += loss.item()

    model.eval()
    va_loss = 0
    with torch.no_grad():
        for X, y in val_loader:
            X, y = X.to(device), y.to(device)
            va_loss += loss_fn(model(X, edge_index, edge_weight), y).item()

    print(f"Epoch {epoch+1:02d} | Train {tr_loss/len(train_loader):.4f} | Val {va_loss/len(val_loader):.4f}", flush=True)
    scheduler.step()

# Capture peak memory after training completes

peak_train_memory_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)

# Profiling helpers
def count_parameters(model):
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def measure_inference(model, loader, device, edge_index, edge_weight, n_warmup_batches=5):
    model.eval()
    use_cuda = device.type == "cuda"

    latencies        = []
    total_snapshots  = 0

    with torch.no_grad():
        for batch_idx, (X, y) in enumerate(loader):
            X = X.to(device)
            B = X.size(0)

            #warm-up
            if batch_idx < n_warmup_batches:
                _ = model(X, edge_index, edge_weight)
                if use_cuda:
                    torch.cuda.synchronize(device)
                continue

            if use_cuda:
                torch.cuda.synchronize(device)

            t0 = time.perf_counter()
            _  = model(X, edge_index, edge_weight)

            if use_cuda:
                torch.cuda.synchronize(device)

            t1 = time.perf_counter()

            latencies.append((t1 - t0) / B)
            total_snapshots += B


    avg_latency_ms = np.mean(latencies) * 1_000   # ms

    return avg_latency_ms, total_snapshots


def print_profiling_report(model, test_loader, device, edge_index, edge_weight, peak_train_memory_mb):
    print("\n" + "=" * 60, flush=True)
    print("  MODEL PROFILING REPORT", flush=True)
    print("=" * 60, flush=True)

    # Parameters
    total_params, trainable_params = count_parameters(model)
    print(f"  Total parameters     : {total_params:,}", flush=True)
    print(f"  Trainable parameters : {trainable_params:,}", flush=True)

    #Inference time
    print("\n  Measuring inference latency …", flush=True)
    avg_lat_ms, n_snapshots = measure_inference(
        model, test_loader, device, edge_index, edge_weight
    )

    print(f"  Snapshots measured   : {n_snapshots:,}", flush=True)
    print(f"  Avg latency/snapshot : {avg_lat_ms:.4f} ms", flush=True)

    # Training peak memory
    print(f"GPU peak train mem: {peak_train_memory_mb:.2f} MB", flush=True)



# Profiling report  (parameters + inference time + memory)
print_profiling_report(model, test_loader, device, edge_index, edge_weight, peak_train_memory_mb)


# Recover call_count scaler stats (first column in Z_SCORE_COLS)
cc_mean = scaler.mean_[0]
cc_std  = scaler.scale_[0]

model.eval()

y_true, y_pred = [], []

with torch.no_grad():
    for X, y in test_loader:
        X = X.to(device)
        y = y.to(device)
        y_hat = model(X, edge_index, edge_weight)
        # Inverse z-score transform on call_count
        y_true.append((y.cpu().numpy() * cc_std + cc_mean))
        y_pred.append((y_hat.cpu().numpy() * cc_std + cc_mean))

y_true = np.concatenate(y_true, axis=0).flatten()
y_pred = np.concatenate(y_pred, axis=0).flatten()

test_mae  = mean_absolute_error(y_true, y_pred)
test_rmse = np.sqrt(mean_squared_error(y_true, y_pred))
test_mse  = test_rmse ** 2
test_r2   = r2_score(y_true, y_pred)