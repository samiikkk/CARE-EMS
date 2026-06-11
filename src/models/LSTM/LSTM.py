import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
import torch.nn as nn
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import csv
import time

# Data Loading & Preprocessing
df = pd.read_csv('data/All_Features_3.csv')
df['datetime'] = pd.to_datetime(df['datetime'])

# Remove null rows 
NULL_CHECK_COLS = ['lag_1', 'lag_24', 'lag_168', 'roll_mean_24', 'roll_std_24']
df = df.dropna(subset=NULL_CHECK_COLS).reset_index(drop=True)
print(f'Rows after null removal: {len(df)}', flush=True)

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
print(f'original df shape : {df.shape}', flush=True)

# Feature Columns
Z_SCORE_COLS = [
    'call_count',
    'lag_1', 'lag_24', 'lag_168',
    'roll_mean_24', 'roll_std_24',
    'Population', 'INITIAL_SEVERITY_LEVEL_CODE'
]

PASS_THROUGH_COLS = [
    'hour_sin', 'hour_cos',
    'dow_sin', 'dow_cos',
    'month_sin', 'month_cos',
    'week_sin', 'week_cos',
    'is_Covid', 'SPECIAL_EVENT_INDICATOR'
]

FEATURE_COLS = Z_SCORE_COLS + PASS_THROUGH_COLS
TARGET_COL   = 'call_count'

TRAIN_RATIO = 0.6
VAL_RATIO   = 0.2

unique_times = sorted(df['datetime'].unique())
n_total      = len(unique_times)
train_end    = unique_times[int(n_total * TRAIN_RATIO) - 1]
val_end      = unique_times[int(n_total * (TRAIN_RATIO + VAL_RATIO)) - 1]

# Z-score scaling
train_mask = df['datetime'] <= train_end

scaler = StandardScaler()
scaler.fit(df.loc[train_mask, Z_SCORE_COLS].values)
df.loc[:, Z_SCORE_COLS] = scaler.transform(df[Z_SCORE_COLS].values)

print('data scaled', flush=True)

# Dataset  (sliding-window)
class EMSData(Dataset):
    def __init__(self, df, timestep=24, mode='train'):
        super().__init__()
        self.timestep = timestep
        self.mode = mode
        self.df = df

        self.indices = []
        self._prepare_indices()

    def _prepare_indices(self):
        print('inside prepare indices', flush=True)
        for zipcode, group in self.df.groupby('ZIPCODE'):
            group = group.copy()

            if self.mode == 'train':
                subset = group[group['datetime'] <= train_end]

            elif self.mode == 'val':
                subset = group[(group['datetime'] > train_end) &
                               (group['datetime'] <= val_end)]

            else:  # test
                subset = group[group['datetime'] > val_end]

            start_idx = subset.index.min()
            end_idx   = subset.index.max()

            # Making sure we have a "next" row to predict
            for i in range(start_idx, end_idx):
                if i + self.timestep < end_idx + 1:
                    self.indices.append(i)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        start      = self.indices[idx]
        window     = self.df.iloc[start:start + self.timestep]
        target_row = self.df.iloc[start + self.timestep]  # the record 'just after' the window

        # Input: explicit FEATURE_COLS 
        x = torch.tensor(window[FEATURE_COLS].values, dtype=torch.float32)
        # Target: next record's call_count (z-scored but we inverse-transform at eval time)
        y = torch.tensor(target_row[TARGET_COL], dtype=torch.float32)

        return x, y
    
# Profiling helpers
def count_parameters(model):
    total      = sum(p.numel() for p in model.parameters())
    trainable  = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def measure_inference(model, loader, device, n_warmup_batches=5):
    model.eval()
    use_cuda = device.type == "cuda"

    latencies   = []    
    total_samples = 0

    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(loader):
            x      = x.to(device)
            y      = y.to(device)
            bs     = x.size(0)

            #warm-up
            if batch_idx < n_warmup_batches:
                _ = model(x)
                if use_cuda:
                    torch.cuda.synchronize(device)
                continue

            if use_cuda:
                torch.cuda.synchronize(device)

            t0 = time.perf_counter()
            _  = model(x)

            if use_cuda:
                torch.cuda.synchronize(device)

            t1 = time.perf_counter()

            latencies.append((t1 - t0) / bs)  
            total_samples += bs

    avg_latency_ms = np.mean(latencies) * 1_000   # in ms

    return avg_latency_ms, total_samples


def print_profiling_report(model, test_loader, device, peak_train_memory_mb):
    print("\n" + "=" * 60, flush=True)
    print("  MODEL PROFILING REPORT", flush=True)
    print("=" * 60, flush=True)

    #Parameters 
    total_params, trainable_params = count_parameters(model)
    print(f"  Total parameters     : {total_params:,}", flush=True)
    print(f"  Trainable parameters : {trainable_params:,}", flush=True)

    #Inference time & memory 
    print("\n  Measuring inference latency", flush=True)
    avg_lat_ms, n_samples = measure_inference(model, test_loader, device)

    print(f"  Samples measured     : {n_samples:,}", flush=True)
    print(f"  Avg latency / sample : {avg_lat_ms:.4f} ms", flush=True)

    print(f"  GPU peak memory: {peak_train_memory_mb:.2f} MB", flush=True)

    print("=" * 60 + "\n", flush=True)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

timestep = 24  

train_dataset = EMSData(df, timestep=timestep, mode='train')
print('train data created', flush=True)
val_dataset   = EMSData(df, timestep=timestep, mode='val')
print('val data created', flush=True)
test_dataset  = EMSData(df, timestep=timestep, mode='test')
print('test data created', flush=True)

input_dim_train_dataset = train_dataset[0][0].shape
print(f"Input dimension: {input_dim_train_dataset}", flush=True)

batch_size = 64

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=False,
    drop_last=True,
    pin_memory=True,
    num_workers=4
)


val_loader  = DataLoader(val_dataset,  batch_size=batch_size, shuffle=False, drop_last=True, pin_memory=True, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True, pin_memory=True, num_workers=4)

# Taking one batch to inspect shapes
batch_x, batch_y = next(iter(train_loader))
print(f"One batch X shape: {batch_x.shape}", flush=True)  # (batch_size, timestep, n_features)
print(f"One batch y shape: {batch_y.shape}", flush=True)  # (batch_size,)

input_dim = batch_x.shape[-1]
print(f"Detected input_dim (n_features): {input_dim}", flush=True)

# Model  
class EMSLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=3, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )
        self.fc1 = nn.Linear(hidden_dim, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        # x: (batch, timestep, features)
        out, _ = self.lstm(x)       # out: (batch, timestep, hidden_dim)
        out = out[:, -1, :]         # last timestep -> (batch, hidden_dim)
        out = self.fc1(out)
        out = self.fc2(out)         # (batch, 1)
        return out.squeeze(-1)      # (batch,)


model = EMSLSTM(input_dim=input_dim).to(device)

# Testing one forward pass
batch_x = batch_x.to(device)
batch_y = batch_y.to(device)

with torch.no_grad():
    preds = model(batch_x)
    print(f"Preds shape: {preds.shape}", flush=True)
    print(f"First 5 preds: {preds[:5]}", flush=True)
    print(f"First 5 targets: {batch_y[:5]}", flush=True)

# Optimizer & scheduler
criterion = nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.7)


# Validation function
def evaluate(loader):
    model.eval()
    total_loss    = 0.0
    total_samples = 0

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            preds = model(x)
            loss  = criterion(preds, y)

            total_loss    += loss.item() * x.size(0)
            total_samples += x.size(0)

    return total_loss / total_samples

num_epochs   = 1
train_losses = []
val_losses   = []

print("Starting training", flush=True)

for epoch in range(num_epochs):
    model.train()
    running_loss    = 0.0
    running_samples = 0

    # Reset peak memory stats before training starts
    use_cuda = device.type == "cuda"
    if use_cuda:
        torch.cuda.reset_peak_memory_stats(device)

    for batch_idx, (x, y) in enumerate(train_loader):
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        preds = model(x)
        loss  = criterion(preds, y)
        loss.backward()
        optimizer.step()

        running_loss    += loss.item() * x.size(0)
        running_samples += x.size(0)  

    # Calculate train loss for epoch
    epoch_train_loss = running_loss / running_samples

    # Validation loss
    epoch_val_loss = evaluate(val_loader)

    train_losses.append(epoch_train_loss)
    val_losses.append(epoch_val_loss)

    print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {epoch_train_loss:.4f} | "
          f"Val Loss: {epoch_val_loss:.4f}",
          flush=True)

    scheduler.step()  

# Capture peak memory after training completes
peak_train_memory_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)

print_profiling_report(model, test_loader, device, peak_train_memory_mb)

# Final Test Evaluation 
print("\nRunning final test evaluation", flush=True)

# Recover call_count scaler stats (first column in Z_SCORE_COLS)
cc_mean = scaler.mean_[0]
cc_std  = scaler.scale_[0]

model.eval()
all_preds   = []
all_targets = []

with torch.no_grad():
    for x, y in test_loader:
        x = x.to(device)
        y = y.to(device)

        preds = model(x)

        # Inverse z-score transform on call_count 
        all_preds.extend((preds.cpu().numpy() * cc_std + cc_mean).tolist())
        all_targets.extend((y.cpu().numpy()   * cc_std + cc_mean).tolist())


all_targets = np.array(all_targets)
all_preds   = np.array(all_preds)

# Compute metrics 
test_mae  = mean_absolute_error(all_targets, all_preds)
test_rmse = np.sqrt(mean_squared_error(all_targets, all_preds))
test_mse  = test_rmse ** 2
test_r2   = r2_score(all_targets, all_preds)


print("Final Test Evaluation Results", flush=True)
print(f"MAE:  {test_mae:.4f}",   flush=True)
print(f"RMSE: {test_rmse:.4f}",  flush=True)
print(f"MSE:  {test_mse:.4f}",   flush=True)
print(f"R²:   {test_r2:.4f}",    flush=True)

