import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    mean_absolute_percentage_error, median_absolute_error
)
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, List, Dict, Optional
import warnings
import time
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

print("=" * 80)
print("TCN FOR EMS CALL COUNT FORECASTING - UPDATED FEATURE SET")
print("=" * 80)

# ============================================================================
# FEATURE SET (per design spec)
# ============================================================================
# NOT scaled (sin/cos already bounded [-1, 1], binary flags):
#   ZIPCODE_*  (one-hot, no BOROUGH_* or INCIDENT_DISPATCH_AREA_*)
#   hour_sin, hour_cos
#   day_of_week_sin, day_of_week_cos
#   month_sin, month_cos
#   week_sin, week_cos
#   is_covid
#   special_event_indicator
#
# Z-scored (StandardScaler, fit on TRAIN only):
#   call_count
#   lag_1, lag_24, lag_168
#   roll_mean_24, roll_std_24
#   Population
#   initial_severity_level
# ============================================================================


# ============================================================================
# 1. DATA LOADING AND PREPROCESSING
# ============================================================================

class EMSDataPreprocessor:
    """
    Handles all data preprocessing for panel (zipcode × time) data.
    Keeps data as 2D (rows × features) — sequences are generated lazily.
    """

    def __init__(self, filepath: str):
        self.filepath = filepath
        self.scaler_X = StandardScaler()   # Features (z-score)
        self.scaler_y = StandardScaler()   # Target (z-score)
        self.continuous_features = None
        self.feature_names = None
        self.target = None
        self.datetime_series = None
        self.zipcode_series = None

    # ------------------------------------------------------------------
    def load_data(self) -> pd.DataFrame:
        print("\n[STEP 1] Loading data...")
        df = pd.read_csv(self.filepath)
        print(f"✅ Loaded {len(df):,} rows, {len(df.columns)} columns")
        print(f"   Memory usage: {df.memory_usage(deep=True).sum() / 1e9:.2f} GB")
        return df

    # ------------------------------------------------------------------
    def sort_by_zipcode_datetime(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        CRITICAL for panel data: sort by ZIPCODE then datetime so that
        the zipcode-boundary index is contiguous and lazy loading is safe.
        """
        print("\n[STEP 2] Sorting by ZIPCODE then datetime...")
        df = df.sort_values(["ZIPCODE", "datetime"]).reset_index(drop=True)
        print("✅ Sorted by ZIPCODE → datetime")
        return df

    # ------------------------------------------------------------------
    def create_target_per_zipcode(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Target = call_count at t+1, computed PER ZIPCODE to prevent
        the last row of zipcode A being used as the first target of zipcode B.
        """
        print("\n[STEP 3] Creating t+1 target per ZIPCODE...")
        if 'ZIPCODE' in df.columns and 'call_count' in df.columns:
            df["target"] = df.groupby("ZIPCODE")["call_count"].shift(-1)
        elif 'call_count' in df.columns:
            df["target"] = df["call_count"].shift(-1)
        else:
            raise ValueError("call_count column not found!")

        # Sanity check: verify target is truly the next row's call_count
        sample_zip = df['ZIPCODE'].unique()[0]
        cols = ['ZIPCODE', 'datetime', 'call_count', 'target']
        print("   Sanity check (first 5 rows of sample zipcode):")
        print(df[df['ZIPCODE'] == sample_zip][cols].head(5).to_string(index=False))
        print("✅ Target = call_count(t+1) per ZIPCODE")
        return df

    # ------------------------------------------------------------------
    def exclude_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Drop columns that should NOT be model inputs.

        KEPT (model inputs):
          call_count, lag_1, lag_24, lag_168,
          roll_mean_24, roll_std_24,
          Population, initial_severity_level,
          hour_sin, hour_cos,
          day_of_week_sin, day_of_week_cos,
          month_sin, month_cos,
          week_sin, week_cos,
          is_covid, special_event_indicator,
          ZIPCODE_*   (one-hot)

        EXCLUDED:
          - Identifiers / metadata
          - Raw spatial (ZIPCODE, BOROUGH, INCIDENT_DISPATCH_AREA)
          - BOROUGH_* and INCIDENT_DISPATCH_AREA_* one-hots
          - Current-hour call-type breakdown (leakage)
          - INITIAL_SEVERITY_LEVEL_CODE (post-call info, leakage)
          - Raw cyclical replaced by sin/cos: dayofweek, month, weekofyear
          - is_weekend (dropped per design spec)
          - Unused lags: lag_0, lag_2, lag_3, lag_6, lag_12, lag_48
          - Severity lags: severity_lag1, severity_lag6, severity_lag24, severity_lag168
          - roll_mean_168 (dropped per design spec)
        """
        print("\n[STEP 4] Excluding features not in design spec...")

        # ---- named columns to exclude ----
        exclude_named = [
            # Identifiers
            'datetime', 'date', 'date_only',
            # Metadata
            'missingDates',
            # Target (handled separately)
            'target',
            # Leakage: current-hour call breakdown
            'Environmental and Poisoning Emergencies',
            'Mass Casualty or Public Incidents',
            'Medical Emergencies',
            'Other',
            'Trauma-Related Incidents',
            # Leakage: post-call severity code
            'INITIAL_SEVERITY_LEVEL_CODE',
            # Raw spatial (one-hot versions kept for ZIPCODE only)
            'ZIPCODE', 'INCIDENT_DISPATCH_AREA', 'BOROUGH',
            # Raw cyclical → replaced by sin/cos
            'day',          # Redundant with dayofweek
            'hour',         # Replaced by hour_sin/hour_cos
            'dayofweek',    # Replaced by day_of_week_sin/cos
            'month',        # Replaced by month_sin/cos
            'weekofyear',   # Replaced by week_sin/cos
            'is_weekend',   # Dropped per design spec
            # Excluded lags (lag_1 and lag_24 are KEPT)
            'lag_0', 'lag_2', 'lag_3', 'lag_6', 'lag_12', 'lag_48',
            # Severity lags (all excluded)
            'severity_lag1', 'severity_lag6', 'severity_lag24', 'severity_lag168',
            # Rolling (only roll_mean_24 and roll_std_24 kept)
            'roll_mean_168',
        ]

        # Extract and preserve target, datetime, zipcode BEFORE dropping
        if 'target' in df.columns:
            self.target = df['target'].copy()
        else:
            raise ValueError("target column not found!")

        if 'datetime' in df.columns:
            self.datetime_series = df['datetime'].copy()
        if 'ZIPCODE' in df.columns:
            self.zipcode_series = df['ZIPCODE'].copy()

        # Drop named exclusions
        cols_to_drop = [c for c in exclude_named if c in df.columns]
        df = df.drop(columns=cols_to_drop, errors='ignore')

        # Drop BOROUGH_* and INCIDENT_DISPATCH_AREA_* one-hot columns
        onehot_to_drop = [
            c for c in df.columns
            if c.startswith('BOROUGH_') or c.startswith('INCIDENT_DISPATCH_AREA_')
        ]
        df = df.drop(columns=onehot_to_drop, errors='ignore')

        self.feature_names = list(df.columns)

        print(f"✅ Named exclusions: {len(cols_to_drop)}")
        print(f"   BOROUGH_* / INCIDENT_DISPATCH_AREA_* one-hots dropped: {len(onehot_to_drop)}")
        print(f"   Remaining features: {len(df.columns)}")

        # Integrity checks
        assert 'call_count' in df.columns, "call_count missing!"
        assert 'lag_1' in df.columns,      "lag_1 missing!"
        assert 'lag_24' in df.columns,     "lag_24 missing!"
        assert 'lag_168' in df.columns,    "lag_168 missing!"
        assert 'lag_0' not in df.columns,  "lag_0 must be excluded!"
        assert not any(c.startswith('BOROUGH_') for c in df.columns), \
            "BOROUGH_* one-hots still present!"
        assert not any(c.startswith('INCIDENT_DISPATCH_AREA_') for c in df.columns), \
            "INCIDENT_DISPATCH_AREA_* one-hots still present!"
        print("✅ Feature integrity checks passed")

        return df

    # ------------------------------------------------------------------
    def drop_nan_rows(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Drop rows where any of the critical features or the target is NaN.
        Keeps datetime_series, zipcode_series, and target in sync.
        """
        print("\n[STEP 5] Dropping NaN rows...")

        # Columns that must be non-NaN for a valid sample
        critical = [
            'call_count',
            'lag_1', 'lag_24', 'lag_168',
            'roll_mean_24', 'roll_std_24',
        ]
        # Add optional columns if present
        for col in ['initial_severity_level', 'Population']:
            if col in df.columns:
                critical.append(col)

        existing = [c for c in critical if c in df.columns]
        n_before = len(df)

        mask = df[existing].notna().all(axis=1) & self.target.notna()
        df = df[mask].reset_index(drop=True)
        self.target = self.target[mask].reset_index(drop=True)
        if self.datetime_series is not None:
            self.datetime_series = self.datetime_series[mask].reset_index(drop=True)
        if self.zipcode_series is not None:
            self.zipcode_series = self.zipcode_series[mask].reset_index(drop=True)

        print(f"   Rows before: {n_before:,}  →  After: {len(df):,}  "
              f"(Dropped: {n_before - len(df):,})")
        return df

    # ------------------------------------------------------------------
    def identify_feature_types(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        """
        Classify features into continuous (z-scored) and non-continuous.
        Only continuous features are passed through StandardScaler.
        """
        print("\n[STEP 6] Identifying feature types...")

        # Z-scored features (per design spec)
        self.continuous_features = [
            'call_count',
            'lag_1', 'lag_24', 'lag_168',
            'roll_mean_24', 'roll_std_24',
            'Population',
            'initial_severity_level',
        ]
        self.continuous_features = [f for f in self.continuous_features if f in df.columns]

        # Non-scaled features (sin/cos already in [-1,1]; binary flags; one-hots)
        non_scaled = [c for c in df.columns if c not in self.continuous_features]

        print(f"   Z-scored ({len(self.continuous_features)}): {self.continuous_features}")
        print(f"   Non-scaled ({len(non_scaled)}): "
              f"{[c for c in non_scaled if not c.startswith('ZIPCODE_')][:10]} "
              f"+ ZIPCODE_* one-hots")
        return {'continuous': self.continuous_features, 'non_scaled': non_scaled}

    # ------------------------------------------------------------------
    def scale_features_2d(self, df: pd.DataFrame, fit: bool = True) -> np.ndarray:
        """
        Apply StandardScaler to continuous columns only.

        Args:
            df:  2D DataFrame (n_rows × n_features), columns = self.feature_names
            fit: True  → fit scaler then transform  (TRAINING DATA ONLY)
                 False → transform only              (VAL / TEST DATA)

        Returns:
            np.ndarray of shape (n_rows, n_features), scaled in-place for continuous cols.

        Leakage note: caller must pass fit=True only for training rows.
        """
        if not self.continuous_features:
            return df.values

        data_2d = df.values.astype(np.float32)
        cont_idx = [i for i, c in enumerate(self.feature_names)
                    if c in self.continuous_features]

        if fit:
            print("   Fitting feature scaler on TRAINING data only...")
            self.scaler_X.fit(data_2d[:, cont_idx])

        data_2d[:, cont_idx] = self.scaler_X.transform(data_2d[:, cont_idx])
        print(f"   ✅ Features scaled ({'fit+transform' if fit else 'transform only'})")
        return data_2d

    # ------------------------------------------------------------------
    def scale_target_1d(self, y: np.ndarray, fit: bool = True) -> np.ndarray:
        """
        Apply StandardScaler to target values.

        Args:
            y:   1D array of target values
            fit: True  → fit scaler (TRAINING TARGET ONLY)
                 False → transform only (VAL / TEST)

        Returns:
            1D scaled array.

        Leakage note: caller must pass fit=True only for training target.
        """
        if fit:
            print("   Fitting target scaler on TRAINING target only...")
            self.scaler_y.fit(y.reshape(-1, 1))
        return self.scaler_y.transform(y.reshape(-1, 1)).flatten()

    # ------------------------------------------------------------------
    def build_zipcode_index(self) -> Dict[int, Tuple[int, int]]:
        """
        Build {zipcode: (start_row, end_row)} so lazy loading never
        creates a sequence window that spans two different zipcodes.
        """
        print("\n[STEP 7] Building zipcode boundary index...")
        zipcode_index = {}
        current_zip = None
        start_idx = 0

        for i, z in enumerate(self.zipcode_series):
            if z != current_zip:
                if current_zip is not None:
                    zipcode_index[current_zip] = (start_idx, i)
                current_zip = z
                start_idx = i
        if current_zip is not None:
            zipcode_index[current_zip] = (start_idx, len(self.zipcode_series))

        print(f"   ✅ Index built for {len(zipcode_index)} zipcodes")
        for z in list(zipcode_index.keys())[:3]:
            s, e = zipcode_index[z]
            print(f"     {z}: rows {s:,}–{e:,}  ({e - s:,} hours)")
        return zipcode_index


# ============================================================================
# 2. LAZY LOADING DATASET
# ============================================================================

class LazyTimeSeriesDataset(Dataset):
    """
    Stores the full 2D feature matrix in RAM.
    Sequence windows (shape: sequence_length × n_features) are sliced
    on-the-fly in __getitem__, so no 3D array is ever materialised.

    Boundary safety: valid_indices only contains window starts whose
    entire window falls within one zipcode's row range.
    """

    def __init__(self,
                 data_2d: np.ndarray,
                 target_1d: np.ndarray,
                 zipcode_series: np.ndarray,
                 datetime_series: np.ndarray,
                 zipcode_index: Dict[int, Tuple[int, int]],
                 sequence_length: int = 48,
                 indices: Optional[np.ndarray] = None):

        self.data_2d = data_2d
        self.target_1d = target_1d
        self.zipcode_series = zipcode_series
        self.datetime_series = datetime_series
        self.zipcode_index = zipcode_index
        self.sequence_length = sequence_length

        self.valid_indices = (indices if indices is not None
                              else self._build_valid_indices())

        print(f"   ✅ LazyDataset: {len(self.valid_indices):,} windows | "
              f"2D shape {data_2d.shape} | "
              f"{data_2d.nbytes / 1e9:.2f} GB in RAM")

    # ------------------------------------------------------------------
    def _build_valid_indices(self) -> np.ndarray:
        """
        A window starting at row i is valid iff rows [i, i+seq_len)
        all belong to the SAME zipcode (no cross-boundary leakage).
        """
        valid = []
        for zipcode, (start, end) in self.zipcode_index.items():
            for ws in range(start, end - self.sequence_length):
                if ws + self.sequence_length <= end:
                    valid.append(ws)
        return np.array(valid, dtype=np.int64)

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        ws = self.valid_indices[idx]
        we = ws + self.sequence_length

        X = self.data_2d[ws:we]           # (seq_len, n_features)
        y = self.target_1d[we - 1]        # scalar: call_count at t+1

        X_t = torch.FloatTensor(X).T      # (n_features, seq_len) for Conv1d
        y_t = torch.FloatTensor([y]).squeeze()
        return X_t, y_t

    # ------------------------------------------------------------------
    def get_subset_by_datetime(self, date_mask: np.ndarray) -> 'LazyTimeSeriesDataset':
        """
        Subset the dataset to windows whose TARGET timestamp falls within
        the boolean mask. Used for train/val/test splits.
        """
        subset = np.array([
            i for i in self.valid_indices
            if date_mask[i + self.sequence_length - 1]
        ])
        return LazyTimeSeriesDataset(
            data_2d=self.data_2d,
            target_1d=self.target_1d,
            zipcode_series=self.zipcode_series,
            datetime_series=self.datetime_series,
            zipcode_index=self.zipcode_index,
            sequence_length=self.sequence_length,
            indices=subset
        )


# ============================================================================
# 3. DATETIME-BASED SPLITTING
# ============================================================================

class DatetimeBasedSplitter:
    """
    Temporal train/val/test split via boolean masks.
    No data arrays are created — only masks over the existing 2D matrix.
    """

    @staticmethod
    def split(datetime_series: np.ndarray,
              train_ratio: float = 0.70,
              val_ratio: float = 0.15) -> Dict[str, np.ndarray]:

        print(f"\n[STEP 8] Temporal split  "
              f"train={train_ratio*100:.0f}% / "
              f"val={val_ratio*100:.0f}% / "
              f"test={(1-train_ratio-val_ratio)*100:.0f}%")

        unique_dt = np.unique(datetime_series)
        n = len(unique_dt)
        train_cut = unique_dt[int(train_ratio * n)]
        val_cut   = unique_dt[int((train_ratio + val_ratio) * n)]

        train_mask = datetime_series < train_cut
        val_mask   = (datetime_series >= train_cut) & (datetime_series < val_cut)
        test_mask  = datetime_series >= val_cut

        for name, mask, series in [
            ('Train', train_mask, datetime_series[train_mask]),
            ('Val',   val_mask,   datetime_series[val_mask]),
            ('Test',  test_mask,  datetime_series[test_mask]),
        ]:
            print(f"   {name:5s}: {series.min()} → {series.max()}  "
                  f"({mask.sum():,} rows)")

        # Verify strict temporal ordering (no overlap)
        assert not (train_mask & val_mask).any(),  "⚠ LEAKAGE: Train/Val overlap!"
        assert not (val_mask & test_mask).any(),   "⚠ LEAKAGE: Val/Test overlap!"
        assert not (train_mask & test_mask).any(), "⚠ LEAKAGE: Train/Test overlap!"
        print("✅ No temporal leakage in splits")

        return {'train_mask': train_mask,
                'val_mask':   val_mask,
                'test_mask':  test_mask}


# ============================================================================
# 4. TCN ARCHITECTURE
# ============================================================================

class Chomp1d(nn.Module):
    """Remove trailing padding to preserve strict causality."""
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x if self.chomp_size == 0 else x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    """
    One dilated causal residual block.
    dilation = 2^i ensures exponentially growing receptive field.
    """
    def __init__(self, n_inputs, n_outputs, kernel_size,
                 stride, dilation, padding, dropout=0.2):
        super().__init__()
        self.conv1 = nn.Conv1d(n_inputs,  n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.chomp1    = Chomp1d(padding)
        self.relu1     = nn.ReLU()
        self.dropout1  = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.chomp2    = Chomp1d(padding)
        self.relu2     = nn.ReLU()
        self.dropout2  = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = (nn.Conv1d(n_inputs, n_outputs, 1)
                           if n_inputs != n_outputs else None)
        self.relu = nn.ReLU()
        self._init_weights()

    def _init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    """Stack of TemporalBlocks with exponential dilation."""
    def __init__(self, num_inputs, num_channels, kernel_size=3, dropout=0.2):
        super().__init__()
        layers = []
        for i, out_ch in enumerate(num_channels):
            dilation  = 2 ** i
            in_ch     = num_inputs if i == 0 else num_channels[i - 1]
            padding   = (kernel_size - 1) * dilation
            layers.append(TemporalBlock(in_ch, out_ch, kernel_size,
                                        stride=1, dilation=dilation,
                                        padding=padding, dropout=dropout))
        self.network = nn.Sequential(*layers)
        # receptive_field = 1 + 2*(k-1)*(2^L - 1)
        L = len(num_channels)
        self.receptive_field = 1 + 2 * (kernel_size - 1) * (2 ** L - 1)

    def forward(self, x):
        return self.network(x)


class TCNForecaster(nn.Module):
    """TCN → Linear(1) for scalar regression."""
    def __init__(self, input_size, num_channels, kernel_size=3, dropout=0.3):
        super().__init__()
        self.tcn    = TemporalConvNet(input_size, num_channels, kernel_size, dropout)
        self.linear = nn.Linear(num_channels[-1], 1)

        print(f"\n   TCN Architecture:")
        print(f"   Input size      : {input_size}")
        print(f"   Channels        : {num_channels}")
        print(f"   Kernel size     : {kernel_size}")
        print(f"   Dropout         : {dropout}")
        print(f"   Receptive field : {self.tcn.receptive_field} timesteps")
        # --- Parameter Count ---
        # Identical to STGCN: sum(p.numel() for p in model.parameters())
        total_params = sum(p.numel() for p in self.parameters())
        print(f"   Parameters      : {total_params:,}")

    def forward(self, x):
        # x: (batch, features, seq_len)
        out = self.tcn(x)              # (batch, channels, seq_len)
        out = self.linear(out[:, :, -1])  # (batch, 1) — use last timestep
        return out.squeeze(-1)         # (batch,)


# ============================================================================
# 5. TRAINING
# ============================================================================

class TCNTrainer:
    def __init__(self, model, device=None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.train_losses = []
        self.val_losses   = []
        self.best_val_loss    = float('inf')
        self.best_model_state = None

    def _run_epoch(self, loader, optimizer, criterion, training: bool):
        self.model.train(training)
        total_loss = 0.0
        ctx = torch.enable_grad() if training else torch.no_grad()
        with ctx:
            for X, y in loader:
                X, y = X.to(self.device), y.to(self.device)
                pred = self.model(X)
                loss = criterion(pred, y)
                if training:
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    optimizer.step()
                total_loss += loss.item()
        return total_loss / len(loader)

    def fit(self, train_loader, val_loader,
            epochs=50, lr=1e-3, patience=15, weight_decay=1e-4):

        print(f"\n[STEP 13] Training  |  device={self.device}  "
              f"epochs={epochs}  lr={lr}  patience={patience}")
        print("-" * 80)

        criterion  = nn.MSELoss()
        optimizer  = optim.Adam(self.model.parameters(),
                                lr=lr, weight_decay=weight_decay)
        scheduler  = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=False)

        patience_ctr = 0
        for epoch in range(epochs):
            tr_loss  = self._run_epoch(train_loader, optimizer, criterion, training=True)
            val_loss = self._run_epoch(val_loader,   None,      criterion, training=False)

            self.train_losses.append(tr_loss)
            self.val_losses.append(val_loss)
            scheduler.step(val_loss)

            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"   Epoch {epoch+1:3d}/{epochs} | "
                      f"Train {tr_loss:.6f} | Val {val_loss:.6f}")

            if val_loss < self.best_val_loss:
                self.best_val_loss    = val_loss
                self.best_model_state = {k: v.clone()
                                         for k, v in self.model.state_dict().items()}
                patience_ctr = 0
            else:
                patience_ctr += 1
                if patience_ctr >= patience:
                    print(f"\n   Early stopping at epoch {epoch + 1}")
                    break

        self.model.load_state_dict(self.best_model_state)
        print(f"\n✅ Best val loss: {self.best_val_loss:.6f}")

    def predict(self, loader):
        self.model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for X, y in loader:
                preds.append(self.model(X.to(self.device)).cpu().numpy())
                trues.append(y.numpy())
        return np.concatenate(preds), np.concatenate(trues)


# ============================================================================
# 6. EVALUATION
# ============================================================================

class ResultAnalyzer:

    @staticmethod
    def metrics(y_true, y_pred) -> dict:
        mae   = mean_absolute_error(y_true, y_pred)
        rmse  = np.sqrt(mean_squared_error(y_true, y_pred))
        mse   = mean_squared_error(y_true, y_pred)
        r2    = r2_score(y_true, y_pred)
        medae = median_absolute_error(y_true, y_pred)
        try:
            mape = mean_absolute_percentage_error(y_true, y_pred)
        except Exception:
            mape = np.inf
        return dict(MAE=mae, RMSE=rmse, MSE=mse, R2=r2, MAPE=mape, MedAE=medae)

    @staticmethod
    def print_metrics(m: dict, title=''):
        print(f"\n{'='*60}\n{title:^60}\n{'='*60}")
        for k, v in m.items():
            val_str = f"{v:>15.4f}" if np.isfinite(v) else f"{'INF/NAN':>15}"
            print(f"  {k:<8}: {val_str}")
        print('=' * 60)

    @staticmethod
    def plot_training_history(train_losses, val_losses, fname='01_training_history_TCN_01_18Features.png'):
        plt.figure(figsize=(12, 5))
        plt.plot(train_losses, label='Train', lw=2)
        plt.plot(val_losses,   label='Val',   lw=2)
        plt.xlabel('Epoch'); plt.ylabel('MSE Loss')
        plt.title('TCN Training History')
        plt.legend(); plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(fname, dpi=300, bbox_inches='tight')
        print(f"✅ Saved {fname}")
        plt.close()

    @staticmethod
    def plot_predictions(y_true, y_pred, title='Test', n=500,
                         fname='02_predictions_TCN_01_18Features.png', scale='Scaled'):
        fig, axes = plt.subplots(2, 1, figsize=(15, 10))
        n_plot = min(n, len(y_true))
        ix = range(n_plot)

        axes[0].plot(ix, y_true[:n_plot], label='Actual',    lw=1.5, alpha=0.8)
        axes[0].plot(ix, y_pred[:n_plot], label='Predicted', lw=1.5, alpha=0.8)
        axes[0].fill_between(ix, y_true[:n_plot], y_pred[:n_plot],
                             alpha=0.15, color='red', label='Error')
        axes[0].set(xlabel='Time Step',
                    ylabel=f'Call Count ({scale})',
                    title=f'{title}: Actual vs Predicted (first {n_plot} samples)')
        axes[0].legend(); axes[0].grid(alpha=0.3)

        axes[1].scatter(y_true, y_pred, alpha=0.3, s=10)
        lo = min(y_true.min(), y_pred.min())
        hi = max(y_true.max(), y_pred.max())
        axes[1].plot([lo, hi], [lo, hi], 'r--', lw=2, label='Perfect')
        axes[1].set(xlabel=f'Actual ({scale})',
                    ylabel=f'Predicted ({scale})',
                    title='Scatter: Actual vs Predicted')
        axes[1].legend(); axes[1].grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(fname, dpi=300, bbox_inches='tight')
        print(f"✅ Saved {fname}")
        plt.close()

    @staticmethod
    def plot_residuals(y_true, y_pred, title='Test', fname='03_residuals_TCN_01_18Features.png'):
        residuals = y_true - y_pred
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        axes[0].hist(residuals, bins=50, edgecolor='black', alpha=0.7)
        axes[0].axvline(0,              color='red',   ls='--', lw=2, label='Zero')
        axes[0].axvline(residuals.mean(), color='green', ls='--', lw=2,
                        label=f'Mean={residuals.mean():.3f}')
        axes[0].set(xlabel='Residual', ylabel='Count', title='Residual Distribution')
        axes[0].legend(); axes[0].grid(alpha=0.3)

        axes[1].scatter(y_pred, residuals, alpha=0.3, s=10)
        axes[1].axhline(0, color='red', ls='--', lw=2)
        axes[1].set(xlabel='Predicted', ylabel='Residual', title='Residual vs Predicted')
        axes[1].grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(fname, dpi=300, bbox_inches='tight')
        print(f"✅ Saved {fname}")
        plt.close()

    @staticmethod
    def diagnostics(y_true, y_pred, metrics, scale='Scaled'):
        print("\n" + "=" * 80)
        print(f"FINAL DIAGNOSTICS  [{scale}]")
        print("=" * 80)
        errors = np.abs(y_true - y_pred)
        print(f"\n  Mean absolute: {errors.mean():.4f}")
        print(f"  Median abs:    {np.median(errors):.4f}")
        print(f"  Max abs:       {errors.max():.4f}")
        print(f"  Std:           {errors.std():.4f}")
        print("\n  Error percentiles:")
        for p in [50, 75, 90, 95, 99]:
            print(f"    {p}th: {np.percentile(errors, p):.4f}")
        print("=" * 80)


# ============================================================================
# 7. MAIN PIPELINE
# ============================================================================

def main():
    print("\n" + "=" * 80)
    print("INITIALIZATION")
    print("=" * 80)

    filepath = 'All_Features_With_Encoding_updated.csv'
    preprocessor = EMSDataPreprocessor(filepath)

    # ------------------------------------------------------------------ #
    # PHASE 1: Load & preprocess (keep 2D)                                #
    # ------------------------------------------------------------------ #
    df = preprocessor.load_data()
    df = preprocessor.sort_by_zipcode_datetime(df)
    df = preprocessor.create_target_per_zipcode(df)
    df = preprocessor.exclude_features(df)
    df = preprocessor.drop_nan_rows(df)
    preprocessor.identify_feature_types(df)

    data_raw_2d    = df.values.astype(np.float32)
    target_raw_1d  = preprocessor.target.values.astype(np.float32)
    print(f"\n   Raw 2D shape : {data_raw_2d.shape}")
    print(f"   RAM usage    : {data_raw_2d.nbytes / 1e9:.2f} GB")

    # ------------------------------------------------------------------ #
    # PHASE 2: Zipcode boundary index                                     #
    # ------------------------------------------------------------------ #
    zipcode_index = preprocessor.build_zipcode_index()

    # ------------------------------------------------------------------ #
    # PHASE 3: Temporal split masks (NO arrays created)                   #
    # ------------------------------------------------------------------ #
    masks = DatetimeBasedSplitter.split(
        preprocessor.datetime_series.values,
        train_ratio=0.60, val_ratio=0.20
    )
    train_mask = masks['train_mask']
    val_mask   = masks['val_mask']
    test_mask  = masks['test_mask']

    # ------------------------------------------------------------------ #
    # PHASE 4: Scaling                                                    #
    #   CRITICAL: fit scalers on TRAINING data only                       #
    # ------------------------------------------------------------------ #
    print("\n[STEP 9] Scaling  (fit on train only, transform all)")

    # --- Feature scaler ---
    # Step A: fit on training rows only
    preprocessor.scale_features_2d(
        pd.DataFrame(data_raw_2d[train_mask], columns=preprocessor.feature_names),
        fit=True
    )
    # Step B: transform ALL rows with the fitted scaler
    data_2d = preprocessor.scale_features_2d(
        pd.DataFrame(data_raw_2d, columns=preprocessor.feature_names),
        fit=False
    )

    # --- Target scaler ---
    # Step A: fit on training target only
    preprocessor.scale_target_1d(target_raw_1d[train_mask], fit=True)
    # Step B: transform ALL targets
    target_1d = preprocessor.scale_target_1d(target_raw_1d, fit=False)

    print(f"✅ Scaling complete  |  data_2d: {data_2d.nbytes / 1e9:.2f} GB")

    # ------------------------------------------------------------------ #
    # PHASE 5: Lazy datasets                                               #
    # ------------------------------------------------------------------ #
    print("\n[STEP 10] Creating lazy datasets  (seq_len=48)...")
    SEQ_LEN = 48

    full_ds   = LazyTimeSeriesDataset(
        data_2d=data_2d, target_1d=target_1d,
        zipcode_series=preprocessor.zipcode_series.values,
        datetime_series=preprocessor.datetime_series.values,
        zipcode_index=zipcode_index,
        sequence_length=SEQ_LEN
    )
    train_ds = full_ds.get_subset_by_datetime(train_mask)
    val_ds   = full_ds.get_subset_by_datetime(val_mask)
    test_ds  = full_ds.get_subset_by_datetime(test_mask)

    print(f"   Train: {len(train_ds):,} windows  |  "
          f"Val: {len(val_ds):,}  |  Test: {len(test_ds):,}")

    # ------------------------------------------------------------------ #
    # PHASE 6: DataLoaders                                                 #
    # ------------------------------------------------------------------ #
    BATCH = 64
    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=False, num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH, shuffle=False, num_workers=0)
    # batch_size=50 matches STGCN and ST-MGCN exactly, ensuring
    # inference time, ms/sample (/50), and memory are directly comparable
    test_loader  = DataLoader(test_ds,  batch_size=50,   shuffle=False, num_workers=0)

    # ------------------------------------------------------------------ #
    # PHASE 7: Model                                                       #
    # ------------------------------------------------------------------ #
    print("\n[STEP 11] Building TCN model...")
    input_size   = data_2d.shape[1]
    num_channels = [32, 32, 32, 32]   # 4 layers → receptive field = 61
    kernel_size  = 3
    dropout      = 0.3

    model = TCNForecaster(input_size, num_channels, kernel_size, dropout)

    # Verify receptive field
    assert model.tcn.receptive_field == 61, \
        f"Expected RF=61, got {model.tcn.receptive_field}"
    print(f"✅ Receptive field verified: {model.tcn.receptive_field} "
          f"(covers {model.tcn.receptive_field/SEQ_LEN*100:.0f}% of seq_len={SEQ_LEN})")

    # ------------------------------------------------------------------ #
    # PHASE 8: Train                                                       #
    # ------------------------------------------------------------------ #
    trainer = TCNTrainer(model)
    trainer.fit(train_loader, val_loader,
                epochs=50, lr=1e-3, patience=15, weight_decay=1e-4)

    # ------------------------------------------------------------------ #
    # PHASE 9: Inference time, peak memory, parameter count               #
    #          Identical to STGCN measurement block                       #
    # ------------------------------------------------------------------ #
    print("\n[STEP 12] Measuring inference time and memory...")
    device = torch.device(trainer.device)

    # --- Parameter Count ---
    # Identical to STGCN: sum(p.numel() for p in model.parameters())
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total_params:,}")

    # --- Inference Time (ms per batch) ---
    # 10-run warmup — no cuda.synchronize(), matches STGCN exactly
    dummy_batch = next(iter(test_loader))[0].to(device)
    model.eval()
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy_batch)

    # Timed loop — no cuda.synchronize(), no pred collection, matches STGCN
    start = time.perf_counter()
    with torch.no_grad():
        for batch_x, _ in test_loader:
            _ = model(batch_x.to(device))
    end = time.perf_counter()

    n_batches = len(test_loader)
    inference_time_ms = ((end - start) / n_batches) * 1000
    print(f"Inference Time (ms per batch): {inference_time_ms:.2f}")
    # hardcoded /50 — matches STGCN exactly
    print(f"Inference Time (ms per sample): {inference_time_ms / 50:.4f}")

    # --- Peak Memory (MB) ---
    # GPU: max_memory_allocated() with no reset — matches STGCN
    # CPU: tracemalloc on a separate inference pass — matches STGCN
    if device.type == 'cuda':
        peak_mem_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        print(f"Peak GPU Memory (MB): {peak_mem_mb:.2f}")
    else:
        import tracemalloc
        tracemalloc.start()
        with torch.no_grad():
            for batch_x, _ in test_loader:
                _ = model(batch_x.to(device))
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        peak_mem_mb = peak / (1024 ** 2)
        print(f"Peak CPU Memory (MB): {peak_mem_mb:.2f}")

    # ------------------------------------------------------------------ #
    # PHASE 10: Evaluate  (both scaled AND original scale)                 #
    # Separate from timed block — matches STGCN's separate evaluate() call #
    # ------------------------------------------------------------------ #
    print("\n[STEP 12] Evaluating...")
    analyzer = ResultAnalyzer()

    def evaluate_split(loader, split_name):
        pred_scaled, true_scaled = trainer.predict(loader)

        # Metrics on scaled values
        m_scaled = analyzer.metrics(true_scaled, pred_scaled)
        analyzer.print_metrics(m_scaled, f"{split_name}  [Z-scored]")

        # Inverse-transform → original call-count scale
        pred_orig = preprocessor.scaler_y.inverse_transform(
            pred_scaled.reshape(-1, 1)).flatten()
        true_orig = preprocessor.scaler_y.inverse_transform(
            true_scaled.reshape(-1, 1)).flatten()
        m_orig = analyzer.metrics(true_orig, pred_orig)
        analyzer.print_metrics(m_orig, f"{split_name}  [Original scale]")

        return pred_scaled, true_scaled, pred_orig, true_orig, m_scaled, m_orig

    (tr_ps, tr_ts, tr_po, tr_to,
     tr_ms, tr_mo) = evaluate_split(train_loader, "TRAIN")

    (va_ps, va_ts, va_po, va_to,
     va_ms, va_mo) = evaluate_split(val_loader, "VALIDATION")

    (te_ps, te_ts, te_po, te_to,
     te_ms, te_mo) = evaluate_split(test_loader, "TEST")

    # ------------------------------------------------------------------ #
    # PHASE 11: Plots (original scale for interpretability)                #
    # ------------------------------------------------------------------ #
    print("\n[STEP 13] Plotting...")
    analyzer.plot_training_history(trainer.train_losses, trainer.val_losses,
                                   '01_training_history_TCN_01_18Features.png')
    analyzer.plot_predictions(te_to, te_po, title='Test', n=500,
                              fname='02_test_predictions_original_TCN_01_18Features.png',
                              scale='Original (call count)')
    analyzer.plot_predictions(te_ts, te_ps, title='Test', n=500,
                              fname='02_test_predictions_scaled_TCN_01_18Features.png',
                              scale='Z-scored')
    analyzer.plot_residuals(te_to, te_po,
                            fname='03_test_residuals_original_TCN_01_18Features.png')
    analyzer.diagnostics(te_to, te_po, te_mo, scale='Original')
    analyzer.diagnostics(te_ts, te_ps, te_ms, scale='Z-scored')

    # ------------------------------------------------------------------ #
    # PHASE 12: Save                                                        #
    # ------------------------------------------------------------------ #
    print("\n[STEP 14] Saving model...")
    torch.save({
        'model_state_dict': model.state_dict(),
        'scaler_X': preprocessor.scaler_X,
        'scaler_y': preprocessor.scaler_y,
        'feature_names': preprocessor.feature_names,
        'model_config': {
            'input_size': input_size,
            'num_channels': num_channels,
            'kernel_size': kernel_size,
            'dropout': dropout,
            'sequence_length': SEQ_LEN,
            'receptive_field': model.tcn.receptive_field,
        },
        'metrics': {
            'train_scaled': tr_ms, 'train_original': tr_mo,
            'val_scaled':   va_ms, 'val_original':   va_mo,
            'test_scaled':  te_ms, 'test_original':  te_mo,
        }
    }, 'tcn_ems_updated.pth')
    print("✅ Saved: tcn_ems_updated.pth")

    # ------------------------------------------------------------------ #
    # SUMMARY                                                              #
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 80)
    print("PIPELINE COMPLETE")
    print("=" * 80)
    print(f"\n  Features used  : {len(preprocessor.feature_names)}")
    print(f"  Seq length     : {SEQ_LEN} hrs")
    print(f"  Receptive field: {model.tcn.receptive_field} timesteps")
    print(f"\n  TEST  [Original scale]")
    print(f"    MAE  : {te_mo['MAE']:.4f}  calls")
    print(f"    RMSE : {te_mo['RMSE']:.4f}  calls")
    print(f"    R²   : {te_mo['R2']:.4f}")
    print(f"\n  TEST  [Z-scored]")
    print(f"    MAE  : {te_ms['MAE']:.6f}")
    print(f"    RMSE : {te_ms['RMSE']:.6f}")
    print(f"    R²   : {te_ms['R2']:.6f}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
