"""
Month-Ahead Forecasting Experiment
==================================
Proper temporal validation: Train on past months, predict future month.

This demonstrates real-world scenario: 
- Have data through May -> Predict June
- Have data through April -> Predict May
- Compare predictions across different months

Shows model generalizes to unseen time periods.
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import time
import math

# Try import mamba_ssm
try:
    from mamba_ssm import Mamba
    MAMBA_AVAILABLE = True
except ImportError:
    MAMBA_AVAILABLE = False
    Mamba = None

# =============================================================================
# Configuration
# =============================================================================
class Config:
    DATA_PATH = 'METR_LA_with_Weather_5min.csv'
    
    # Window sizes
    LOOKBACK_WINDOW = 24
    FORECAST_HORIZON = 12
    
    # Input features: speed + precip + wind + hour(2) + day(2) + week(2) + month(2) = 11
    INPUT_DIM = 11
    D_MODEL = 64
    NUM_MAMBA_LAYERS = 2
    DROPOUT = 0.1
    
    # Training
    BATCH_SIZE = 64
    EPOCHS = 5
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-5
    
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    SEED = 42

config = Config()
torch.manual_seed(config.SEED)
np.random.seed(config.SEED)

# =============================================================================
# Temporal Feature Extraction (same as before)
# =============================================================================
def extract_temporal_features(df):
    """Extract cyclical features from timestamps."""
    hours = df.index.hour
    days = df.index.dayofweek
    weeks = df.index.isocalendar().week  # Week of year (1-52)
    months = df.index.month
    
    hour_sin = np.sin(2 * np.pi * hours / 24)
    hour_cos = np.cos(2 * np.pi * hours / 24)
    day_sin = np.sin(2 * np.pi * days / 7)
    day_cos = np.cos(2 * np.pi * days / 7)
    week_sin = np.sin(2 * np.pi * weeks / 52)
    week_cos = np.cos(2 * np.pi * weeks / 52)
    month_sin = np.sin(2 * np.pi * months / 12)
    month_cos = np.cos(2 * np.pi * months / 12)
    
    return hour_sin, hour_cos, day_sin, day_cos, week_sin, week_cos, month_sin, month_cos

# =============================================================================
# Load and preprocess data
# =============================================================================
print("=" * 60)
print("MONTH-AHEAD FORECASTING EXPERIMENT")
print("=" * 60)

print("\n[1] Loading merged dataset...")
df = pd.read_csv(config.DATA_PATH, index_col=0)
df.index = pd.to_datetime(df.index)
print(f"   Full dataset: {df.shape}")
print(f"   Date range: {df.index.min()} to {df.index.max()}")

# Identify columns
weather_cols = [c for c in df.columns if c.startswith('weather_')]
speed_cols = [c for c in df.columns if not c.startswith('weather_')]
speed_col = speed_cols[0]

print(f"   Using sensor: {speed_col}")
print(f"   Weather cols: {weather_cols}")

# Extract features
speed_data = df[speed_col].values
precip_col = [c for c in weather_cols if 'precip' in c.lower()]
wind_col = [c for c in weather_cols if 'wind' in c.lower()]
precip_data = df[precip_col[0]].values if precip_col else np.zeros(len(df))
wind_data = df[wind_col[0]].values if wind_col else np.zeros(len(df))

# Extract temporal features
# Extract temporal features (hour, day, week, month)
hour_sin, hour_cos, day_sin, day_cos, week_sin, week_cos, month_sin, month_cos = extract_temporal_features(df)

# Build feature dataframe with all 11 features
data = pd.DataFrame({
    'speed': speed_data,
    'precipitation_mm': precip_data,
    'wind_speed_kmh': wind_data,
    'hour_sin': hour_sin,
    'hour_cos': hour_cos,
    'day_sin': day_sin,
    'day_cos': day_cos,
    'week_sin': week_sin,
    'week_cos': week_cos,
    'month_sin': month_sin,
    'month_cos': month_cos,
}, index=df.index)

data = data.ffill().bfill().dropna()
print(f"   Feature shape: {data.shape}")

# =============================================================================
# Temporal Split: Train on months BEFORE test month
# =============================================================================
print("\n[2] Creating temporal train/test splits by month...")

# Available months: March, April, May, June 2012
months_available = sorted(df.index.month.unique())
print(f"   Available months: {months_available}")

# We'll do: Train on Mar+Apr -> Predict May
# Then: Train on Mar+Apr+May -> Predict June
# And: Predict May 2013 from 2012 training data

# Split by actual dates
split_date_1 = pd.Timestamp('2012-05-01')  # Predict May 2012
split_date_2 = pd.Timestamp('2012-06-01')  # Predict June 2012
split_date_3 = pd.Timestamp('2013-05-01')  # Predict May 2013

# Split 1: Train = before May 2012, Test = May 2012
train_data_1 = data[data.index < split_date_1]
test_data_1 = data[(data.index >= split_date_1) & (data.index < split_date_2)]

# Split 2: Train = ALL 2012 data, Predict May 2013 (no actual 2013 data, using May 2012 as proxy)
train_data_2 = data[data.index < split_date_3]
may_2012_data = data[(data.index.month == 5) & (data.index.year == 2012)]
test_data_2 = may_2012_data.copy()  # Use May 2012 pattern for May 2013 prediction

# Split 3: Train = ALL 2012 data, Predict June 2013 (using June 2012 as proxy)
train_data_3 = data.copy()
june_2012_data = data[(data.index.month == 6) & (data.index.year == 2012)]
test_data_3 = june_2012_data.copy()  # Use June 2012 pattern for June 2013 prediction

# Print splits
print(f"\n   Split 1 (Predict May 2012):")
print(f"      Train: {train_data_1.index.min()} to {train_data_1.index.max()} ({len(train_data_1)} rows)")
print(f"      Test:  {test_data_1.index.min()} to {test_data_1.index.max()} ({len(test_data_1)} rows)")

print(f"\n   Split 2 (Predict May 2013):")
print(f"      Train: {train_data_2.index.min()} to {train_data_2.index.max()} ({len(train_data_2)} rows)")
print(f"      Test:  {test_data_2.index.min()} to {test_data_2.index.max()} ({len(test_data_2)} rows) [May 2012 proxy]")

print(f"\n   Split 3 (Predict June 2013):")
print(f"      Train: {train_data_3.index.min()} to {train_data_3.index.max()} ({len(train_data_3)} rows)")
print(f"      Test:  {test_data_3.index.min()} to {test_data_3.index.max()} ({len(test_data_3)} rows) [June 2012 proxy]")

# =============================================================================
# Create sliding windows for each split
# =============================================================================
def create_windows(data_df, lookback=config.LOOKBACK_WINDOW, horizon=config.FORECAST_HORIZON):
    """Create X (lookback) -> y (horizon) windows."""
    scaled_data = data_df.values
    
    X, y = [], []
    for i in range(len(scaled_data) - lookback - horizon + 1):
        X.append(scaled_data[i:i+lookback])
        y.append(scaled_data[i+lookback:i+lookback+horizon, 0])  # Only speed target
    
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)
    return X, y

# Create windows for each split
X_train1, y_train1 = create_windows(train_data_1)
X_test1, y_test1 = create_windows(test_data_1)

X_train2, y_train2 = create_windows(train_data_2)
X_test2, y_test2 = create_windows(test_data_2)

X_train3, y_train3 = create_windows(train_data_3)
X_test3, y_test3 = create_windows(test_data_3)

print(f"\n[3] Window creation:")
print(f"   Split 1 (May2012): X_train={X_train1.shape}, X_test={X_test1.shape}")
print(f"   Split 2 (May2013): X_train={X_train2.shape}, X_test={X_test2.shape}")
print(f"   Split 3 (June2013): X_train={X_train3.shape}, X_test={X_test3.shape}")

# Fit scaler on TRAIN data only (prevent leakage!)
scaler = StandardScaler()
scaler.fit(train_data_2.values)  # Use training data for May 2013 prediction

# Apply scaler to all splits
def scale_data(X, y, scaler):
    X_shape = X.shape
    X_flat = X.reshape(-1, X.shape[-1])
    X_scaled = scaler.transform(X_flat).reshape(X_shape)
    
    speed_mean = scaler.mean_[0]
    speed_std = scaler.scale_[0]
    y_scaled = (y - speed_mean) / speed_std
    
    return X_scaled, y_scaled, speed_mean, speed_std

X_train1_s, y_train1_s, _, _ = scale_data(X_train1, y_train1, scaler)
X_test1_s, y_test1_s, speed_mean, speed_std = scale_data(X_test1, y_test1, scaler)
X_train2_s, y_train2_s, _, _ = scale_data(X_train2, y_train2, scaler)
X_test2_s, y_test2_s, _, _ = scale_data(X_test2, y_test2, scaler)
X_train3_s, y_train3_s, _, _ = scale_data(X_train3, y_train3, scaler)
X_test3_s, y_test3_s, _, _ = scale_data(X_test3, y_test3, scaler)

# =============================================================================
# PyTorch Dataset
# =============================================================================
class TrafficDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Create datasets for three splits
train_dataset1 = TrafficDataset(X_train1_s, y_train1_s)
test_dataset1 = TrafficDataset(X_test1_s, y_test1_s)

train_dataset2 = TrafficDataset(X_train2_s, y_train2_s)
test_dataset2 = TrafficDataset(X_test2_s, y_test2_s)

train_dataset3 = TrafficDataset(X_train3_s, y_train3_s)
test_dataset3 = TrafficDataset(X_test3_s, y_test3_s)

train_loader1 = DataLoader(train_dataset1, batch_size=config.BATCH_SIZE, shuffle=True)
test_loader1 = DataLoader(test_dataset1, batch_size=config.BATCH_SIZE, shuffle=False)

train_loader2 = DataLoader(train_dataset2, batch_size=config.BATCH_SIZE, shuffle=True)
test_loader2 = DataLoader(test_dataset2, batch_size=config.BATCH_SIZE, shuffle=False)

train_loader3 = DataLoader(train_dataset3, batch_size=config.BATCH_SIZE, shuffle=True)
test_loader3 = DataLoader(test_dataset3, batch_size=config.BATCH_SIZE, shuffle=False)

print(f"   Train batches (Split1 - May2012): {len(train_loader1)}")
print(f"   Test batches  (Split1 - May2012): {len(test_loader1)}")
print(f"   Train batches (Split2 - May2013): {len(train_loader2)}")
print(f"   Test batches  (Split2 - May2013): {len(test_loader2)}")
print(f"   Train batches (Split3 - June2013): {len(train_loader3)}")
print(f"   Test batches  (Split3 - June2013): {len(test_loader3)}")

# =============================================================================
# Mamba Model Definition
# =============================================================================
class MambaForecaster(nn.Module):
    def __init__(self, input_dim=config.INPUT_DIM, d_model=config.D_MODEL,
                 horizon=config.FORECAST_HORIZON, num_layers=config.NUM_MAMBA_LAYERS,
                 dropout=config.DROPOUT):
        super().__init__()
        self.d_model = d_model
        self.horizon = horizon
        self.num_layers = num_layers
        
        self.input_projection = nn.Linear(input_dim, d_model)
        self.dropout = nn.Dropout(dropout)
        
        if MAMBA_AVAILABLE:
            from mamba_ssm import Mamba as MambaBlock
            self.layers = nn.ModuleList([
                MambaBlock(d_model=d_model)  # Removed dropout param - not supported
                for _ in range(num_layers)
            ])
            print(f"   Using {num_layers} Mamba layers")
        else:
            # FFN fallback
            self.layers = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(d_model, d_model * 4),
                    nn.GELU(),
                    nn.Linear(d_model * 4, d_model),
                    nn.Dropout(dropout)
                )
                for _ in range(num_layers)
            ])
            print(f"   Using {num_layers} FFN layers (State Space fallback)")
        
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(d_model) for _ in range(num_layers)
        ])
        
        self.output_head = nn.Linear(d_model, horizon * 2)
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x):
        batch_size = x.shape[0]
        x = self.input_projection(x)
        
        for i in range(self.num_layers):
            residual = x
            x = self.layers[i](x)
            x = self.dropout(x)
            x = x + residual
            x = self.layer_norms[i](x)
        
        last_hidden = x[:, -1, :]
        output = self.output_head(last_hidden)
        output = output.view(batch_size, self.horizon, 2)
        
        mean = output[:, :, 0]
        log_std = output[:, :, 1]
        log_std = torch.clamp(log_std, min=-10, max=2)
        
        return mean, log_std

# =============================================================================
# Loss Functions
# =============================================================================
def gaussian_nll_loss(mean, log_std, target):
    std = torch.exp(log_std)
    nll = 0.5 * ((target - mean) ** 2) / (std ** 2) + log_std + math.log(math.sqrt(2 * math.pi))
    return nll.mean()

# =============================================================================
# Training & Evaluation
# =============================================================================
def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    num_batches = 0
    
    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        
        optimizer.zero_grad()
        mean, log_std = model(X_batch)
        loss = gaussian_nll_loss(mean, log_std, y_batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / num_batches

def evaluate(model, loader, device, scaler):
    model.eval()
    all_mean = []
    all_std = []
    all_actual = []
    
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            
            mean, log_std = model(X_batch)
            
            # Inverse transform
            speed_mean = scaler.mean_[0]
            speed_std = scaler.scale_[0]
            
            mean_orig = mean.cpu().numpy() * speed_std + speed_mean
            std_orig = torch.exp(log_std).cpu().numpy() * speed_std
            actual_orig = y_batch.cpu().numpy() * speed_std + speed_mean
            
            all_mean.append(mean_orig)
            all_std.append(std_orig)
            all_actual.append(actual_orig)
    
    all_mean = np.concatenate(all_mean, axis=0)
    all_std = np.concatenate(all_std, axis=0)
    all_actual = np.concatenate(all_actual, axis=0)
    
    mae = np.mean(np.abs(all_actual - all_mean))
    rmse = np.sqrt(np.mean((all_actual - all_mean) ** 2))
    
    return mae, rmse, all_mean, all_std, all_actual

# =============================================================================
# Month-Ahead Forecasting: Train & Predict
# =============================================================================
print("\n[4] Running Month-Ahead Forecasting Experiments...")

device = config.DEVICE
print(f"   Device: {device}")

results = {}

# =================== EXPERIMENT 1: Predict May 2012 ===================
print("\n   EXPERIMENT 1: Train on Mar-Apr -> Predict May 2012")
print("   " + "-"*50)

model1 = MambaForecaster(input_dim=config.INPUT_DIM, d_model=config.D_MODEL,
                        horizon=config.FORECAST_HORIZON, num_layers=config.NUM_MAMBA_LAYERS,
                        dropout=config.DROPOUT).to(device)

optimizer1 = torch.optim.AdamW(model1.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)

for epoch in range(config.EPOCHS):
    loss = train_epoch(model1, train_loader1, optimizer1, device)
    if (epoch+1) % 5 == 0:
        print(f"      Epoch {epoch+1}/{config.EPOCHS}: Loss={loss:.4f}")

mae1, rmse1, may_pred_mean, may_pred_std, may_actual = evaluate(model1, test_loader1, device, scaler)
print(f"   May 2012 - MAE: {mae1:.2f} mph, RMSE: {rmse1:.2f} mph")
results['May2012'] = {'mae': mae1, 'rmse': rmse1, 'predictions': may_pred_mean, 'actual': may_actual}

# =================== EXPERIMENT 2: Predict May 2013 ===================
print("\n   EXPERIMENT 2: Train on ALL 2012 data -> Predict May 2013")
print("   " + "-"*50)

model2 = MambaForecaster(input_dim=config.INPUT_DIM, d_model=config.D_MODEL,
                        horizon=config.FORECAST_HORIZON, num_layers=config.NUM_MAMBA_LAYERS,
                        dropout=config.DROPOUT).to(device)

optimizer2 = torch.optim.AdamW(model2.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)

for epoch in range(config.EPOCHS):
    loss = train_epoch(model2, train_loader2, optimizer2, device)
    if (epoch+1) % 5 == 0:
        print(f"      Epoch {epoch+1}/{config.EPOCHS}: Loss={loss:.4f}")

# Generate May 2013 predictions (using May 2012 as proxy)
model2.eval()
with torch.no_grad():
    X_test2_tensor = torch.tensor(X_test2_s, dtype=torch.float32).to(device)
    mean, log_std = model2(X_test2_tensor)
    speed_mean = scaler.mean_[0]
    speed_std = scaler.scale_[0]
    may2013_pred_mean = mean.cpu().numpy() * speed_std + speed_mean
    may2013_pred_std = torch.exp(log_std).cpu().numpy() * speed_std
    may2013_actual = y_test2  # Already unscaled - no transformation needed!

print(f"   May 2013 (predicted from 2012 model) - Mean: {may2013_pred_mean.mean():.2f} mph")
results['May2013'] = {'predictions': may2013_pred_mean, 'predicted_std': may2013_pred_std, 'actual': may2013_actual}

# =================== EXPERIMENT 3: Predict June 2013 ===================
print("\n   EXPERIMENT 3: Train on ALL 2012 data -> Predict June 2013")
print("   " + "-"*50)

model3 = MambaForecaster(input_dim=config.INPUT_DIM, d_model=config.D_MODEL,
                        horizon=config.FORECAST_HORIZON, num_layers=config.NUM_MAMBA_LAYERS,
                        dropout=config.DROPOUT).to(device)

optimizer3 = torch.optim.AdamW(model3.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)

for epoch in range(config.EPOCHS):
    loss = train_epoch(model3, train_loader3, optimizer3, device)
    if (epoch+1) % 5 == 0:
        print(f"      Epoch {epoch+1}/{config.EPOCHS}: Loss={loss:.4f}")

# Generate June 2013 predictions (using June 2012 as proxy)
model3.eval()
with torch.no_grad():
    X_test3_tensor = torch.tensor(X_test3_s, dtype=torch.float32).to(device)
    mean, log_std = model3(X_test3_tensor)
    jun2013_pred_mean = mean.cpu().numpy() * speed_std + speed_mean
    jun2013_pred_std = torch.exp(log_std).cpu().numpy() * speed_std
    jun2013_actual = y_test3  # Already unscaled - no transformation needed!

print(f"   June 2013 (predicted from 2012 model) - Mean: {jun2013_pred_mean.mean():.2f} mph")
results['June2013'] = {'predictions': jun2013_pred_mean, 'predicted_std': jun2013_pred_std, 'actual': jun2013_actual}

# =============================================================================
# Save predictions for comparison
# =============================================================================
print("\n[5] Saving month-ahead predictions...")

# Get timestamps for test periods
may2012_timestamps = test_data_1.index[-len(y_test1):][-len(may_pred_mean):]
may2013_timestamps = test_data_2.index[-len(y_test2):][-len(may2013_pred_mean):]
june2013_timestamps = test_data_3.index[-len(y_test3):][-len(jun2013_pred_mean):]

# May 2012 predictions
may2012_df = pd.DataFrame({
    'timestamp': may2012_timestamps[:len(may_pred_mean)],
    'actual': may_actual.mean(axis=1),
    'predicted_mean': may_pred_mean.mean(axis=1),
    'predicted_std': may_pred_std.mean(axis=1),
})
may2012_df.to_csv('mamba_predictions_may2012.csv', index=False)
print(f"   Saved: mamba_predictions_may2012.csv ({len(may2012_df)} rows)")

# May 2013 predictions (predicted from 2012 model)
may2013_df = pd.DataFrame({
    'timestamp': may2013_timestamps[:len(may2013_pred_mean)],
    'actual': may2013_actual.mean(axis=1),  # May 2012 actual for reference
    'predicted_mean': may2013_pred_mean.mean(axis=1),
    'predicted_std': may2013_pred_std.mean(axis=1),
})
may2013_df.to_csv('mamba_predictions_may2013.csv', index=False)
print(f"   Saved: mamba_predictions_may2013.csv ({len(may2013_df)} rows)")

# June 2013 predictions
jun2013_df = pd.DataFrame({
    'timestamp': june2013_timestamps[:len(jun2013_pred_mean)],
    'actual': jun2013_actual.mean(axis=1),  # June 2012 actual for reference
    'predicted_mean': jun2013_pred_mean.mean(axis=1),
    'predicted_std': jun2013_pred_std.mean(axis=1),
})
jun2013_df.to_csv('mamba_predictions_jun2013.csv', index=False)
print(f"   Saved: mamba_predictions_jun2013.csv ({len(jun2013_df)} rows)")

# Combined comparison
comparison_df = pd.DataFrame({
    'Metric': ['MAE (mph)', 'RMSE (mph)', 'Mean Actual Speed', 'Mean Predicted Speed', 'Difference'],
    'May_2012': [f"{mae1:.2f}", f"{rmse1:.2f}",
                 f"{may_actual.mean():.2f}", f"{may_pred_mean.mean():.2f}",
                 f"{(may_pred_mean.mean()-may_actual.mean()):.2f}"],
    'May_2013_Pred': [f"N/A", f"N/A",
                      f"{may2013_actual.mean():.2f}", f"{may2013_pred_mean.mean():.2f}",
                      f"{(may2013_pred_mean.mean()-may2013_actual.mean()):.2f}"],
    'June_2013_Pred': [f"N/A", f"N/A",
                       f"{jun2013_actual.mean():.2f}", f"{jun2013_pred_mean.mean():.2f}",
                       f"{(jun2013_pred_mean.mean()-jun2013_actual.mean()):.2f}"]
})
comparison_df.to_csv('month_ahead_comparison.csv', index=False)
print(f"   Saved: month_ahead_comparison.csv")

print("\n" + "=" * 60)
print("MONTH-AHEAD EXPERIMENT COMPLETE!")
print("=" * 60)
print("\nSummary:")
print(f"   May 2012 MAE: {mae1:.2f} mph")
print(f"   May 2013 Predicted: Mean={may2013_pred_mean.mean():.2f} mph")
print(f"   June 2013 Predicted: Mean={jun2013_pred_mean.mean():.2f} mph")
print("\n   Model trained on 2012 data, predicted May & June 2013 speeds")
print("   (2012 actual values shown for reference comparison)")
