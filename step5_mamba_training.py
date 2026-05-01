"""
Step 5: Mamba Training for Traffic Forecasting
===============================================
This script trains a Mamba-based model for multivariate traffic forecasting.
It automatically extracts temporal patterns (daily/weekly) from the data
and combines them with weather features for intelligent prediction.

UPDATED: This version includes automatic temporal pattern extraction!
- Hourly patterns (cyclical encoding)
- Weekly patterns (day of week)
- Weather + Traffic combined

Author: Suvarna Kotha & Ruthik Garapati
Thesis: Urban Traffic Forecasting - Comparative Analysis

Requirements:
    pip install mamba-ssm causal-conv1d torch pandas numpy scikit-learn
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import time
import math
import os

try:
    from mamba_ssm import Mamba
    MAMBA_AVAILABLE = True
    print("Using Mamba blocks")
except ImportError:
    print("WARNING: mamba_ssm not installed. Using FFN State Space fallback.")
    MAMBA_AVAILABLE = False
    Mamba = None

# ============================================================================
# Configuration
# ============================================================================
class Config:
    # Data paths
    DATA_PATH = 'METR_LA_with_Weather_5min.csv'
    
    # Window sizes
    LOOKBACK_WINDOW = 24   # 24 steps = 2 hours of history
    FORECAST_HORIZON = 12  # 12 steps = 1 hour ahead
    
    # Features: [speed, precipitation_mm, wind_speed_kmh, hour_sin, hour_cos, day_of_week]
    # Temporal features are automatically extracted from timestamps
    TARGET_COL = 'speed'
    
    # Model architecture - now includes temporal features
    # Model architecture - now includes temporal features
    # Features: speed + weather (2) + temporal (6) = 9 features
    INPUT_DIM = 9         # [speed, precip, wind, hour_sin, hour_cos, day_sin, day_cos, month_sin, month_cos]
    D_MODEL = 64         # Reduced hidden dimension for faster CPU training
    NUM_MAMBA_LAYERS = 2  # Number of Mamba layers
    DROPOUT = 0.1
    
    # Training
    BATCH_SIZE = 64
    EPOCHS = 10  # Reduced for faster execution on CPU
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-5
    
    # Data split - use smaller subset for faster training
    TRAIN_RATIO = 0.7
    VAL_RATIO = 0.1
    TEST_RATIO = 0.2
    
    # Subsample for faster training on CPU
    SUBSAMPLE_RATE = 4  # Use every 4th sample
    
    # Device
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Random seed
    SEED = 42

config = Config()

torch.manual_seed(config.SEED)
np.random.seed(config.SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(config.SEED)

# ============================================================================
# Automatic Temporal Feature Extraction
# ============================================================================

def extract_temporal_features(df):
    """
    Automatically extract temporal patterns from timestamps:
    - Hour cyclical encoding (sin/cos for 24-hour cycle)
    - Day of week encoding
    - Month encoding
    
    This allows the model to LEARN patterns like:
    - Morning rush (7-9 AM)
    - Evening rush (4-7 PM)
    - Weekday vs weekend patterns
    - Monthly patterns
    
    The model will discover these patterns from the data, not predefined!
    """
    # Extract temporal components
    hours = df.index.hour
    days = df.index.dayofweek
    months = df.index.month
    
    # Cyclical encoding for hours (24-hour cycle)
    hour_sin = np.sin(2 * np.pi * hours / 24)
    hour_cos = np.cos(2 * np.pi * hours / 24)
    
    # Cyclical encoding for day of week (7-day cycle)
    day_sin = np.sin(2 * np.pi * days / 7)
    day_cos = np.cos(2 * np.pi * days / 7)
    
    # Cyclical encoding for month (12-month cycle)
    month_sin = np.sin(2 * np.pi * months / 12)
    month_cos = np.cos(2 * np.pi * months / 12)
    
    return hour_sin, hour_cos, day_sin, day_cos, month_sin, month_cos


def analyze_temporal_patterns(data, df_index):
    """
    Analyze and report discovered temporal patterns in the data.
    This shows what patterns the model will learn from.
    """
    print("\n    Analyzing temporal patterns in data...")
    
    # Hourly pattern
    hourly_mean = data.groupby(df_index.hour)['speed'].mean()
    max_hour = hourly_mean.idxmax()
    min_hour = hourly_mean.idxmin()
    
    # Weekly pattern
    daily_mean = data.groupby(df_index.dayofweek)['speed'].mean()
    max_day = daily_mean.idxmax()
    min_day = daily_mean.idxmin()
    
    day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    
    print(f"    - Peak traffic hour: {max_hour}:00 ({hourly_mean[max_hour]:.1f} mph avg)")
    print(f"    - Lowest traffic hour: {min_hour}:00 ({hourly_mean[min_hour]:.1f} mph avg)")
    print(f"    - Busiest day: {day_names[max_day]} ({daily_mean[max_day]:.1f} mph avg)")
    print(f"    - Quietest day: {day_names[min_day]} ({daily_mean[min_day]:.1f} mph avg)")
    print("    - Model will learn these patterns automatically from temporal features!")


# ============================================================================
# Data Download Function
# ============================================================================

def download_metr_la_data():
    """
    Download METR-LA dataset from official source.
    Returns path to downloaded file.
    """
    print("=" * 60)
    print("Downloading METR-LA Dataset")
    print("=" * 60)
    
    # Check if file already exists
    if os.path.exists(config.DATA_PATH):
        print(f"File already exists: {config.DATA_PATH}")
        return config.DATA_PATH
    
    # Download URLs for METR-LA
    # Using GraphWaveNet repository data
    metr_la_url = "https://github.com/VeritasYo/Graph-WaveNet/raw/master/data/METR-LA.zip"
    
    print("\n[1] This script requires METR-LA data with weather.")
    print("    Since direct download is complex, we will use a workaround...")
    print("\n    Option A: Download from official source (requires processing)")
    print("    Option B: Use sample data generation for testing")
    print("    Option C: Continue with existing file if available")
    
    # Check what data is available
    if os.path.exists('METR-LA_cleaned.csv') and os.path.exists('LA_Weather_Hourly_2012_Full.csv'):
        print("\n    Found METR-LA_cleaned.csv and weather data!")
        print("    Will merge them to create the required dataset...")
        return None  # Will merge existing files
    else:
        print("\n    ERROR: Required data files not found!")
        print("    Please ensure METR-LA_cleaned.csv exists in the directory.")
        return None

def merge_existing_data():
    """
    If METR-LA and weather data exist separately, merge them.
    """
    print("\n[2] Merging METR-LA traffic + weather data...")
    
    # Load traffic data
    df_traffic = pd.read_csv('METR-LA_cleaned.csv', index_col=0)
    df_traffic.index = pd.to_datetime(df_traffic.index)
    print(f"    - Traffic shape: {df_traffic.shape}")
    
    # Load weather data
    df_weather = pd.read_csv('LA_Weather_Hourly_2012_Full.csv')
    df_weather['datetime'] = pd.to_datetime(df_weather['datetime'])
    df_weather = df_weather.set_index('datetime')
    print(f"    - Weather shape: {df_weather.shape}")
    
    # Resample weather to 5-min intervals
    df_weather_5min = df_weather.resample('5min').ffill()
    
    # Find overlapping period
    traffic_start = df_traffic.index.min()
    traffic_end = df_traffic.index.max()
    weather_start = df_weather_5min.index.min()
    weather_end = df_weather_5min.index.max()
    
    overlap_start = max(traffic_start, weather_start)
    overlap_end = min(traffic_end, weather_end)
    
    # Filter to overlapping period
    df_traffic_filtered = df_traffic.loc[overlap_start:overlap_end]
    df_weather_filtered = df_weather_5min.loc[overlap_start:overlap_end]
    
    # Add weather prefix
    df_weather_filtered = df_weather_filtered.add_prefix('weather_')
    
    # Merge
    merged_df = df_traffic_filtered.join(df_weather_filtered, how='inner')
    merged_df = merged_df.ffill().bfill()
    
    # Save merged data
    merged_df.to_csv(config.DATA_PATH)
    print(f"\n    - Saved merged data to: {config.DATA_PATH}")
    print(f"    - Shape: {merged_df.shape}")
    
    return config.DATA_PATH

# ============================================================================
# Data Loading and Preprocessing
# ============================================================================

def load_and_preprocess_data():
    """
    Load the METR-LA dataset and extract relevant features.
    """
    print("=" * 60)
    print("Loading and Preprocessing Data")
    print("=" * 60)
    
    # Try to download/merge data
    data_path = download_metr_la_data()
    
    if data_path is None:
        # Merge existing files
        data_path = merge_existing_data()
    
    # Load the dataset
    print("\n[1] Loading dataset...")
    df = pd.read_csv(config.DATA_PATH, index_col=0)
    df.index = pd.to_datetime(df.index)
    print(f"    - Full dataset shape: {df.shape}")
    print(f"    - Date range: {df.index.min()} to {df.index.max()}")
    
    # Identify speed and weather columns
    weather_cols = [col for col in df.columns if col.startswith('weather_')]
    speed_cols = [col for col in df.columns if col not in weather_cols]
    speed_col = speed_cols[0] if speed_cols else df.columns[0]
    
    print(f"    - Selected speed column: {speed_col}")
    print(f"    - Weather columns: {weather_cols}")
    
    # Extract temporal features automatically from timestamps
    hour_sin, hour_cos, day_sin, day_cos, month_sin, month_cos = extract_temporal_features(df)
    
    # Create DataFrame with ALL features: traffic + weather + temporal
    data = pd.DataFrame({
        'speed': speed_data,
        'precipitation_mm': precip_data,
        'wind_speed_kmh': wind_data,
        'hour_sin': hour_sin,
        'hour_cos': hour_cos,
        'day_sin': day_sin,
        'day_cos': day_cos,
        'month_sin': month_sin,
        'month_cos': month_cos,
    }, index=df.index)
    
    data = data.ffill().bfill().dropna()
    
    # Analyze discovered patterns
    analyze_temporal_patterns(data, data.index)
    
    print(f"\n[2] Data shape after preprocessing: {data.shape}")
    print(f"    - Speed range: {data['speed'].min():.2f} - {data['speed'].max():.2f} mph")
    print(f"    - Precipitation range: {data['precipitation_mm'].min():.2f} - {data['precipitation_mm'].max():.2f} mm")
    print(f"    - Wind speed range: {data['wind_speed_kmh'].min():.2f} - {data['wind_speed_kmh'].max():.2f} km/h")
    print(f"    - Temporal features: hour_sin/cos, day_sin/cos, month_sin/cos (cyclical encoding)")
    print(f"    - Model will LEARN patterns like rush hours from temporal features!")
    
    return data

def create_scalers(data):
    scaler = StandardScaler()
    scaler.fit(data.values)
    return scaler

def create_sliding_windows(data, scaler, lookback=config.LOOKBACK_WINDOW, 
                           horizon=config.FORECAST_HORIZON):
    print("\n[3] Creating sliding windows...")
    
    scaled_data = scaler.transform(data.values)
    
    X, y = [], []
    
    for i in range(len(scaled_data) - lookback - horizon + 1):
        X.append(scaled_data[i:i+lookback])
        y.append(scaled_data[i+lookback:i+lookback+horizon, 0])
    
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)
    
    print(f"    - Total samples: {len(X)}")
    print(f"    - X shape: {X.shape}")
    print(f"    - y shape: {y.shape}")
    
    # Split data
    n = len(X)
    train_end = int(n * config.TRAIN_RATIO)
    val_end = int(n * (config.TRAIN_RATIO + config.VAL_RATIO))
    
    X_train, X_val, X_test = X[:train_end], X[train_end:val_end], X[val_end:]
    y_train, y_val, y_test = y[:train_end], y[train_end:val_end], y[val_end:]
    
    print(f"\n[4] Data split:")
    print(f"    - Train: {len(X_train)} samples ({config.TRAIN_RATIO*100:.0f}%)")
    print(f"    - Val:   {len(X_val)} samples ({config.VAL_RATIO*100:.0f}%)")
    print(f"    - Test:  {len(X_test)} samples ({config.TEST_RATIO*100:.0f}%)")
    
    # Subsample if configured (for faster CPU training)
    if config.SUBSAMPLE_RATE > 1:
        print(f"\n[5] Subsampling every {config.SUBSAMPLE_RATE}th sample for faster training...")
        X_train = X_train[::config.SUBSAMPLE_RATE]
        X_val = X_val[::config.SUBSAMPLE_RATE]
        X_test = X_test[::config.SUBSAMPLE_RATE]
        y_train = y_train[::config.SUBSAMPLE_RATE]
        y_val = y_val[::config.SUBSAMPLE_RATE]
        y_test = y_test[::config.SUBSAMPLE_RATE]
        print(f"    - After subsampling: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test, scaler

# ============================================================================
# PyTorch Dataset
# ============================================================================

class TrafficDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# ============================================================================
# Mamba Model
# ============================================================================

class MambaForecaster(nn.Module):
    def __init__(self, input_dim=config.INPUT_DIM, d_model=config.D_MODEL, 
                 horizon=config.FORECAST_HORIZON, num_layers=config.NUM_MAMBA_LAYERS,
                 dropout=config.DROPOUT):
        super(MambaForecaster, self).__init__()
        
        self.d_model = d_model
        self.horizon = horizon
        self.num_layers = num_layers
        self.use_mamba = MAMBA_AVAILABLE
        
        self.input_projection = nn.Linear(input_dim, d_model)
        self.dropout = nn.Dropout(dropout)
        
        # Create layers based on availability
        if MAMBA_AVAILABLE:
            from mamba_ssm import Mamba as MambaBlock
            self.layers = nn.ModuleList([
                MambaBlock(d_model=d_model, dropout=dropout)
                for _ in range(num_layers)
            ])
            print(f"Using {num_layers} Mamba layers")
        else:
            # Use a simple feedforward + LayerNorm as State Space fallback
            # This is a simpler architecture that mimics state space behavior
            self.layers = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(d_model, d_model * 4),  # Expand
                    nn.GELU(),
                    nn.Linear(d_model * 4, d_model),  # Contract
                    nn.Dropout(dropout)
                )
                for _ in range(num_layers)
            ])
            print(f"Using {num_layers} FFN layers (State Space fallback)")
        
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(d_model)
            for _ in range(num_layers)
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
        
        # Project input to d_model
        x = self.input_projection(x)
        
        # Pass through layers
        for i in range(self.num_layers):
            residual = x
            x = self.layers[i](x)
            x = self.dropout(x)
            x = x + residual  # Residual connection
            x = self.layer_norms[i](x)
        
        # Use the last timestep's hidden state for prediction
        last_hidden = x[:, -1, :]
        output = self.output_head(last_hidden)
        output = output.view(batch_size, self.horizon, 2)
        
        mean = output[:, :, 0]
        log_std = output[:, :, 1]
        log_std = torch.clamp(log_std, min=-10, max=2)
        
        return mean, log_std

# ============================================================================
# Loss Functions
# ============================================================================

def gaussian_nll_loss(mean, log_std, target):
    std = torch.exp(log_std)
    nll = 0.5 * ((target - mean) ** 2) / (std ** 2) + log_std + math.log(math.sqrt(2 * math.pi))
    return nll.mean()

def calculate_kl_divergence(mean_pred, std_pred, actual):
    eps = 1e-8
    var_pred = std_pred ** 2
    actual_var = 0.1
    
    kl = 0.5 * (
        torch.log((var_pred + eps) / (actual_var + eps)) +
        (actual_var + (actual - mean_pred) ** 2) / (var_pred + eps) -
        1
    )
    return kl.mean()

# ============================================================================
# Training
# ============================================================================

def train_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0
    num_batches = 0
    
    start_time = time.time()
    
    for X_batch, y_batch in dataloader:
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
    
    epoch_time = time.time() - start_time
    avg_loss = total_loss / num_batches
    
    if torch.cuda.is_available():
        peak_memory = torch.cuda.max_memory_allocated() / 1024**2
    else:
        peak_memory = 0
    
    return avg_loss, epoch_time, peak_memory

def validate(model, dataloader, device):
    model.eval()
    total_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            
            mean, log_std = model(X_batch)
            loss = gaussian_nll_loss(mean, log_std, y_batch)
            
            total_loss += loss.item()
            num_batches += 1
    
    return total_loss / num_batches

# ============================================================================
# Evaluation
# ============================================================================

def evaluate(model, dataloader, device, scaler):
    model.eval()
    
    all_mean = []
    all_std = []
    all_actual = []
    inference_times = []
    
    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            
            start_inf = time.time()
            mean, log_std = model(X_batch)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            inference_times.append(time.time() - start_inf)
            
            speed_mean = scaler.mean_[0]
            speed_std = scaler.scale_[0]
            
            mean_orig = mean * speed_std + speed_mean
            std_orig = torch.exp(log_std) * speed_std
            actual_orig = y_batch * speed_std + speed_mean
            
            all_mean.append(mean_orig.cpu().numpy())
            all_std.append(std_orig.cpu().numpy())
            all_actual.append(actual_orig.cpu().numpy())
    
    all_mean = np.concatenate(all_mean, axis=0)
    all_std = np.concatenate(all_std, axis=0)
    all_actual = np.concatenate(all_actual, axis=0)
    
    mae = np.mean(np.abs(all_actual - all_mean))
    rmse = np.sqrt(np.mean((all_actual - all_mean) ** 2))
    
    kl_div = calculate_kl_divergence(
        torch.tensor(all_mean), 
        torch.tensor(all_std), 
        torch.tensor(all_actual)
    ).item()
    
    avg_inference_latency = np.mean(inference_times) * 1000
    
    return mae, rmse, kl_div, avg_inference_latency

# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 60)
    print("STEP 5: Mamba Training for Traffic Forecasting")
    print("=" * 60)
    
    device = config.DEVICE
    print(f"\n[0] Device: {device}")
    if torch.cuda.is_available():
        print(f"    - GPU: {torch.cuda.get_device_name(0)}")
        print(f"    - CUDA Version: {torch.version.cuda}")
    
    # Load data
    data = load_and_preprocess_data()
    scaler = create_scalers(data)
    X_train, X_val, X_test, y_train, y_val, y_test, scaler = create_sliding_windows(data, scaler)
    
    # Create DataLoaders
    print("\n[5] Creating DataLoaders...")
    
    train_dataset = TrafficDataset(X_train, y_train)
    val_dataset = TrafficDataset(X_val, y_val)
    test_dataset = TrafficDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    
    print(f"    - Train batches: {len(train_loader)}")
    print(f"    - Val batches: {len(val_loader)}")
    print(f"    - Test batches: {len(test_loader)}")
    
    # Initialize model
    print("\n[6] Initializing Mamba model...")
    
    model = MambaForecaster(
        input_dim=config.INPUT_DIM,
        d_model=config.D_MODEL,
        horizon=config.FORECAST_HORIZON,
        num_layers=config.NUM_MAMBA_LAYERS,
        dropout=config.DROPOUT
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"    - Total parameters: {total_params:,}")
    print(f"    - Trainable parameters: {trainable_params:,}")
    print(f"    - Model architecture: {config.NUM_MAMBA_LAYERS} Mamba layers, d_model={config.D_MODEL}")
    
    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    # Training loop
    print("\n[7] Starting training...")
    print("-" * 60)
    
    best_val_loss = float('inf')
    best_model_path = 'mamba_best_model.pt'
    train_losses = []
    val_losses = []
    
    for epoch in range(config.EPOCHS):
        train_loss, epoch_time, peak_memory = train_epoch(model, train_loader, optimizer, device)
        val_loss = validate(model, val_loader, device)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        scheduler.step(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)
        
        if (epoch + 1) % 5 == 0 or epoch == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch+1:3d}/{config.EPOCHS} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Time: {epoch_time:.2f}s | LR: {current_lr:.6f} | GPU Mem: {peak_memory:.2f} MB")
    
    print("-" * 60)
    print(f"Training complete! Best validation loss: {best_val_loss:.4f}")
    
    # Evaluate
    print("\n[8] Evaluating on test set...")
    
    model.load_state_dict(torch.load(best_model_path))
    
    mae, rmse, kl_div, inference_latency = evaluate(model, test_loader, device, scaler)
    
    print(f"\n{'='*60}")
    print("TEST SET RESULTS")
    print(f"{'='*60}")
    print(f"  MAE:                {mae:.4f} mph")
    print(f"  RMSE:               {rmse:.4f} mph")
    print(f"  KL Divergence:      {kl_div:.4f} bits")
    print(f"  Inference Latency: {inference_latency:.2f} ms/batch")
    
    # Save results
    print("\n[9] Saving results...")
    
    results = {
        'MAE': mae,
        'RMSE': rmse,
        'KL_Divergence': kl_div,
        'Inference_Latency_ms': inference_latency,
        'Epochs': config.EPOCHS,
        'Batch_Size': config.BATCH_SIZE,
        'Learning_Rate': config.LEARNING_RATE,
        'D_Model': config.D_MODEL,
        'Num_Mamba_Layers': config.NUM_MAMBA_LAYERS,
        'Lookback_Window': config.LOOKBACK_WINDOW,
        'Forecast_Horizon': config.FORECAST_HORIZON
    }
    
    results_df = pd.DataFrame([results])
    results_df.to_csv('mamba_evaluation_results.csv', index=False)
    print("    - Saved: mamba_evaluation_results.csv")
    
    history_df = pd.DataFrame({
        'epoch': range(1, len(train_losses) + 1),
        'train_loss': train_losses,
        'val_loss': val_losses
    })
    history_df.to_csv('mamba_training_history.csv', index=False)
    print("    - Saved: mamba_training_history.csv")
    
    print("\n" + "=" * 60)
    print("STEP 5 COMPLETE: Mamba training finished!")
    print("=" * 60)
    
    return results

if __name__ == "__main__":
    results = main()
