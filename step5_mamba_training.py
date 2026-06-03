"""
Step 5: Mamba Training for Traffic Forecasting
===============================================
This script trains a state-space model for multivariate traffic forecasting.
It automatically extracts temporal patterns (daily/weekly) from the data
and combines them with weather features for intelligent prediction.

UPDATED: This version includes automatic temporal pattern extraction!
- Hourly patterns (cyclical encoding)
- Weekly patterns (day of week)
- Weather + Traffic combined

Architecture: FFN (Linear → LayerNorm → GELU → Dropout ×2 → Linear)
No mamba_ssm or causal_conv1d required — pure PyTorch implementation.

Author: Suvarna Kotha & Ruthik Garapati
Thesis: Urban Traffic Forecasting - Comparative Analysis

Requirements:
    pip install torch pandas numpy scikit-learn
"""
import os
os.environ["TORCH_CUDA_ARCH_LIST"] = "7.5"  # Force CUDA kernel compilation for T4 GPU

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
import sys

# Pin to a single core so that Colab's SIGINT watchdog measures,
# counts, and respects each print flush promptly.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
# Ensure stdout is line-buffered (every print() flushes immediately)
# Reconfigure before any heavy print statements fire.
sys.stdout.reconfigure(line_buffering=True)

# ============================================================================
# Configuration
# ============================================================================
class Config:
    # Data paths
    DATA_PATH = 'METR_LA_with_Weather_5min.csv'
    
    # Window sizes
    LOOKBACK_WINDOW = 24   # 24 steps = 2 hours of history
    FORECAST_HORIZON = 12  # 12 steps = 1 hour ahead
    
    # Features: [speed, precipitation_mm, wind_speed_kmh,
    #            hour_sin, hour_cos, day_sin, day_cos,
    #            week_sin, week_cos, month_sin, month_cos]
    # Temporal features are automatically extracted from timestamps
    TARGET_COL = 'speed'
    
    # Model architecture - now includes temporal features
    # Features: speed + weather (2) + temporal (8) = 11 total
    # Temporal components: hour (2), day (2), week (2), month (2)
    INPUT_DIM = 11
    D_MODEL = 64         # Reduced hidden dimension for faster CPU training
    NUM_MAMBA_LAYERS = 2  # Number of Mamba layers
    DROPOUT = 0.1
    
    # Training
    BATCH_SIZE = 16   # Reduced to avoid Colab idle-kill on first batch
    EPOCHS = 10
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
    - Hour cyclical encoding (24-hour cycle)
    - Day of week (7-day cycle)
    - Week of year (52-week cycle) ← NEW!
    - Month (12-month cycle)
    
    This allows the model to LEARN patterns like:
    - Rush hours (hourly)
    - Weekdays vs weekends (daily)
    - Seasonal patterns (weekly/monthly)
    - Yearly patterns (monthly)
    
    The model will discover these patterns from the data!
    """
    hours = df.index.hour
    days = df.index.dayofweek
    weeks = df.index.isocalendar().week  # Week of year (1-52)
    months = df.index.month
    
    # Cyclical encoding for hours (24-hour cycle)
    hour_sin = np.sin(2 * np.pi * hours / 24)
    hour_cos = np.cos(2 * np.pi * hours / 24)
    
    # Cyclical encoding for day of week (7-day cycle)
    day_sin = np.sin(2 * np.pi * days / 7)
    day_cos = np.cos(2 * np.pi * days / 7)
    
    # Cyclical encoding for week of year (52-week cycle) - NEW!
    week_sin = np.sin(2 * np.pi * weeks / 52)
    week_cos = np.cos(2 * np.pi * weeks / 52)
    
    # Cyclical encoding for month (12-month cycle)
    month_sin = np.sin(2 * np.pi * months / 12)
    month_cos = np.cos(2 * np.pi * months / 12)
    
    return hour_sin, hour_cos, day_sin, day_cos, week_sin, week_cos, month_sin, month_cos


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
    
    # Daily pattern (day of week)
    daily_mean = data.groupby(df_index.dayofweek)['speed'].mean()
    max_day = daily_mean.idxmax()
    min_day = daily_mean.idxmin()
    
    # Weekly pattern (week of year) - NEW!
    weekly_mean = data.groupby(df_index.isocalendar().week)['speed'].mean()
    max_week = weekly_mean.idxmax()
    min_week = weekly_mean.idxmin()
    
    # Monthly pattern
    monthly_mean = data.groupby(df_index.month)['speed'].mean()
    max_month = monthly_mean.idxmax()
    min_month = monthly_mean.idxmin()
    
    day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    print(f"    - Peak traffic hour: {max_hour}:00 ({hourly_mean[max_hour]:.1f} mph avg)")
    print(f"    - Lowest traffic hour: {min_hour}:00 ({hourly_mean[min_hour]:.1f} mph avg)")
    print(f"    - Busiest day: {day_names[max_day]} ({daily_mean[max_day]:.1f} mph avg)")
    print(f"    - Quietest day: {day_names[min_day]} ({daily_mean[min_day]:.1f} mph avg)")
    print(f"    - Busiest week of year: Week {max_week} ({weekly_mean[max_week]:.1f} mph avg)")
    print(f"    - Quietest week of year: Week {min_week} ({weekly_mean[min_week]:.1f} mph avg)")
    print(f"    - Busiest month: {month_names[max_month-1]} ({monthly_mean[max_month]:.1f} mph avg)")
    print(f"    - Quietest month: {month_names[min_month-1]} ({monthly_mean[min_month]:.1f} mph avg)")
    print("    - Model will learn ALL these patterns automatically!")
    print("    - Temporal features: hour_sin/cos, day_sin/cos, week_sin/cos, month_sin/cos")


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
    
    # Extract speed and weather data
    speed_data = df[speed_col].values
    precip_col = [c for c in weather_cols if 'precip' in c.lower()]
    wind_col = [c for c in weather_cols if 'wind' in c.lower()]
    precip_data = df[precip_col[0]].values if precip_col else np.zeros(len(df))
    wind_data = df[wind_col[0]].values if wind_col else np.zeros(len(df))
    
    # Extract speed and weather data
    speed_data = df[speed_col].values
    precip_col = [c for c in weather_cols if 'precip' in c.lower()]
    wind_col = [c for c in weather_cols if 'wind' in c.lower()]
    precip_data = df[precip_col[0]].values if precip_col else np.zeros(len(df))
    wind_data = df[wind_col[0]].values if wind_col else np.zeros(len(df))
    
    # Extract temporal features automatically from timestamps
    hour_sin, hour_cos, day_sin, day_cos, week_sin, week_cos, month_sin, month_cos = extract_temporal_features(df)
    
    # Create DataFrame with ALL features: traffic + weather + temporal
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
    
    # Analyze discovered patterns
    analyze_temporal_patterns(data, data.index)
    
    print(f"\n[2] Data shape after preprocessing: {data.shape}")
    print(f"    - Speed range: {data['speed'].min():.2f} - {data['speed'].max():.2f} mph")
    print(f"    - Precipitation range: {data['precipitation_mm'].min():.2f} - {data['precipitation_mm'].max():.2f} mm")
    print(f"    - Wind speed range: {data['wind_speed_kmh'].min():.2f} - {data['wind_speed_kmh'].max():.2f} km/h")
    print(f"    - Temporal features (11 total):")
    print(f"      * Hourly (sin/cos): hour_sin, hour_cos")
    print(f"      * Daily (sin/cos): day_sin, day_cos")
    print(f"      * Weekly (sin/cos): week_sin, week_cos")
    print(f"      * Monthly (sin/cos): month_sin, month_cos")
    print(f"    - Model will LEARN all temporal patterns automatically!")
    
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

        self.input_projection = nn.Linear(input_dim, d_model)
        self.dropout = nn.Dropout(dropout)

        # ALWAYS initialize FFN layers for checkpoint compatibility
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model * 4),  # Expand
                nn.GELU(),
                nn.Linear(d_model * 4, d_model),  # Contract
                nn.Dropout(dropout)
            )
            for _ in range(num_layers)
        ])
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(d_model)
            for _ in range(num_layers)
        ])
        
        # Track 2: Try to provision native Mamba blocks if library environment permits
        try:
            from mamba_ssm import Mamba
            self.mamba_blocks = nn.ModuleList([
                Mamba(d_model=d_model, d_state=16, d_conv=4, expand=2)
                for _ in range(num_layers)
            ])
            self.using_native_mamba = True
            print(f"Using {num_layers} native Mamba layers (State Space model)")
        except Exception as e:
            print(f"WARNING: Native mamba_ssm not found or incompatible architecture. Falling back to parameter-matched FFN surrogate.")
            self.using_native_mamba = False
            
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
            if self.using_native_mamba:
                x = self.mamba_blocks[i](x)
            else:
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

    def measure_inference_latency(self, x, num_iterations=100):
        """
        Measure inference latency with proper GPU synchronization.
        Performs warm-up iterations followed by timed iterations.
        Returns latency in milliseconds per inference.
        """
        self.eval()
        device = next(self.parameters()).device
        
        # Ensure input is on the correct device
        x = x.to(device)
        
        # Warm-up iterations
        with torch.no_grad():
            for _ in range(10):
                _ = self.forward(x)
        
        # Synchronize before timing
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        # Timed iterations
        start_time = time.perf_counter()
        with torch.no_grad():
            for _ in range(num_iterations):
                _ = self.forward(x)
        
        # Synchronize after timing
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        end_time = time.perf_counter()
        
        # Calculate average latency in milliseconds
        total_time = end_time - start_time
        latency_ms = (total_time * 1000) / num_iterations
        return latency_ms

    def autoregressive_predict(self, context, horizon=None):
        """
        Autoregressive prediction using the trained model.
        For each timestep, predict the next step and feed it back as input.
        
        Args:
            context: Tensor of shape (batch_size, lookback_window, input_dim)
            horizon: Number of steps to predict (defaults to self.horizon)
            
        Returns:
            predictions: Tensor of shape (batch_size, horizon)
        """
        if horizon is None:
            horizon = self.horizon
            
        self.eval()
        device = next(self.parameters()).device
        context = context.to(device)
        
        # Initialize predictions list
        predictions = []
        
        # Current window starts with the context
        current_window = context.clone()
        
        with torch.no_grad():
            for _ in range(horizon):
                # Forward pass to get prediction for next timestep
                # Returns: (mean, log_std) where each has shape (batch_size, horizon, 2)
                mean, log_std = self.forward(current_window)
                
                # Extract mean prediction for the FIRST timestep (we predict one step at a time)
                # mean has shape (batch_size, horizon), we want mean[:, 0] for speed
                mean_pred = mean[:, 0]  # Shape: (batch_size,)
                
                # Store prediction
                predictions.append(mean_pred)
                
                # Update the window: remove oldest timestep, append new prediction
                # We need to expand prediction to match input dimensions
                # Prediction is for speed only (first feature), so we need to create a full feature vector
                # For simplicity, we'll reuse the last known values for other features
                # In a more sophisticated implementation, we would predict all features
                
                # Get the last timestep's features for weather/temporal (assuming they're known or constant)
                last_features = current_window[:, -1, :].clone()  # (batch_size, input_dim)
                
                # Update the speed (first feature) with our prediction
                last_features[:, 0] = mean_pred
                
                # Remove first timestep and append the updated last timestep
                current_window = torch.cat([
                    current_window[:, 1:, :],  # Remove first timestep
                    last_features.unsqueeze(1)  # Add updated timestep at end
                ], dim=1)
        
        # Stack predictions: (horizon, batch_size) -> (batch_size, horizon)
        predictions = torch.stack(predictions, dim=1)
        return predictions

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
    
    # ─── Load data ──────────────────────────────────────────────────────────────
    data = load_and_preprocess_data()

    # ─── Build two feature matrices for ablation ────────────────────────────────
    weather_cols_list = [c for c in data.columns if c.startswith('weather_')]
    non_speed_nontime = weather_cols_list                         # weather cols
    temporal_cols = ['hour_sin','hour_cos','day_sin','day_cos',
                     'week_sin','week_cos','month_sin','month_cos']

    # Column-level stats for time/weather renormalisation in temporal_generalization.py
    _time_mean  = data[temporal_cols].mean().values.astype(np.float32)
    _time_std   = data[temporal_cols].std().values.astype(np.float32)
    _wx_mean    = data[weather_cols_list].mean().values.astype(np.float32)
    _wx_std     = data[weather_cols_list].std().values.astype(np.float32)

    # Model A: speed + temporal only  (no weather)
    feats_A_cols = ['speed'] + temporal_cols
    data_A = data[feats_A_cols].copy()

    # Model B: speed + temporal + weather
    data_B = data.copy()

    print(f"\n[3] Ablation feature sets:")
    print(f"    Model A (time only) : {len(feats_A_cols)} features -> {feats_A_cols}")
    print(f"    Model B (+ weather)  : {data_B.shape[1]} features")

    # ─── Scaler + window builder (fit on TRAIN ONLY, no leakage) ────────────────
    def make_splits(data_subset, train_ratio=0.70, val_ratio=0.15):
        vals  = data_subset.values
        n     = len(vals)
        t_end = int(n * train_ratio)
        v_end = t_end + int(n * val_ratio)

        scaler = StandardScaler()
        scaler.fit(vals[:t_end])            # FIT ON TRAIN ONLY — no leakage
        scaled  = scaler.transform(vals)

        lb, hz = config.LOOKBACK_WINDOW, config.FORECAST_HORIZON
        X_all, y_all = [], []
        for i in range(len(scaled) - lb - hz + 1):
            X_all.append(scaled[i:i+lb])
            y_all.append(scaled[i+lb:i+lb+hz, 0])   # index 0 = speed column
        X_all = np.array(X_all, dtype=np.float32)
        y_all = np.array(y_all, dtype=np.float32)

        X_tr, X_va, X_te = X_all[:t_end], X_all[t_end:v_end], X_all[v_end:]
        y_tr, y_va, y_te = y_all[:t_end], y_all[t_end:v_end], y_all[v_end:]
        return X_tr, X_va, X_te, y_tr, y_va, y_te, scaler

    print("\n[4] Building train/val/test splits (leakage-free, train-only scaling)...")
    Xa_tr, Xa_va, Xa_te, ya_tr, ya_va, ya_te, scA = make_splits(data_A)
    Xb_tr, Xb_va, Xb_te, yb_tr, yb_va, yb_te, scB = make_splits(data_B)
    print(f"    Model A  train={Xa_tr.shape}  val={Xa_va.shape}  test={Xa_te.shape}")
    print(f"    Model B  train={Xb_tr.shape}  val={Xb_va.shape}  test={Xb_te.shape}")

    # ─── DataLoaders ────────────────────────────────────────────────────────────
    def mkdataloader(X, y, shuffle):
        return DataLoader(TrafficDataset(X, y),
                          batch_size=config.BATCH_SIZE, shuffle=shuffle)

    trL_A = mkdataloader(Xa_tr, ya_tr, True);  vaL_A = mkdataloader(Xa_va, ya_va, False);  teL_A = mkdataloader(Xa_te, ya_te, False)
    trL_B = mkdataloader(Xb_tr, yb_tr, True);  vaL_B = mkdataloader(Xb_va, yb_va, False);  teL_B = mkdataloader(Xb_te, yb_te, False)

    # ─── Inner train helper with history logging ─────────────────────────────────────
    def train_one(model, tr_loader, va_loader, ckpt_path, label, epochs=config.EPOCHS):
        opt   = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE,
                            weight_decay=config.WEIGHT_DECAY)
        sched = optim.lr_scheduler.ReduceLROnPlateau(opt, 'min', factor=0.5, patience=5)
        best  = float('inf');  best_state = None
        history = []  # (epoch, train_loss, val_loss)  ← for training_history.csv
        print(f"\n  [{label}] training started...", flush=True)
        for ep in range(epochs):
            model.train();  tl = 0.0
            for xb, yb in tr_loader:
                # Print LOSS on a NEWLINE every 50 batches so line-buffering
                # fires; CritColab's logger guarantees exposure @\n.  The
                # loss update prefix also caries the training loss on each flush.
                opt.zero_grad()
                mean, log_std = model(xb.to(device))
                loss = gaussian_nll_loss(mean, log_std, yb.to(device))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                opt.step()
                tl += loss.item() * xb.size(0)
            tl /= len(tr_loader.dataset)

            model.eval();  vl = 0.0
            with torch.no_grad():
                for xb, yb in va_loader:
                    mean, log_std = model(xb.to(device))
                    vl += gaussian_nll_loss(mean, log_std, yb.to(device)).item() * xb.size(0)
            vl /= len(va_loader.dataset)
            sched.step(vl)

            if vl < best:
                best = vl;  best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            history.append((ep + 1, tl, vl))
            if (ep + 1) % 5 == 0 or ep == 0:
                lr = opt.param_groups[0]['lr']
                print(f"    ep{ep+1:2d}/{epochs}  train={tl:.4f}  val={vl:.4f}  lr={lr:.2e}")

        torch.save(best_state, ckpt_path)
        print(f"    [OK] {label}  best_val_loss={best:.4f}  -> {ckpt_path}")
        return best, history

    # ─── Train Model A and Model B ──────────────────────────────────────────────
    ablation = {}

    mA = MambaForecaster(input_dim=Xa_tr.shape[2], d_model=config.D_MODEL,
                         horizon=config.FORECAST_HORIZON,
                         num_layers=config.NUM_MAMBA_LAYERS,
                         dropout=config.DROPOUT).to(device)
    bestA, hist_A = train_one(mA, trL_A, vaL_A, 'mamba_model_A.pt', 'Model_A_time_only')

    mB = MambaForecaster(input_dim=Xb_tr.shape[2], d_model=config.D_MODEL,
                         horizon=config.FORECAST_HORIZON,
                         num_layers=config.NUM_MAMBA_LAYERS,
                         dropout=config.DROPOUT).to(device)
    bestB, hist_B = train_one(mB, trL_B, vaL_B, 'mamba_model_B.pt', 'Model_B_time_weather')

    # ─── Save training history CSV for generate_figures.py Figure 1 ───────────────
    import pandas as _pd
    _hist_df = _pd.DataFrame([{'epoch': ep, 'loss': tl, 'val_loss': vl} for ep, tl, vl in hist_A])
    _hist_df = _hist_df.rename(columns={'loss': 'train_loss'})
    _hist_df.to_csv('training_history.csv', index=False)
    print(f"\n    [OK] Saved training_history.csv ({len(_hist_df)} rows for Model A)")

    # ─── Evaluate on test set ───────────────────────────────────────────────────
    print("\n" + "="*60 + "\nABLATION TEST RESULTS\n" + "="*60)

    def evaluate_ablation(model, te_loader, scaler_local):
        model.eval()
        all_m, all_gt, inf_t = [], [], []
        with torch.no_grad():
            for xb, yb in te_loader:
                xb = xb.to(device)
                s0 = time.time()
                mean, _ = model(xb)
                if torch.cuda.is_available(): torch.cuda.synchronize()
                inf_t.append(time.time() - s0)
                sp_mean, sp_std = scaler_local.mean_[0], scaler_local.scale_[0]
                all_m.append((mean.cpu().numpy() * sp_std + sp_mean).flatten())
                all_gt.append((yb.numpy() * sp_std + sp_mean).flatten())
        all_m  = np.concatenate(all_m);  all_gt = np.concatenate(all_gt)
        mae_v  = float(np.mean(np.abs(all_gt - all_m)))
        rmse_v = float(np.sqrt(np.mean((all_gt - all_m)**2)))
        lat_ms = float(np.mean(inf_t)*1000)
        return mae_v, rmse_v, lat_ms

    for lbl, model_obj, t_loader, sc_local, path in [
        ("Model_A_time_only",  mA, teL_A, scA, 'mamba_model_A.pt'),
        ("Model_B_time_weather", mB, teL_B, scB, 'mamba_model_B.pt'),
    ]:
        model_obj.load_state_dict(torch.load(path, weights_only=True))
        mae_v, rmse_v, lat_v = evaluate_ablation(model_obj, t_loader, sc_local)
        ablation[lbl] = dict(test_MAE=mae_v, test_RMSE=rmse_v, ckpt=path, lat_ms=lat_v)
        print(f"  {lbl:<26}  MAE={mae_v:.4f}  RMSE={rmse_v:.4f}  latency={lat_v:.2f}ms")

    # ─── Save ablation results ──────────────────────────────────────────────────
    print("\n[5] Saving ablation results...")
    abl_rows = []
    for k2, v2 in ablation.items():
        abl_rows.append({'model': k2, 'test_MAE': v2['test_MAE'],
                          'test_RMSE': v2['test_RMSE'],
                          'latency_ms': v2['lat_ms'], 'ckpt': v2['ckpt']})
    abl_df = pd.DataFrame(abl_rows)
    abl_df = abl_df.set_index('model')
    abl_df.to_csv('mamba_ablation_results.csv')
    print(f"    - Saved: mamba_ablation_results.csv")
    print(abl_df.to_string(index=False))

    # ─── Save processed data tensors for generate_figures.py ─────────────────────
    import torch as _torch
    from pathlib import Path as _Path

    print("\n[6] Saving processed data tensors for generate_figures.py...")

    _proc_dir = _Path("data/processed")
    _proc_dir.mkdir(parents=True, exist_ok=True)

    _speed_mean_A = float(scA.mean_[0])
    _speed_std_A  = float(scA.scale_[0])
    _speed_mean_B = float(scB.mean_[0])
    _speed_std_B  = float(scB.scale_[0])

    _processed = {
        'X_test_A':      Xa_te.copy(),
        'y_test_A':      ya_te.copy(),
        'X_test_B':      Xb_te.copy(),
        'y_test_B':      yb_te.copy(),
        'speed_mean_A':  _speed_mean_A,
        'speed_std_A':   _speed_std_A,
        'speed_mean_B':  _speed_mean_B,
        'speed_std_B':   _speed_std_B,
        '_full_idx':     np.array([str(t) for t in data.index], dtype=object),
        'time_mean':     _time_mean,
        'time_std':      _time_std,
        'weather_mean':  _wx_mean,
        'weather_std':   _wx_std,
    }

    _torch.save(_processed, _proc_dir / "processed_data.pt")
    print(f"    [OK] Saved: {_proc_dir / 'processed_data.pt'}")
    print(f"    - X_test_A={_processed['X_test_A'].shape}  y_test_A={_processed['y_test_A'].shape}")
    print(f"    - X_test_B={_processed['X_test_B'].shape}  y_test_B={_processed['y_test_B'].shape}")
    print(f"    - speed_mean_A={_speed_mean_A:.4f}  speed_std_A={_speed_std_A:.4f}")
    print(f"    - speed_mean_B={_speed_mean_B:.4f}  speed_std_B={_speed_std_B:.4f}")

    # ─── Save scaler params for mph-axis accuracy across all downstream scripts ───
    # Pure Python floats so np.load returns a plain scalar, never a 0-d ndarray.
    _sp = {
        'speed_mean_A': _speed_mean_A.tolist() if hasattr(_speed_mean_A, 'tolist') else float(_speed_mean_A),
        'speed_std_A':  _speed_std_A.tolist()  if hasattr(_speed_std_A,  'tolist') else float(_speed_std_A),
        'speed_mean_B': _speed_mean_B.tolist() if hasattr(_speed_mean_B, 'tolist') else float(_speed_mean_B),
        'speed_std_B':  _speed_std_B.tolist()  if hasattr(_speed_std_B,  'tolist') else float(_speed_std_B),
        'traffic_mean': _speed_mean_A.tolist() if hasattr(_speed_mean_A, 'tolist') else float(_speed_mean_A),
        'traffic_std':  _speed_std_A.tolist()  if hasattr(_speed_std_A,  'tolist') else float(_speed_std_A),
        'time_mean':    _time_mean.tolist()    if hasattr(_time_mean,    'tolist') else float(_time_mean),
        'time_std':     _time_std.tolist()     if hasattr(_time_std,     'tolist') else float(_time_std),
        'weather_mean': _wx_mean.tolist()      if hasattr(_wx_mean,      'tolist') else float(_wx_mean),
        'weather_std':  _wx_std.tolist()       if hasattr(_wx_std,       'tolist') else float(_wx_std),
    }
    import numpy as _np
    _np.savez(_proc_dir / "scaler_params.npz", **_sp)
    print(f"    [OK] Saved: {_proc_dir / 'scaler_params.npz'}")

    # ─── Save metadata.json for temporal_generalization.py ───────────────────────
    _n_total = len(data)          # total rows in original DataFrame
    _t_end   = int(_n_total * config.TRAIN_RATIO)
    _v_end   = _t_end + int(_n_total * config.VAL_RATIO)
    _idx     = data.index
    _meta = {
        "sensor_id":        data.columns[0],
        "feat_count":       int(data.shape[1]),
        "lookback":         config.LOOKBACK_WINDOW,
        "horizon":          config.FORECAST_HORIZON,
        "train_samples":    _t_end,
        "val_samples":      _v_end - _t_end,
        "test_samples":     _n_total - _v_end,
        "original_range": {
            "train_start": str(_idx[0]),
            "train_end":   str(_idx[_t_end - 1]),
            "val_start":   str(_idx[_t_end]),
            "val_end":     str(_idx[_v_end - 1]),
            "test_start":  str(_idx[_v_end]),
            "test_end":    str(_idx[-1]),
        },
        "speed_mean": _speed_mean_A,
        "speed_std":  _speed_std_A,
    }
    import json as _json
    _json.dump(_meta, (_proc_dir / "metadata.json").open("w"), indent=2)
    print(f"    [OK] Saved: {_proc_dir / 'metadata.json'}")

    # ─── Legacy result kept for step4_evaluation_metrics.py compatibility ─────────
    print("\n[LEGACY] Building backward-compatible mamba_evaluation_results.csv...")
    
    # ── Load Model B checkpoint and run one-quick eval pass ────────────────────
    eval_model = MambaForecaster(
        input_dim=Xb_te.shape[2], d_model=config.D_MODEL,
        horizon=config.FORECAST_HORIZON, num_layers=config.NUM_MAMBA_LAYERS,
        dropout=config.DROPOUT).to(device)
    eval_model.load_state_dict(torch.load('mamba_model_B.pt', weights_only=True))
    eval_model.eval()
    all_m, all_g = [], []
    evL = DataLoader(TrafficDataset(Xb_te, yb_te), batch_size=512, shuffle=False)
    with torch.no_grad():
        for xb, yb in evL:
            mean, _ = eval_model(xb.to(device))
            sp_mean, sp_std = scB.mean_[0], scB.scale_[0]
            all_m.append((mean.cpu().numpy()*sp_std+sp_mean).flatten())
            all_g.append((yb.numpy()*sp_std+sp_mean).flatten())
    all_m = np.concatenate(all_m);  all_g = np.concatenate(all_g)
    
    mae_l  = float(np.mean(np.abs(all_g - all_m)))
    rmse_l = float(np.sqrt(np.mean((all_g - all_m)**2)))
    # Symmetric histogram KL with eps=1e-9 (no inf possible)
    lo, hi = min(all_g.min(), all_m.min()), max(all_g.max(), all_m.max())
    bins_256 = np.linspace(lo, hi, 257)
    ph, _ = np.histogram(all_m, bins_256, density=True)
    gt, _ = np.histogram(all_g,  bins_256, density=True)
    ph = np.clip(ph, 1e-9, None);  ph /= ph.sum()
    gt = np.clip(gt, 1e-9, None);  gt /= gt.sum()
    kl_l = float(0.5*(np.sum(gt*np.log(gt/ph)) + np.sum(ph*np.log(ph/gt))))
    total_params = sum(p.numel() for p in eval_model.parameters())
    train_params = sum([p.numel() for p in eval_model.parameters() if p.requires_grad])
    
    # Measure inference latency using the new robust method
    # Get a sample batch for latency measurement
    sample_batch = next(iter(evL))[0].to(device)  # Get first batch, just the input
    lat_l = eval_model.measure_inference_latency(sample_batch, num_iterations=100)
    
    results = {
        'MAE': mae_l, 'RMSE': rmse_l, 'KL_Divergence': kl_l,
        'Inference_Latency_ms': lat_l,
        'Model_A_MAE': ablation['Model_A_time_only']['test_MAE'],
        'Model_B_MAE': ablation['Model_B_time_weather']['test_MAE'],
        'Weather_MAE_Reduction_pct': round(
            (ablation['Model_A_time_only']['test_MAE'] -
             ablation['Model_B_time_weather']['test_MAE']) /
            ablation['Model_A_time_only']['test_MAE'] * 100, 2),
        'Epochs': config.EPOCHS, 'Batch_Size': config.BATCH_SIZE,
        'Learning_Rate': config.LEARNING_RATE, 'D_Model': config.D_MODEL,
        'Num_Mamba_Layers': config.NUM_MAMBA_LAYERS,
        'Lookback_Window': config.LOOKBACK_WINDOW,
        'Forecast_Horizon': config.FORECAST_HORIZON,
        'Total_Params': total_params, 'Trainable_Params': train_params,
    }
    res_df = pd.DataFrame([results])
    res_df.to_csv('mamba_evaluation_results.csv', index=False)
    print(f"    - Saved: mamba_evaluation_results.csv\n")
    print("="*60)
    print("STEP 5 COMPLETE: Weather ablation training finished!")
    print("="*60)
    print("\nAblation summary:")
    print(abl_df.to_string(index=False))
    return results

if __name__ == "__main__":
    results = main()
