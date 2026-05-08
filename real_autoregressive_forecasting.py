"""
Real Autoregressive Forecasting for ONE Month (2012 vs 2013)
================================================================

This script performs REAL 2013 traffic predictions using actual 2013 weather data
via an Autoregressive (Rolling) Forecasting Loop for a SINGLE month.

SELECT ONE MONTH that exists in BOTH:
  - 2012 traffic dataset (March-June available)
  - 2013 weather dataset 72295.csv (all 12 months available)

For example: May 2012 (standard) vs May 2013 (autoregressive with real 2013 weather)

This replaces the old proxy simulation approach.

Author: Suvarna Kotha & Ruthik Garapati
Thesis: Urban Traffic Forecasting - Comparative Analysis
"""

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import time
import os

# Use non-interactive backend for matplotlib (no display needed)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

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
    # Data paths
    TRAFFIC_DATA_PATH = 'METR-LA_cleaned.csv'  # 2012 traffic data
    WEATHER_2013_PATH = '72295.csv'             # 2013 weather data
    MODEL_PATH = 'mamba_best_model.pt'          # Pre-trained model

    # Window sizes
    LOOKBACK_WINDOW = 24   # 2 hours (5-min intervals)
    FORECAST_HORIZON = 12  # 1 hour ahead

    # Features: speed + prcp + wspd + hour_sin/cos + day_sin/cos = 7 total
    # (Week/month encodings omitted to match saved model checkpoint)
    INPUT_DIM = 7
    D_MODEL = 64
    NUM_MAMBA_LAYERS = 2
    DROPOUT = 0.1

    # SELECT ONE MONTH HERE (must be 3,4,5,6 for 2012 data)
    PREDICT_MONTH = 5  # May (change to 3=Mar, 4=Apr, 6=Jun as needed)

    # Device
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    SEED = 42

config = Config()
torch.manual_seed(config.SEED)
np.random.seed(config.SEED)

# =============================================================================
# Temporal Feature Extraction
# =============================================================================
def extract_temporal_features(df):
    """Extract cyclical temporal features from timestamps."""
    hours = df.index.hour
    days = df.index.dayofweek
    weeks = df.index.isocalendar().week
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
# PHASE 1: Load 2013 Weather Data (72295.csv)
# =============================================================================
def load_2013_weather_data(filepath, month):
    """
    Load 2013 weather from 72295.csv for the specified month.
    Returns DataFrame with prcp, wspd, and 8 temporal encodings at 5-min frequency.
    """
    print("=" * 60)
    print("PHASE 1: Load 2013 Weather Data")
    print("=" * 60)
    month_name = pd.Timestamp(2013, month, 1).strftime('%B')

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Missing: {filepath}")

    df = pd.read_csv(filepath)
    print(f"\n[1.1] Loaded: {df.shape[0]} rows × {df.shape[1]} cols")

    # Parse datetime from year, month, day, hour
    df['datetime'] = pd.to_datetime(df[['year', 'month', 'day', 'hour']])
    df = df.set_index('datetime')

    # Identify prcp and wspd
    prcp_col = [c for c in df.columns if 'prcp' in c.lower()][0]
    wspd_col = [c for c in df.columns if 'wspd' in c.lower()][0]

    # Fill missing prcp with 0.0
    df[prcp_col] = df[prcp_col].fillna(0.0)
    df[wspd_col] = df[wspd_col].fillna(df[wspd_col].median())

    # Resample hourly -> 5-minute (forward fill)
    print(f"[1.2] Resampling hourly -> 5-minute intervals...")
    df_5min = df.resample('5T').ffill()

    # Extract target month
    df_month = df_5min[(df_5min.index.month == month) & (df_5min.index.year == 2013)].copy()
    print(f"[1.3] {month_name} 2013: {len(df_month)} timesteps")
    print(f"       Range: {df_month.index.min()} to {df_month.index.max()}")

    # Generate temporal encodings (only hour & day to match model input_dim=7)
    h_s, h_c, d_s, d_c, w_s, w_c, m_s, m_c = extract_temporal_features(df_month)
    weather_features = pd.DataFrame({
        'precipitation_mm': df_month[prcp_col].values,
        'wind_speed_kmh': df_month[wspd_col].values,
        'hour_sin': h_s, 'hour_cos': h_c,
        'day_sin': d_s, 'day_cos': d_c,
        # week/month omitted to match 7-D input
    }, index=df_month.index)

    print(f"[1.4] Weather features ready: {weather_features.shape} (6 cols + speed later)")
    return weather_features

# =============================================================================
# Load 2012 Traffic + Weather Data
# =============================================================================
def load_2012_traffic_data():
    """
    Load the single-sensor merged dataset created by step2_data_preprocessing.py.
    Returns a DataFrame with standardized columns (7 total):
      speed, precipitation_mm, wind_speed_kmh,
      hour_sin, hour_cos, day_sin, day_cos
    (week/month encodings omitted to match model input_dim=7)
    """
    print("\n" + "=" * 60)
    print("PHASE 2: Load 2012 Traffic + Weather")
    print("=" * 60)

    merged_file = 'METR_LA_with_Weather_5min.csv'
    if not os.path.exists(merged_file):
        raise FileNotFoundError(
            f"Required file not found: {merged_file}\n"
            f"Please run step2_data_preprocessing.py first to generate it."
        )

    print(f"\n[2.1] Loading merged single-sensor data...")
    df = pd.read_csv(merged_file, index_col=0)
    df.index = pd.to_datetime(df.index)
    print(f"   - Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"   - Range: {df.index.min()} -> {df.index.max()}")
    print(f"   - Columns: {list(df.columns)}")

    # Identify traffic column (should be 'traffic_speed' from step2, but be robust)
    if 'traffic_speed' in df.columns:
        speed_series = df['traffic_speed']
    else:
        # Fallback: first column that is not weather-prefixed
        traffic_cols = [c for c in df.columns if not c.startswith('weather_')]
        if not traffic_cols:
            raise ValueError("No traffic speed column found in merged file")
        speed_series = df[traffic_cols[0]]
        print(f"   - Using traffic column: '{traffic_cols[0]}'")

    # Identify weather columns
    weather_cols = [c for c in df.columns if c.startswith('weather_')]
    precip_col = [c for c in weather_cols if 'precip' in c.lower()]
    wind_col = [c for c in weather_cols if 'wind' in c.lower()]
    if not precip_col or not wind_col:
        raise ValueError(f"Weather columns missing. Found: {weather_cols}")
    precip_col = precip_col[0]
    wind_col = wind_col[0]

    print(f"[2.2] Column mapping:")
    print(f"   - Traffic: '{speed_series.name}' -> 'speed'")
    print(f"   - Precipitation: '{precip_col}' -> 'precipitation_mm'")
    print(f"   - Wind: '{wind_col}' -> 'wind_speed_kmh'")

    # Build standardized feature DataFrame
    data = pd.DataFrame({
        'speed': speed_series.values,
        'precipitation_mm': df[precip_col].values,
        'wind_speed_kmh': df[wind_col].values,
    }, index=df.index)

    # Add 4 cyclical temporal features (hour + day only) -> total 7 features
    print(f"\n[2.3] Generating hour+day cyclical temporal features...")
    h_s, h_c, d_s, d_c, w_s, w_c, m_s, m_c = extract_temporal_features(df)
    data['hour_sin'] = h_s; data['hour_cos'] = h_c
    data['day_sin'] = d_s; data['day_cos'] = d_c
    # week_sin/week_cos and month_sin/month_cos intentionally omitted (model expects 7-D input)

    data = data.ffill().bfill()
    print(f"\n[2.4] Final 2012 dataset ready:")
    print(f"   - Shape: {data.shape[0]} rows × {data.shape[1]} columns")
    print(f"   - Columns: {list(data.columns)}")
    print(f"   - Date range: {data.index.min()} -> {data.index.max()}")
    return data

# =============================================================================
# Model Definition
# =============================================================================
class MambaForecaster(nn.Module):
    def __init__(self, input_dim=config.INPUT_DIM, d_model=config.D_MODEL,
                 horizon=config.FORECAST_HORIZON, num_layers=config.NUM_MAMBA_LAYERS,
                 dropout=config.DROPOUT):
        super().__init__()
        self.input_projection = nn.Linear(input_dim, d_model)
        self.dropout = nn.Dropout(dropout)
        self.d_model = d_model
        self.horizon = horizon
        self.num_layers = num_layers

        if MAMBA_AVAILABLE:
            from mamba_ssm import Mamba as MambaBlock
            self.layers = nn.ModuleList([MambaBlock(d_model=d_model) for _ in range(num_layers)])
        else:
            self.layers = nn.ModuleList([
                nn.Sequential(nn.Linear(d_model, d_model * 4), nn.GELU(),
                              nn.Linear(d_model * 4, d_model), nn.Dropout(dropout))
                for _ in range(num_layers)
            ])

        self.layer_norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(num_layers)])
        self.output_head = nn.Linear(d_model, horizon * 2)

    def forward(self, x):
        B, L, _ = x.shape
        x = self.input_projection(x)
        for i in range(len(self.layers)):
            residual = x
            x = self.layers[i](x)
            x = self.dropout(x) + residual
            x = self.layer_norms[i](x)
        last = x[:, -1, :]
        out = self.output_head(last).view(B, self.horizon, 2)
        mean = out[:, :, 0]
        log_std = torch.clamp(out[:, :, 1], min=-10, max=2)
        return mean, log_std

# =============================================================================
# PHASE 2: Standard Inference on 2012 Month
# =============================================================================
def standard_inference_2012(model, data_2012, month, device):
    """Supervised evaluation on a 2012 month."""
    month_name = pd.Timestamp(2012, month, 1).strftime('%B')
    print(f"\n{'='*60}")
    print(f"STANDARD INFERENCE: {month_name} 2012")
    print(f"{'='*60}")

    # Split: train on earlier months in 2012, test on the target month
    test_data = data_2012[(data_2012.index.month == month) & (data_2012.index.year == 2012)]
    if test_data.empty:
        raise ValueError(f"No 2012 data for month {month}. Check range: {data_2012.index.min()} to {data_2012.index.max()}")

    train_data = data_2012[data_2012.index < test_data.index.min()]  # all data before test month

    if train_data.empty:
        raise ValueError(f"No training data for month {month} (no data before {test_data.index.min()}). Use month >= 4.")

    min_required = config.LOOKBACK_WINDOW + config.FORECAST_HORIZON
    if len(test_data) < min_required:
        raise ValueError(f"Test month {month} too short: need {min_required} timesteps, got {len(test_data)}")

    print(f"\n[2.1] Split:")
    print(f"    Train: {train_data.index.min()} -> {train_data.index.max()} ({len(train_data)} rows)")
    print(f"    Test:  {test_data.index.min()} -> {test_data.index.max()} ({len(test_data)} rows)")

    # Fit StandardScaler on TRAIN only (no data leakage)
    scaler = StandardScaler()
    scaler.fit(train_data.values)
    mu_speed = scaler.mean_[0]
    sigma_speed = scaler.scale_[0]
    print(f"    - Scaler fitted on train (speed mu={mu_speed:.2f}, sigma={sigma_speed:.2f})")

    # Scale entire test set in one go using training statistics
    test_array = test_data.values  # (n_test, n_features)
    test_scaled = scaler.transform(test_array)  # (n_test, 7)

    # Build sliding windows from scaled test data
    L, H = config.LOOKBACK_WINDOW, config.FORECAST_HORIZON
    X_list, y_list, ts_list = [], [], []
    for i in range(len(test_scaled) - L - H + 1):
        X_list.append(test_scaled[i:i+L])
        y_list.append(test_scaled[i+L:i+L+H, 0])  # speed column (scaled)
        ts_list.append(test_data.index[i+L])
    X = np.array(X_list, dtype=np.float32)   # shape (n_windows, L, 7)
    y_scaled = np.array(y_list, dtype=np.float32)  # shape (n_windows, H)

    print(f"\n[2.2] Windows: {len(X)}  |  X:{X.shape}  y:{y_scaled.shape}")

    # Model inference
    model.eval()
    preds_scaled = []
    with torch.no_grad():
        for i in range(0, len(X), 64):
            batch = torch.tensor(X[i:i+64], dtype=torch.float32).to(device)
            m, _ = model(batch)
            preds_scaled.append(m.cpu().numpy())
    pred_scaled = np.concatenate(preds_scaled, axis=0)  # (n_windows, H)

    # Inverse transform to original mph
    pred_orig = pred_scaled * sigma_speed + mu_speed
    y_orig = y_scaled * sigma_speed + mu_speed

    # Metrics
    mae = np.mean(np.abs(y_orig - pred_orig))
    rmse = np.sqrt(np.mean((y_orig - pred_orig)**2))
    print(f"\n[2.3] Results: MAE={mae:.4f} mph, RMSE={rmse:.4f} mph")
    print(f"    Windows: {len(pred_orig)}, Horizon: {H} steps each")

    return ts_list, pred_orig, y_orig, scaler, mae, rmse

# =============================================================================
# PHASE 3: Autoregressive Rolling Forecast for 2013 Month
# =============================================================================
def autoregressive_forecast_2013(model, data_2012, weather_2013, month, scaler, device):
    """
    Autoregressive rolling forecast for the specified 2013 month.

    H₀ = last 24 traffic speeds from 2012 data (June 27, 2012)
    For each t ∈ target_month_2013:
      X_t = [H_t, W_t]  where W_t = (prcp_t, wspd_t, 8 temporal_t)
      ŷ_{t+1} = f(X_t)
      H_{t+1} = [H_t[1:], ŷ_{t+1}]
    """
    month_name = pd.Timestamp(2013, month, 1).strftime('%B')
    print(f"\n{'='*60}")
    print(f"AUTOREGRESSIVE FORECAST: {month_name} 2013 (REAL WEATHER)")
    print(f"{'='*60}")

    # Seed window H₀: last 24 traffic speeds from 2012
    H_t = data_2012['speed'].iloc[-config.LOOKBACK_WINDOW:].values
    print(f"\n[3.1] Seed window H₀:")
    print(f"    Source: {data_2012.index.max()}")
    print(f"    Values: mean={H_t.mean():.1f}, range=[{H_t.min():.1f}, {H_t.max():.1f}] mph")

    # Target month weather
    w_month = weather_2013[(weather_2013.index.month == month) &
                           (weather_2013.index.year == 2013)].copy()
    print(f"\n[3.2] Target: {len(w_month)} timesteps")
    print(f"    Range: {w_month.index.min()} -> {w_month.index.max()}")

    # Rolling loop
    model.eval()
    predictions = []
    start_time = time.time()
    log_int = max(50, len(w_month) // 20)

    with torch.no_grad():
        for idx, (ts, w_row) in enumerate(w_month.iterrows()):
            # W_t: 10 features
            W_t = w_row.values.astype(np.float32)

            # Build X_t: L×11 matrix
            X_t = np.zeros((config.LOOKBACK_WINDOW, config.INPUT_DIM), dtype=np.float32)
            X_t[:, 0] = H_t               # traffic history
            X_t[:, 1:] = W_t              # weather + temporal (broadcast)

            # Scale & forward
            X_t_s = scaler.transform(X_t)
            x_tensor = torch.tensor(X_t_s, dtype=torch.float32).unsqueeze(0).to(device)
            mean_pred, _ = model(x_tensor)
            y_hat_s = mean_pred[0, 0].item()

            # Inverse scale
            y_hat = y_hat_s * scaler.scale_[0] + scaler.mean_[0]

            predictions.append({
                'timestamp': ts,
                'predicted_mean': y_hat,
                'weather_precip': w_row['precipitation_mm'],
                'weather_wind': w_row['wind_speed_kmh'],
            })

            # Update H_t ← [H_t[1:], ŷ]
            H_t = np.roll(H_t, -1)
            H_t[-1] = y_hat

            if (idx+1) % log_int == 0 or idx == len(w_month)-1:
                pct = (idx+1)/len(w_month)*100
                print(f"  [{pct:5.1f}%] {idx+1}/{len(w_month)} | ŷ={y_hat:6.2f} mph | "
                      f"H=[{H_t[0]:.1f}...{H_t[-1]:.1f}] | t={time.time()-start_time:.1f}s")

    pred_df = pd.DataFrame(predictions).set_index('timestamp')
    print(f"\n[3.3] Complete: {len(pred_df)} predictions")
    print(f"    Mean speed: {pred_df['predicted_mean'].mean():.2f} mph")
    return pred_df

# =============================================================================
# PHASE 4: Save + Plot
# =============================================================================
def save_and_plot(month, ts_2012, pred_2012, actual_2012, mae, rmse, pred_2013):
    """Save CSVs and generate comparison figures."""
    month_name = pd.Timestamp(2012, month, 1).strftime('%B')

    print("\n" + "=" * 60)
    print("PHASE 4: Save Outputs & Generate Figures")
    print("=" * 60)

    # Expand 2012 predictions (horizon=12 -> individual timesteps)
    rows_2012 = []
    for i, ts in enumerate(ts_2012):
        for h in range(12):
            rows_2012.append({
                'timestamp': ts + pd.Timedelta(minutes=5*(h+1)),
                'month': month_name,
                'predicted_mean': pred_2012[i, h],
                'actual_speed': actual_2012[i, h],
                'dataset': '2012_Standard'
            })
    df2012 = pd.DataFrame(rows_2012)
    df2012.to_csv('autoregressive_predictions_2012_standard.csv', index=False)
    print(f"\n[4.1] Saved: autoregressive_predictions_2012_standard.csv ({len(df2012)} rows)")

    # 2013 rolling predictions
    df2013 = pred_2013.reset_index()
    df2013['month'] = month_name
    df2013['dataset'] = '2013_Rolling_RealWeather'
    df2013.to_csv('autoregressive_predictions_2013_rolling.csv', index=False)
    print(f"[4.2] Saved: autoregressive_predictions_2013_rolling.csv ({len(df2013)} rows)")

    # Summary
    summary = pd.DataFrame([{
        'Month': month_name,
        'Year': 2012,
        'Method': 'Standard',
        'MAE': f"{mae:.4f}",
        'RMSE': f"{rmse:.4f}",
        'Mean_Pred_Speed': f"{pred_2012.mean():.2f}",
        'Mean_Actual_Speed': f"{actual_2012.mean():.2f}",
    }, {
        'Month': month_name,
        'Year': 2013,
        'Method': 'Autoregressive_Rolling',
        'MAE': 'N/A (no ground truth)',
        'RMSE': 'N/A (no ground truth)',
        'Mean_Pred_Speed': f"{pred_2013['predicted_mean'].mean():.2f}",
        'Mean_Actual_Speed': 'N/A',
    }])
    summary.to_csv('forecasting_summary.csv', index=False)
    print(f"[4.3] Saved: forecasting_summary.csv")
    print(f"\n{summary.to_string(index=False)}")

    # ============ FIGURES ============
    print(f"\n[4.4] Generating figures...")

    # Figure 1: Time series (first 7 days)
    fig, ax = plt.subplots(figsize=(14, 6))
    week_slice = slice(0, 7*24*12)  # 7 days
    ts_week = ts_2012[:7*24*12]
    pred_2012_week = pred_2012[:7*24*12].mean(axis=1)
    actual_2012_week = actual_2012[:7*24*12].mean(axis=1)
    pred_2013_week = pred_2013.iloc[:7*24*12]

    ax.plot(ts_week, actual_2012_week, label='2012 Actual (Ground Truth)', color='green', lw=1.5)
    ax.plot(ts_week, pred_2012_week, label='2012 Standard Prediction', color='blue', ls='--', lw=1.5)
    ax.plot(pred_2013_week.index, pred_2013_week['predicted_mean'],
            label='2013 Rolling (REAL 2013 Weather)', color='red', ls=':', lw=2)
    ax.set_xlabel('Timestamp')
    ax.set_ylabel('Speed (mph)')
    ax.set_title(f'{month_name}: First Week Comparison - 2012 vs 2013')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('figure_timeseries_comparison.png', dpi=150, bbox_inches='tight')
    print("    [OK] figure_timeseries_comparison.png")

    # Figure 2: Speed distributions
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(pred_2012.flatten(), bins=50, alpha=0.6, label='2012 Standard', color='blue', density=True)
    ax.hist(pred_2013['predicted_mean'].values, bins=50, alpha=0.6, label='2013 Rolling (Real Weather)', color='orange', density=True)
    ax.set_xlabel('Speed (mph)')
    ax.set_ylabel('Density')
    ax.set_title(f'{month_name}: Predicted Speed Distributions')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('figure_distribution_comparison.png', dpi=150, bbox_inches='tight')
    print("    [OK] figure_distribution_comparison.png")

    # Figure 3: Weather overlay (precipitation)
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(pred_2013.index, pred_2013['predicted_mean'], label='2013 Predicted Speed', color='red', alpha=0.7)
    ax2 = ax.twinx()
    ax2.fill_between(pred_2013.index, 0, pred_2013['weather_precip'],
                     label='2013 Precipitation (mm)', color='blue', alpha=0.3)
    ax.set_xlabel('Timestamp')
    ax.set_ylabel('Speed (mph)', color='red')
    ax2.set_ylabel('Precipitation (mm)', color='blue')
    ax.set_title(f'{month_name} 2013: Speed Predictions vs Actual Precipitation')
    ax.legend(loc='upper left')
    ax2.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('figure_weather_overlay.png', dpi=150, bbox_inches='tight')
    print("    [OK] figure_weather_overlay.png")

# =============================================================================
# Main
# =============================================================================
def main():
    print("=" * 60)
    print("REAL AUTOREGRESSIVE FORECASTING (SINGLE MONTH)")
    print("=" * 60)
    month = config.PREDICT_MONTH
    month_name = pd.Timestamp(2012, month, 1).strftime('%B')
    print(f"\nTarget month: {month_name}")
    print(f"  - 2012: Standard inference (with real traffic)")
    print(f"  - 2013: Autoregressive (with REAL weather from 72295.csv)")
    print(f"Device: {config.DEVICE}")

    # Load model
    print("\n" + "=" * 60)
    print("Loading model...")
    model = MambaForecaster(
        input_dim=config.INPUT_DIM, d_model=config.D_MODEL,
        horizon=config.FORECAST_HORIZON, num_layers=config.NUM_MAMBA_LAYERS,
        dropout=config.DROPOUT
    ).to(config.DEVICE)
    model.load_state_dict(torch.load(config.MODEL_PATH, map_location=config.DEVICE))
    model.eval()
    print(f"  [OK] Model loaded: {config.MODEL_PATH}")

    # PHASE 1: 2013 weather
    weather_2013 = load_2013_weather_data(config.WEATHER_2013_PATH, month)

    # PHASE 2: 2012 traffic
    data_2012 = load_2012_traffic_data()

    # PHASE 2: Standard 2012 inference
    ts, pred, actual, scaler, mae, rmse = standard_inference_2012(model, data_2012, month, config.DEVICE)

    # PHASE 3: Autoregressive 2013 forecast
    pred_2013 = autoregressive_forecast_2013(model, data_2012, weather_2013, month, scaler, config.DEVICE)

    # PHASE 4: Save & plot
    save_and_plot(month, ts, pred, actual, mae, rmse, pred_2013)

    print("\n" + "=" * 60)
    print("COMPLETE")
    print("=" * 60)
    print("\nOutput files:")
    print("  1. autoregressive_predictions_2012_standard.csv")
    print("  2. autoregressive_predictions_2013_rolling.csv")
    print("  3. forecasting_summary.csv")
    print("  4. figure_timeseries_comparison.png")
    print("  5. figure_distribution_comparison.png")
    print("  6. figure_weather_overlay.png")
    print(f"\n{month_name} 2013 predictions use REAL weather from 72295.csv")
    print(f"(Not a proxy simulation)")

if __name__ == "__main__":
    main()
