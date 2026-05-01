"""
Step 3: Chronos-2 Zero-Shot Inference Pipeline
==============================================
This script runs zero-shot inference using Chronos-2 foundation model
for urban traffic forecasting.

Chronos-2 is a pre-trained time-series foundation model that can make
predictions without task-specific training (zero-shot).

Features:
- Loads Chronos-2 model from HuggingFace
- Prepares METR-LA traffic data
- Runs forecasting with configurable context/prediction lengths
- Outputs probabilistic predictions for KL Divergence evaluation

Author: Suvarna Kotha & Ruthik Garapati
Thesis: Urban Traffic Forecasting - Comparative Analysis
"""

import pandas as pd
import numpy as np
import torch
import time
import os

print("=" * 60)
print("STEP 3: Chronos-2 Zero-Shot Inference")
print("=" * 60)

# ============================================================================
# 3.1 Configuration
# ============================================================================
print("\n[3.1] Configuration...")

# Chronos-2 model parameters
MODEL_NAME = "amazon/chronos-t5-small"  # Options: small, base, large
PREDICTION_LENGTH = 12  # Predict next 12 timesteps (1 hour at 5-min intervals)
CONTEXT_LENGTH = 144   # Use past 144 timesteps (12 hours of history)
NUM_SAMPLES = 100       # Number of samples for probabilistic forecasting

print(f"   - Model: {MODEL_NAME}")
print(f"   - Prediction length: {PREDICTION_LENGTH} timesteps (1 hour)")
print(f"   - Context length: {CONTEXT_LENGTH} timesteps (12 hours)")
print(f"   - Num samples: {NUM_SAMPLES}")

# Check for GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"   - Device: {device}")

# ============================================================================
# 3.2 Load Preprocessed Data
# ============================================================================
print("\n[3.2] Loading preprocessed data...")

try:
    df = pd.read_csv('single_sensor_with_weather.csv', index_col=0)
    df.index = pd.to_datetime(df.index)
    print(f"   - Loaded single sensor data: {df.shape}")
    print(f"   - Date range: {df.index.min()} to {df.index.max()}")
except FileNotFoundError:
    print("   ERROR: single_sensor_with_weather.csv not found!")
    print("   Please run step2_data_preprocessing.py first.")
    exit(1)

# ============================================================================
# 3.3 Install/Import Chronos
# ============================================================================
print("\n[3.3] Loading Chronos-2 model...")

try:
    # Try the correct import
    from chronos import ChronosPipeline
except ImportError:
    print("   Installing chronos-forecasting package...")
    os.system("pip install chronos-forecasting -q")
    from chronos import ChronosPipeline

# Load the Chronos-2 model
print(f"   Loading {MODEL_NAME}...")
start_time = time.time()

pipeline = ChronosPipeline.from_pretrained(
    MODEL_NAME,
    device_map=device,
)

load_time = time.time() - start_time
print(f"   Model loaded in {load_time:.2f} seconds")

# ============================================================================
# 3.4 Prepare Data for Forecasting
# ============================================================================
print("\n[3.4] Preparing data for forecasting...")

# Extract traffic speed time series
traffic_series = df['traffic_speed'].values

print(f"   - Total timesteps: {len(traffic_series)}")
print(f"   - Traffic speed range: {traffic_series.min():.2f} - {traffic_series.max():.2f} mph")

# Calculate how many forecast windows we can generate
total_needed = CONTEXT_LENGTH + PREDICTION_LENGTH
num_forecasts = (len(traffic_series) - total_needed) // PREDICTION_LENGTH

print(f"   - Can generate {num_forecasts} forecast windows")
print(f"   - Using last {total_needed} timesteps for demonstration")

# ============================================================================
# 3.5 Run Zero-Shot Forecasting
# ============================================================================
print("\n[3.5] Running zero-shot forecasting...")

# Get the context (last CONTEXT_LENGTH timesteps before the test period)
context = traffic_series[:-PREDICTION_LENGTH][-CONTEXT_LENGTH:]
print(f"   - Context shape: {context.shape}")
print(f"   - Context period: {df.index[-CONTEXT_LENGTH-PREDICTION_LENGTH]} to {df.index[-PREDICTION_LENGTH-1]}")

# Run prediction
print(f"   Running inference (this may take a minute)...")
start_inference = time.time()

# Chronos expects context as a tensor - use the forecast method
forecast = pipeline.predict(
    inputs=torch.tensor(context, dtype=torch.float32).unsqueeze(0),
    prediction_length=PREDICTION_LENGTH,
    num_samples=NUM_SAMPLES
)

inference_time = time.time() - start_inference
print(f"   - Inference time: {inference_time:.2f} seconds")

# Output shape: (batch_size, num_samples, prediction_length) = (1, 100, 12)
# Need to squeeze batch dimension first
forecast = forecast.squeeze(0)  # Now (100, 12)
print(f"   - Forecast shape: {forecast.shape}")

# ============================================================================
# 3.6 Extract Predictions
# ============================================================================
print("\n[3.6] Extracting predictions...")

# Get actual values (ground truth)
actual = traffic_series[-PREDICTION_LENGTH:]
print(f"   - Actual values shape: {actual.shape}")

# Calculate statistics from forecast samples
# forecast shape is now (num_samples, prediction_length) = (100, 12)
forecast_np = forecast.numpy()  # Convert to numpy for easier handling
forecast_mean = forecast_np.mean(axis=0)  # Mean across samples (axis=0)
forecast_std = forecast_np.std(axis=0)    # Std across samples (axis=0)
forecast_median = np.median(forecast_np, axis=0)

print(f"   - Predicted mean range: {forecast_mean.min():.2f} - {forecast_mean.max():.2f} mph")
print(f"   - Predicted std range: {forecast_std.min():.2f} - {forecast_std.max():.2f} mph")

# Get actual timestamps
actual_timestamps = df.index[-PREDICTION_LENGTH:]

# ============================================================================
# 3.7 Save Predictions
# ============================================================================
print("\n[3.7] Saving predictions...")

# Create predictions DataFrame
predictions_df = pd.DataFrame({
    'timestamp': actual_timestamps,
    'actual': actual,
    'predicted_mean': forecast_mean,
    'predicted_median': forecast_median,
    'predicted_std': forecast_std
})

# Add individual samples for KL divergence calculation
for i in range(min(NUM_SAMPLES, 10)):  # Save first 10 samples
    predictions_df[f'sample_{i}'] = forecast[i].numpy()

# Save to CSV
output_file = 'chronos_predictions.csv'
predictions_df.to_csv(output_file, index=False)
print(f"   - Saved to: {output_file}")

# Print sample predictions
print("\n   Sample Predictions (first 6 timesteps):")
print("   " + "-" * 50)
print(f"   {'Time':<20} {'Actual':>8} {'Mean':>8} {'Std':>8}")
print("   " + "-" * 50)
for i in range(min(6, PREDICTION_LENGTH)):
    print(f"   {str(actual_timestamps[i]):<20} {actual[i]:>8.2f} {forecast_mean[i]:>8.2f} {forecast_std[i]:>8.2f}")

# ============================================================================
# 3.8 Calculate Basic Metrics
# ============================================================================
print("\n[3.8] Basic Evaluation Metrics...")

# MAE
mae = np.mean(np.abs(actual - forecast_mean))
print(f"   - MAE: {mae:.4f} mph")

# RMSE
rmse = np.sqrt(np.mean((actual - forecast_mean) ** 2))
print(f"   - RMSE: {rmse:.4f} mph")

# Mean Absolute Percentage Error (MAPE)
mape = np.mean(np.abs((actual - forecast_mean) / (actual + 1e-8))) * 100
print(f"   - MAPE: {mape:.2f}%")

# ============================================================================
# 3.9 Save Model Info
# ============================================================================
print("\n[3.9] Saving model configuration...")

model_info = {
    'model_name': MODEL_NAME,
    'prediction_length': PREDICTION_LENGTH,
    'context_length': CONTEXT_LENGTH,
    'num_samples': NUM_SAMPLES,
    'device': device,
    'model_load_time': load_time,
    'inference_time': inference_time,
    'mae': mae,
    'rmse': rmse
}

# Save as text file
with open('chronos_model_info.txt', 'w') as f:
    for key, value in model_info.items():
        f.write(f"{key}: {value}\n")

print("   - Saved to: chronos_model_info.txt")

print("\n" + "=" * 60)
print("STEP 3 COMPLETE: Chronos-2 inference finished!")
print("=" * 60)
print("\nNext step:")
print("  - Run step4_evaluation_metrics.py for detailed evaluation")
print("    including KL Divergence analysis")
