e"""
Step 2: Data Preprocessing - Merge METR-LA Traffic + Weather Data
==================================================================
This script merges the METR-LA traffic data with the downloaded weather data.
It handles:
1. Loading both datasets
2. Temporal alignment (5-min traffic with hourly weather)
3. Forward-filling weather to match traffic frequency
4. Data cleaning and validation
5. Saving the merged dataset

Author: Suvarna Kotha & Ruthik Garapati
Thesis: Urban Traffic Forecasting - Comparative Analysis
"""

import pandas as pd
import numpy as np
import os

print("=" * 60)
print("STEP 2: Data Preprocessing - Merging Traffic + Weather")
print("=" * 60)

# ============================================================================
# 2.1 Load METR-LA Traffic Data
# ============================================================================
print("\n[2.1] Loading METR-LA traffic data...")

# Load METR-LA (if available in the workspace)
# Note: METR-LA typically has timestamp as index and sensor IDs as columns
try:
    df_traffic = pd.read_csv('METR-LA_cleaned.csv', index_col=0)
    df_traffic.index = pd.to_datetime(df_traffic.index)
    print(f"   - Loaded METR-LA: {df_traffic.shape[0]} rows x {df_traffic.shape[1]} sensors")
    print(f"   - Date range: {df_traffic.index.min()} to {df_traffic.index.max()}")
    print(f"   - Time frequency: 5-minute intervals")
    print(f"   - Sample columns: {list(df_traffic.columns[:5])}")
except FileNotFoundError:
    print("   WARNING: METR-LA_cleaned.csv not found!")
    print("   Please ensure METR-LA data is available.")
    df_traffic = None

# ============================================================================
# 2.2 Load Weather Data
# ============================================================================
print("\n[2.2] Loading Weather data...")

try:
    df_weather = pd.read_csv('LA_Weather_Hourly_2012_Full.csv')
    df_weather['datetime'] = pd.to_datetime(df_weather['datetime'])
    df_weather = df_weather.set_index('datetime')
    print(f"   - Loaded Weather: {df_weather.shape[0]} rows")
    print(f"   - Date range: {df_weather.index.min()} to {df_weather.index.max()}")
    print(f"   - Features: {list(df_weather.columns)}")
except FileNotFoundError:
    print("   WARNING: LA_Weather_Hourly_2012_Full.csv not found!")
    print("   Please run step1_download_weather.py first.")
    df_weather = None

# ============================================================================
# 2.3 Check if both datasets loaded successfully
# ============================================================================
if df_traffic is None or df_weather is None:
    print("\n   ERROR: Cannot proceed without both datasets!")
    print("   Please ensure both files are available.")
else:
    # ============================================================================
    # 2.4 Temporal Alignment - Forward Fill Weather
    # ============================================================================
    print("\n[2.3] Aligning temporal resolution...")
    print("   - Weather is hourly (60 min), Traffic is 5-minute")
    print("   - Using forward-fill to interpolate weather to 5-min intervals")
    
    # Resample weather to 5-minute intervals using forward fill
    # This fills each hour's weather forward to all 12 subsequent 5-min intervals
    df_weather_5min = df_weather.resample('5min').ffill()
    
    print(f"   - Weather after resampling: {df_weather_5min.shape[0]} rows")
    
    # ============================================================================
    # 2.5 Merge Datasets
    # ============================================================================
    print("\n[2.4] Merging traffic and weather data...")
    
    # Find overlapping date range
    traffic_start = df_traffic.index.min()
    traffic_end = df_traffic.index.max()
    weather_start = df_weather_5min.index.min()
    weather_end = df_weather_5min.index.max()
    
    # Calculate overlap
    overlap_start = max(traffic_start, weather_start)
    overlap_end = min(traffic_end, weather_end)
    
    print(f"   - Traffic date range: {traffic_start} to {traffic_end}")
    print(f"   - Weather date range: {weather_start} to {weather_end}")
    print(f"   - Overlapping period: {overlap_start} to {overlap_end}")
    
    # Filter both datasets to overlapping period
    df_traffic_filtered = df_traffic.loc[overlap_start:overlap_end]
    df_weather_filtered = df_weather_5min.loc[overlap_start:overlap_end]
    
    # Merge on timestamp (inner join to keep only overlapping times)
    # Add prefix to weather columns to avoid name conflicts
    df_weather_filtered = df_weather_filtered.add_prefix('weather_')
    
    # Perform the merge
    merged_df = df_traffic_filtered.join(df_weather_filtered, how='inner')
    
    # Handle any remaining missing values
    missing_before = merged_df.isnull().sum().sum()
    merged_df = merged_df.ffill()  # Forward fill any remaining gaps
    merged_df = merged_df.bfill()  # Backward fill for any initial gaps
    missing_after = merged_df.isnull().sum().sum()
    
    print(f"\n[2.5] Merge complete!")
    print(f"   - Merged dataset shape: {merged_df.shape}")
    print(f"   - Missing values before cleaning: {missing_before}")
    print(f"   - Missing values after cleaning: {missing_after}")
    
    # ============================================================================
    # 2.6 Basic Data Statistics
    # ============================================================================
    print("\n[2.6] Dataset Statistics:")
    print(f"   - Total timesteps: {merged_df.shape[0]}")
    print(f"   - Total features: {merged_df.shape[1]} (207 traffic + 3 weather)")
    
    # Traffic speed statistics
    traffic_cols = [c for c in merged_df.columns if not c.startswith('weather_')]
    weather_cols = [c for c in merged_df.columns if c.startswith('weather_')]
    
    print(f"\n   Traffic Speed Statistics:")
    print(f"   - Mean: {merged_df[traffic_cols].mean().mean():.2f} mph")
    print(f"   - Std: {merged_df[traffic_cols].std().mean():.2f} mph")
    print(f"   - Min: {merged_df[traffic_cols].min().min():.2f} mph")
    print(f"   - Max: {merged_df[traffic_cols].max().max():.2f} mph")
    
    print(f"\n   Weather Statistics:")
    for col in weather_cols:
        print(f"   - {col}: min={merged_df[col].min():.2f}, max={merged_df[col].max():.2f}, mean={merged_df[col].mean():.2f}")
    
    # ============================================================================
    # 2.7 Save Merged Dataset
    # ============================================================================
    print("\n[2.7] Saving merged dataset...")
    
    output_file = 'METR_LA_with_Weather_5min.csv'
    merged_df.to_csv(output_file)
    
    print(f"   - Saved to: {output_file}")
    print(f"   - File size: {os.path.getsize(output_file) / (1024*1024):.2f} MB")
    
    # ============================================================================
    # 2.8 Create Single Sensor Dataset (for Chronos-2)
    # ============================================================================
    print("\n[2.8] Creating single-sensor dataset for Chronos-2...")
    
    # Select first sensor for initial experiments
    sensor_id = traffic_cols[0]
    single_sensor_df = merged_df[[sensor_id] + weather_cols].copy()
    single_sensor_df.columns = ['traffic_speed', 'temperature', 'precipitation', 'wind_speed']
    
    # Save single sensor dataset
    single_sensor_file = 'single_sensor_with_weather.csv'
    single_sensor_df.to_csv(single_sensor_file)
    
    print(f"   - Saved single sensor data to: {single_sensor_file}")
    print(f"   - Shape: {single_sensor_df.shape}")
    print(f"   - First few rows:")
    print(single_sensor_df.head(10).to_string())
    
    print("\n" + "=" * 60)
    print("STEP 2 COMPLETE: Data preprocessing finished successfully!")
    print("=" * 60)
    print("\nNext steps:")
    print("  1. Run step3_chronos_inference.py for forecasting")
    print("  2. Run step4_evaluation_metrics.py for model evaluation")