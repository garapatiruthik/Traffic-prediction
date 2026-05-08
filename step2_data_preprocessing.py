"""
Step 2: Data Preprocessing - Single-Sensor Extraction & Weather Merge
======================================================================

This script performs robust extraction of a single sensor from METR-LA,
cleans it (0.0 -> NaN -> ffill/bfill), then merges with 2012 weather data.

Key improvements for production readiness:
1. HDF5/CSV dual-format loading with fallback
2. Single-sensor isolation (Sensor ID: 773869)
3. Explicit 0.0 -> NaN imputation (dead sensor handling)
4. Comprehensive logging for academic defense

Author: Suvarna Kotha & Ruthik Garapati
Thesis: Urban Traffic Forecasting - Comparative Analysis
"""

import pandas as pd
import numpy as np
import os
import sys

print("=" * 60)
print("STEP 2: Single-Sensor Extraction + Weather Merge")
print("=" * 60)

# ============================================================================
# CONFIGURATION
# ============================================================================
SENSOR_ID = '773869'          # Chosen sensor for univariate forecasting
TRAFFIC_COL_NAME = 'traffic_speed'  # Standardized output column name

METR_LA_H5 = 'metr-la.h5'     # Preferred format (compressed)
METR_LA_CSV = 'METR-LA_cleaned.csv'  # Fallback format
WEATHER_FILE = 'LA_Weather_Hourly_2012_Full.csv'
OUTPUT_MERGED = 'METR_LA_with_Weather_5min.csv'
OUTPUT_SINGLE = 'single_sensor_with_weather.csv'

# ============================================================================
# 2.1 Load METR-LA Traffic Data (Robust HDF5/CSV Loading)
# ============================================================================
print("\n[2.1] Loading METR-LA traffic data...")

df_traffic = None

# Strategy 1: Try HDF5 (compressed, efficient)
if os.path.exists(METR_LA_H5):
    try:
        print(f"   -> Found {METR_LA_H5}, loading via pd.read_hdf()...")
        df_traffic = pd.read_hdf(METR_LA_H5)
        print(f"   [OK] HDF5 loaded successfully")
    except Exception as e:
        print(f"   [FAIL] HDF5 load failed: {e}")
        df_traffic = None

# Strategy 2: Fallback to CSV
if df_traffic is None and os.path.exists(METR_LA_CSV):
    try:
        print(f"   -> Falling back to {METR_LA_CSV}...")
        df_traffic = pd.read_csv(METR_LA_CSV, index_col=0)
        print(f"   [OK] CSV loaded successfully")
    except Exception as e:
        print(f"   [FAIL] CSV load failed: {e}")
        df_traffic = None

# Strategy 3: Not found
if df_traffic is None:
    print("\n   ERROR: METR-LA data not found!")
    print("   Expected files:")
    print(f"     - {METR_LA_H5} (preferred)")
    print(f"     - {METR_LA_CSV} (fallback)")
    print("   Please download METR-LA dataset first.")
    sys.exit(1)

# Convert index to DatetimeIndex
try:
    df_traffic.index = pd.to_datetime(df_traffic.index)
except Exception as e:
    print(f"   ERROR: Failed to parse datetime index: {e}")
    sys.exit(1)

print(f"\n   Original dataset:")
print(f"   - Shape: {df_traffic.shape[0]} rows × {df_traffic.shape[1]} columns")
print(f"   - Index range: {df_traffic.index.min()} to {df_traffic.index.max()}")
print(f"   - Frequency: {pd.infer_freq(df_traffic.index) or 'irregular'}")

# ============================================================================
# 2.2 Extract Single Sensor (773869) and Standardize Column Name
# ============================================================================
print(f"\n[2.2] Extracting single sensor: {SENSOR_ID}")

# Check if sensor exists
if SENSOR_ID not in df_traffic.columns:
    print(f"   ERROR: Sensor {SENSOR_ID} not found in dataset!")
    print(f"   Available sensors (first 10): {list(df_traffic.columns[:10])}")
    sys.exit(1)

# Extract only the target sensor column
df_single = df_traffic[[SENSOR_ID]].copy()
df_single.columns = [TRAFFIC_COL_NAME]  # Rename to standardized name

print(f"   [OK] Extracted sensor {SENSOR_ID}")
print(f"   - New shape: {df_single.shape}")
print(f"   - Column name: '{TRAFFIC_COL_NAME}'")

# Drop all other sensors to save memory
del df_traffic
import gc; gc.collect()
print(f"   - Dropped other {207-1} sensors -> memory optimized")

# ============================================================================
# 2.3 Data Cleaning: 0.0 -> NaN Imputation
# ============================================================================
print(f"\n[2.3] Cleaning data: handling zero-speed values...")

# CRITICAL EXPLANATION for thesis defense:
# In the METR-LA dataset, a speed reading of 0.0 mph does NOT mean
# traffic is stopped. It indicates:
#   - Sensor malfunction/dead sensor
#   - Temporary communication failure
#   - Placeholder for missing data
#
# Therefore, we treat 0.0 as missing and impute via forward/backward fill.
# This preserves temporal continuity without injecting artificial patterns.

zeros_mask = (df_single[TRAFFIC_COL_NAME] == 0.0)
n_zeros = zeros_mask.sum()
print(f"   - Zero-speed values found: {n_zeros} ({(n_zeros/len(df_single)*100):.2f}%)")

if n_zeros > 0:
    # Replace 0.0 with NaN for proper imputation
    df_single.loc[zeros_mask, TRAFFIC_COL_NAME] = np.nan
    print(f"   -> Replaced {n_zeros} zeros with NaN")

# Impute: forward-fill then backward-fill to close gaps
missing_before = df_single[TRAFFIC_COL_NAME].isna().sum()
print(f"   - Missing values before ffill/bfill: {missing_before}")

df_single[TRAFFIC_COL_NAME] = df_single[TRAFFIC_COL_NAME].ffill()
df_single[TRAFFIC_COL_NAME] = df_single[TRAFFIC_COL_NAME].bfill()

missing_after = df_single[TRAFFIC_COL_NAME].isna().sum()
print(f"   - Missing after forward+backward fill: {missing_after}")

if missing_after > 0:
    print(f"   WARNING: {missing_after} values still missing!")
    print(f"   Dropping remaining NaNs...")
    df_single = df_single.dropna()
    print(f"   Remaining after dropna: {len(df_single)} rows")

# ============================================================================
# 2.4 Load Weather Data
# ============================================================================
print(f"\n[2.4] Loading weather data...")

if not os.path.exists(WEATHER_FILE):
    print(f"   ERROR: Weather file not found: {WEATHER_FILE}")
    print("   Run step1_download_weather.py first.")
    sys.exit(1)

df_weather = pd.read_csv(WEATHER_FILE)
print(f"   - Raw weather shape: {df_weather.shape}")

# Parse datetime
if 'datetime' in df_weather.columns:
    df_weather['datetime'] = pd.to_datetime(df_weather['datetime'])
    df_weather = df_weather.set_index('datetime')
elif {'year', 'month', 'day', 'hour'}.issubset(df_weather.columns):
    df_weather['datetime'] = pd.to_datetime(df_weather[['year', 'month', 'day', 'hour']])
    df_weather = df_weather.set_index('datetime')
else:
    print("   ERROR: Cannot parse datetime from weather data")
    sys.exit(1)

print(f"   - Date range: {df_weather.index.min()} -> {df_weather.index.max()}")
print(f"   - Columns: {list(df_weather.columns)}")

# ============================================================================
# 2.5 Temporal Alignment: Resample Weather to 5-Minute Intervals
# ============================================================================
print(f"\n[2.5] Resampling weather to 5-minute intervals...")

# Weather is hourly -> need to upsample to 5-min to match traffic
df_weather_5min = df_weather.resample('5T').ffill()

print(f"   - After resampling: {df_weather_5min.shape[0]} rows")
print(f"   - New frequency: 5-minute intervals")

# ============================================================================
# 2.6 Merge: Align to Overlapping Date Range
# ============================================================================
print(f"\n[2.6] Merging traffic and weather data...")

# Find intersection of date ranges
traffic_start = df_single.index.min()
traffic_end = df_single.index.max()
weather_start = df_weather_5min.index.min()
weather_end = df_weather_5min.index.max()

overlap_start = max(traffic_start, weather_start)
overlap_end = min(traffic_end, weather_end)

print(f"   Traffic range: {traffic_start} -> {traffic_end}")
print(f"   Weather range: {weather_start} -> {weather_end}")
print(f"   Overlap:       {overlap_start} -> {overlap_end}")

if overlap_start >= overlap_end:
    print("   ERROR: No overlapping date range!")
    sys.exit(1)

# Filter to overlap
df_traffic_filt = df_single.loc[overlap_start:overlap_end]
df_weather_filt = df_weather_5min.loc[overlap_start:overlap_end]

print(f"   - Filtered traffic: {len(df_traffic_filt)} rows")
print(f"   - Filtered weather: {len(df_weather_filt)} rows")

# Add prefix to weather columns to avoid collision
df_weather_filt = df_weather_filt.add_prefix('weather_')

# Inner join on index (timestamp)
merged = df_traffic_filt.join(df_weather_filt, how='inner')

# Handle any edge-case NaNs from join misalignment
missing_pre = merged.isnull().sum().sum()
merged = merged.ffill().bfill()
missing_post = merged.isnull().sum().sum()

print(f"\n[2.7] Merge complete!")
print(f"   - Final shape: {merged.shape[0]} rows × {merged.shape[1]} columns")
print(f"   - Missing before final clean: {missing_pre}")
print(f"   - Missing after final clean: {missing_post}")

# ============================================================================
# 2.8 Save Merged Dataset
# ============================================================================
print(f"\n[2.8] Saving merged dataset...")

merged.to_csv(OUTPUT_MERGED)
print(f"   [OK] Saved: {OUTPUT_MERGED}")
print(f"   - File size: {os.path.getsize(OUTPUT_MERGED) / (1024*1024):.2f} MB")

# ============================================================================
# 2.9 Create Single-Sensor Dataset (for Chronos/other models)
# ============================================================================
print(f"\n[2.9] Creating single-sensor dataset (Chronos format)...")

# Ensure traffic column is named 'traffic_speed' (already done)
# Weather columns already have 'weather_' prefix
single_sensor_df = merged.copy()  # Already has only 1 traffic col + weather cols

single_sensor_df.to_csv(OUTPUT_SINGLE)
print(f"   [OK] Saved: {OUTPUT_SINGLE}")
print(f"   - Shape: {single_sensor_df.shape}")
print(f"   - Columns: {list(single_sensor_df.columns)}")

# ============================================================================
# 2.10 Final Statistics & Integrity Report
# ============================================================================
print("\n" + "=" * 60)
print("FINAL DATA INTEGRITY REPORT")
print("=" * 60)

traffic_col = [c for c in merged.columns if not c.startswith('weather_')][0]
weather_cols = [c for c in merged.columns if c.startswith('weather_')]

print(f"\nTraffic Speed ({traffic_col}):")
stats = merged[traffic_col].describe()
print(f"   - Count:   {stats['count']:.0f}")
print(f"   - Mean:    {stats['mean']:.2f} mph")
print(f"   - Std:     {stats['std']:.2f} mph")
print(f"   - Min:     {stats['min']:.2f} mph")
print(f"   - Max:     {stats['max']:.2f} mph")
print(f"   - Median:  {stats['50%']:.2f} mph")

print(f"\nWeather Features:")
for col in weather_cols:
    s = merged[col].describe()
    print(f"   - {col}: mean={s['mean']:.2f}, min={s['min']:.2f}, max={s['max']:.2f}")

print(f"\nDate Range: {merged.index.min()} -> {merged.index.max()}")
print(f"Total duration: {(merged.index.max() - merged.index.min()).days} days")
print(f"Missing values: {merged.isnull().sum().sum()} (should be 0)")

# ============================================================================
# Summary
# ============================================================================
print("\n" + "=" * 60)
print("STEP 2 COMPLETE")
print("=" * 60)
print(f"\nOutputs generated:")
print(f"  1. {OUTPUT_MERGED}")
print(f"  2. {OUTPUT_SINGLE}")
print(f"\nNext step: Run step3_chronos_inference.py or step5_mamba_training.py")

