#!/usr/bin/env python3
"""
VERIFY OUTPUTS - Check for double-scaling bug and missing files
Run this AFTER running the full pipeline.
"""

import os
import pandas as pd
import numpy as np

print("=" * 70)
print("TRAFFIC FORECASTING - OUTPUT VERIFICATION & BUG CHECK")
print("=" * 70)

# ============================================================================
# 1. CHECK FOR CRITICAL BUG: 793+ mph values
# ============================================================================
print("\n[1] CHECKING FOR DOUBLE-SCALING BUG")
print("-" * 70)
print("Bug: 'Actual' speeds showing 700-800 mph (should be 60-70 mph)")
print("Cause: y_test * speed_std + speed_mean when y_test already unscaled\n")

bug_detected = False

for fname, label in [
    ('mamba_predictions_may2013.csv', 'May 2013'),
    ('mamba_predictions_jun2013.csv', 'June 2013'),
    ('mamba_predictions_may2012.csv', 'May 2012 (baseline)')
]:
    if os.path.exists(fname):
        df = pd.read_csv(fname)
        if 'actual' in df.columns:
            mean_actual = df['actual'].mean()
            mean_pred = df['predicted_mean'].mean()

            # Flag if > 100 mph (impossible for LA traffic)
            if mean_actual > 100 or mean_pred > 100:
                print(f"  [FAIL] {label}: Actual mean = {mean_actual:.2f} mph (BUG!)")
                bug_detected = True
            else:
                print(f"  [OK] {label}: Actual = {mean_actual:.2f} mph, Predicted = {mean_pred:.2f} mph")
        else:
            print(f"  [?] {label}: No 'actual' column found")
    else:
        print(f"  [FAIL] {label}: File missing")

if bug_detected:
    print("\n" + "=" * 70)
    print("BUG DETECTED! Values > 100 mph indicate double-scaling.")
    print("FIX: In month_ahead_forecasting.py, change:")
    print("  may2013_actual = y_test2 * speed_std + speed_mean")
    print("  jun2013_actual = y_test3 * speed_std + speed_mean")
    print("To:")
    print("  may2013_actual = y_test2")
    print("  jun2013_actual = y_test3")
    print("Then re-run month_ahead_forecasting.py")
    print("=" * 70)
else:
    print("\n[OK] No double-scaling bug detected - values look realistic!")

# ============================================================================
# 2. CHECK FOR MISSING OUTPUT FILES
# ============================================================================
print("\n[2] CHECKING FOR MISSING OUTPUT FILES")
print("-" * 70)

required_files = {
    'METR_LA_with_Weather_5min.csv': 'Merged dataset (step2 output)',
    'single_sensor_with_weather.csv': 'Chronos input (step2 output)',
    'mamba_best_model.pt': 'Trained Mamba model',
    'mamba_predictions_may2012.csv': 'May 2012 predictions',
    'mamba_predictions_may2013.csv': 'May 2013 predictions (CRITICAL)',
    'mamba_predictions_jun2013.csv': 'June 2013 predictions (CRITICAL)',
    'month_ahead_comparison.csv': 'Metrics table',
    'FIGURE3_same_month_different_year.png': 'FIG 3: May 2012 vs 2013',
    'FIGURE4_month_ahead_comparison.png': 'FIG 4: Month-ahead comparison',
}

missing = []
for fname, desc in required_files.items():
    if os.path.exists(fname):
        size = os.path.getsize(fname)
        if fname.endswith('.png'):
            print(f"  [OK] {desc:45s} [{size/1024:.0f} KB]")
        elif fname.endswith('.pt'):
            print(f"  [OK] {desc:45s} [{size/1048576:.1f} MB]")
        else:
            print(f"  [OK] {desc:45s} [{size/1024:.0f} KB]")
    else:
        print(f"  [FAIL] {desc:45s} [MISSING]")
        missing.append(fname)

if missing:
    print(f"\n[WARN] Missing {len(missing)} file(s):")
    for f in missing:
        print(f"  - {f}")
    print("\nRun these scripts to generate them:")
    if 'mamba_best_model.pt' in missing:
        print("  1. python step5_mamba_training.py")
    if any(f in missing for f in ['mamba_predictions_may2013.csv', 'mamba_predictions_jun2013.csv']):
        print("  2. python month_ahead_forecasting.py")
    if 'FIGURE3_same_month_different_year.png' in missing:
        print("  3. python create_month_comparison_actual.py")
    if 'FIGURE4_month_ahead_comparison.png' in missing:
        print("  4. python create_month_ahead_viz.py")
    if 'METR_LA_with_Weather_5min.csv' in missing:
        print("  5. python step2_data_preprocessing.py")
else:
    print("\n[OK] All required files present!")

# ============================================================================
# 3. SUMMARY TABLE
# ============================================================================
print("\n[3] FINAL STATUS SUMMARY")
print("-" * 70)

if os.path.exists('month_ahead_comparison.csv'):
    df = pd.read_csv('month_ahead_comparison.csv')
    print("\nMonth-Ahead Comparison Table:")
    print(df.to_string(index=False))
else:
    print("  month_ahead_comparison.csv not found")

print("\n" + "=" * 70)
print("NEXT STEPS:")
print("=" * 70)
if bug_detected:
    print("1. Fix double-scaling bug in month_ahead_forecasting.py")
    print("2. Re-run: python month_ahead_forecasting.py")
elif missing:
    print("1. Run missing scripts from list above")
    print("2. Re-run this verification script")
else:
    print("[OK] ALL CHECKS PASSED")
    print("[OK] May 2013 and June 2013 predictions are realistic (~60-70 mph)")
    print("[OK] All figures and CSVs generated successfully")
    print("\nYou can now use these files for your thesis report!")
print("=" * 70)