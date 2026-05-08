# FIXED COLAB RUNNER - Continues past optional step failures
import os, sys, subprocess

print("="*70)
print("TRAFFIC FORECASTING PIPELINE - COLAB RUNNER")
print("="*70)

# Install dependencies (with error handling)
print("\n[1] Installing dependencies...")
try:
    import pandas, numpy, sklearn, matplotlib, torch
    print("✓ Core libraries already installed")
except ImportError:
    print("Installing core libraries...")
    !pip install pandas numpy scikit-learn matplotlib -q

# Chronos (optional - may fail on Colab due to compatibility)
print("\n[2] Installing Chronos-2...")
chronos_installed = False
try:
    !pip install chronos-forecasting -q
    from chronos import ChronosPipeline
    chronos_installed = True
    print("✓ Chronos-2 installed successfully")
except Exception as e:
    print(f"⚠ Chronos-2 installation failed: {e}")
    print("  → Chronos step will be skipped")

# Mamba (GPU only, OK to fail on CPU)
print("\n[3] Installing Mamba...")
try:
    !pip install mamba-ssm causal-conv1d -q
    print("✓ Mamba installed")
except Exception as e:
    print(f"⚠ Mamba installation failed: {e}")
    print("  → Will use FFN fallback (still works)")

# ============================================================================
# DEFINE SCRIPTS WITH DEPENDENCIES
# ============================================================================
scripts = [
    {
        'file': 'step1_download_weather.py',
        'desc': 'Download weather data from Open-Meteo',
        'required_before': [],
        'skip_if': ['LA_Weather_Hourly_2012_Full.csv'],
        'critical': False
    },
    {
        'file': 'step2_data_preprocessing.py',
        'desc': 'Merge traffic + weather data',
        'required_before': ['METR-LA_cleaned.csv'],
        'skip_if': ['METR_LA_with_Weather_5min.csv'],
        'critical': True
    },
    {
        'file': 'step3_chronos_inference.py',
        'desc': 'Chronos-2 zero-shot baseline',
        'required_before': ['single_sensor_with_weather.csv'],
        'skip_if': ['chronos_predictions.csv'],
        'critical': False,  # Optional baseline
        'requires_module': 'chronos'
    },
    {
        'file': 'step4_evaluation_metrics.py',
        'desc': 'Evaluate Chronos predictions',
        'required_before': ['chronos_predictions.csv'],
        'skip_if': ['chronos_evaluation_results.csv'],
        'critical': False
    },
    {
        'file': 'step5_mamba_training.py',
        'desc': 'Train Mamba model',
        'required_before': ['METR_LA_with_Weather_5min.csv'],
        'skip_if': ['mamba_best_model.pt'],
        'critical': True
    },
    {
        'file': 'month_ahead_forecasting.py',
        'desc': 'Predict May & June 2013',
        'required_before': ['METR_LA_with_Weather_5min.csv'],
        'skip_if': ['mamba_predictions_may2013.csv'],
        'critical': True
    },
    {
        'file': 'create_month_comparison_actual.py',
        'desc': 'Generate FIGURE3: May 2012 vs May 2013',
        'required_before': ['mamba_predictions_may2013.csv'],
        'skip_if': ['FIGURE3_same_month_different_year.png'],
        'critical': True
    },
    {
        'file': 'create_month_ahead_viz.py',
        'desc': 'Generate FIGURE4: Full comparison',
        'required_before': ['mamba_predictions_may2012.csv', 'mamba_predictions_may2013.csv', 'mamba_predictions_jun2013.csv'],
        'skip_if': ['FIGURE4_month_ahead_comparison.png'],
        'critical': True
    }
]

# ============================================================================
# RUN PIPELINE
# ============================================================================
results = []

for i, script_info in enumerate(scripts, 1):
    script = script_info['file']
    desc = script_info['desc']
    required = script_info['required_before']
    skip_if = script_info['skip_if']
    critical = script_info['critical']
    needs_chronos = script_info.get('requires_module', None)

    print(f"\n{'='*70}")
    print(f"[{i}/{len(scripts)}] {script}")
    print(f"      {desc}")
    print(f"{'='*70}")

    # Check if we should skip (outputs already exist)
    skip = all(os.path.exists(f) for f in skip_if)
    if skip:
        print(f"  ⊗ SKIPPING - outputs already exist: {', '.join(skip_if)}")
        results.append((script, 'SKIPPED', None))
        continue

    # Check required input files exist
    missing_inputs = [f for f in required if not os.path.exists(f)]
    if missing_inputs:
        print(f"  ✗ MISSING INPUT: {', '.join(missing_inputs)}")
        if critical:
            print("  → CRITICAL STEP - cannot continue!")
            results.append((script, 'FAILED', 'Missing inputs'))
            break
        else:
            print("  → Skipping (optional)")
            results.append((script, 'SKIPPED', 'Missing inputs'))
            continue

    # Check module availability
    if needs_chronos and needs_chronos == 'chronos' and not chronos_installed:
        print(f"  ✗ Chronos module not available - skipping {script}")
        results.append((script, 'SKIPPED', 'Chronos not installed'))
        continue

    # Run the script
    print(f"  → Running: python {script}")
    ret = os.system(f'python {script}')

    if ret == 0:
        print(f"  ✓ COMPLETED successfully")
        results.append((script, 'SUCCESS', None))
    else:
        print(f"  ✗ FAILED with exit code {ret}")
        results.append((script, 'FAILED', f'Exit code {ret}'))
        if critical:
            print("  → CRITICAL STEP FAILED - stopping pipeline")
            break
        else:
            print("  → Continuing (optional step)")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*70)
print("PIPELINE SUMMARY")
print("="*70)

for script, status, error in results:
    status_symbol = {'SUCCESS': '✓', 'FAILED': '✗', 'SKIPPED': '⊝'}.get(status, '?')
    print(f"  {status_symbol} {script:45s} {status:10s}", end='')
    if error and status == 'FAILED':
        print(f" ({error})")
    else:
        print()

# Check which final outputs exist
print("\n" + "="*70)
print("FINAL OUTPUTS:")
print("="*70)

final_outputs = {
    'FIGURE3_same_month_different_year.png': 'May 2012 vs May 2013 comparison',
    'FIGURE4_month_ahead_comparison.png': 'Month-ahead forecast figure',
    'mamba_predictions_may2013.csv': 'May 2013 predictions',
    'mamba_predictions_jun2013.csv': 'June 2013 predictions',
    'month_ahead_comparison.csv': 'Metrics comparison table'
}

for fname, desc in final_outputs.items():
    if os.path.exists(fname):
        size = os.path.getsize(fname)
        if fname.endswith('.png'):
            print(f"  ✓ {desc}: {fname} [{size/1e3:.0f} KB]")
        else:
            print(f"  ✓ {desc}: {fname} [{size/1e3:.0f} KB]")
    else:
        print(f"  ✗ {desc}: {fname} NOT FOUND")

print("\n" + "="*70)
print("NEXT STEPS:")
print("="*70)
print("1. If any files failed, check the error messages above")
print("2. Download results using:")
print("   from google.colab import files")
print("   files.download('FIGURE3_same_month_different_year.png')")
print("   files.download('FIGURE4_month_ahead_comparison.png')")
print("3. Download CSVs for thesis data")
print("="*70)
