"""
Month-Ahead Forecasting Visualization — Figure 4
==================================================
Compares May 2012 (baseline) and May 2013 (autoregressive projection).

Panels:
  A — May 2012: Actual vs Predicted  (ground-truth evaluation)
  B — May 2013: Predicted vs Historical 2012 Reference  (autoregressive projection)
  C — Error Distribution  (May 2012 only; 2013 note included)
  D — Statistical Summary

This script loads ONLY May 2012 and May 2013 prediction files.
It gracefully handles missing 'actual' and 'predicted_std' columns
in the 2013 file (which contains NaN actuals because no real 2013
traffic data exists).

Author: Suvarna Kotha & Ruthik Garapati
Thesis: Urban Traffic Forecasting (May 2026)
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# Publication-quality style
# =============================================================================
plt.rcParams.update({
    'font.size': 10,
    'font.family': 'serif',
    'axes.linewidth': 1.2,
    'xtick.major.width': 1.2,
    'ytick.major.width': 1.2,
    'xtick.major.size': 5,
    'ytick.major.size': 5,
    'figure.dpi': 200,
    'savefig.dpi': 200,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.3,
})

try:
    import seaborn as sns
    sns.set_style('whitegrid', {'grid.linestyle': '--', 'grid.alpha': 0.4})
except ImportError:
    plt.style.use('seaborn-v0_8-whitegrid')

print("=" * 60)
print("FIGURE 4: Month-Ahead Traffic Forecast Visualization")
print("=" * 60)

# =============================================================================
# PHASE 1: DATA LOADING  (May 2012 + May 2013 only)
# =============================================================================
print("\n[1] Loading prediction data...")

# --- May 2012  (Group B — primary + backup) ---
_may2012_paths = ['mamba_predictions_may2012.csv', 'autoregressive_predictions_2012_standard.csv']
_may2012_file = next((p for p in _may2012_paths if os.path.exists(p)), None)
if _may2012_file is None:
    raise FileNotFoundError("Could not find May 2012 prediction file — "
                            "looked in: " + ", ".join(_may2012_paths))
may2012_df = pd.read_csv(_may2012_file)
may2012_df['timestamp'] = pd.to_datetime(may2012_df['timestamp'])
print(f"  May 2012:    {len(may2012_df)} rows, cols={list(may2012_df.columns)}")

# --- May 2013  (Group C — primary + backup) ---
_may2013_paths = ['mamba_predictions_may2013.csv', 'autoregressive_predictions_2013_rolling.csv']
_may2013_file = next((p for p in _may2013_paths if os.path.exists(p)), None)
if _may2013_file is None:
    raise FileNotFoundError("Could not find May 2013 prediction file — "
                            "looked in: " + ", ".join(_may2013_paths))
may2013_df = pd.read_csv(_may2013_file)
may2013_df['timestamp'] = pd.to_datetime(may2013_df['timestamp'])
print(f"  May 2013:    {len(may2013_df)} rows, cols={list(may2013_df.columns)}")

# --- Graceful handling of missing columns in May 2013 / May 2012 ---
# May 2012 files use 'actual_speed'; May 2013 files have NO ground-truth column.
# Neither file provides 'predicted_std'.
_actual_col_2012 = 'actual_speed' if 'actual_speed' in may2012_df.columns else (
                   'actual'          if 'actual'          in may2012_df.columns else None)
may2012_has_actual   = _actual_col_2012 is not None and not may2012_df[_actual_col_2012].isna().all()
may2013_has_actual   = 'actual_speed' in may2013_df.columns and not may2013_df['actual_speed'].isna().all()
may2013_has_actual   = may2013_has_actual or ('actual' in may2013_df.columns and not may2013_df['actual'].isna().all())
may2012_has_pred_std = False   # neither 2012 nor 2013 file carries predicted_std
may2013_has_pred_std = False

# Show all weather columns that were returned by the forecasting script
_weather_cols = [c for c in may2013_df.columns if c.startswith('weather_')]
print(f"  May 2013 weather cols: {_weather_cols}" if _weather_cols else "")

print(f"\n[2] Column availability check:")
_a12  = 'YES' if may2012_has_actual     else 'NO/MISSING'
_p12  = 'YES' if may2012_has_pred_std   else 'NO/MISSING'
_a13  = 'YES' if may2013_has_actual     else 'NO/MISSING'
_p13  = 'YES' if may2013_has_pred_std   else 'NO/MISSING'
print(f"  May 2012 actual ({_actual_col_2012}):       {_a12}")
print(f"  May 2012 predicted_std: {_p12}")
print(f"  May 2013 actual:       {_a13}")
print(f"  May 2013 predicted_std: {_p13}")

# =============================================================================
# PHASE 2: FIGURE — 4 PANELS (2x2 GridSpec)
# =============================================================================
print("\n[3] Building 4-panel Figure 4...")

fig = plt.figure(figsize=(18, 11))
gs = gridspec.GridSpec(2, 2, figure=fig,
                        hspace=0.38, wspace=0.30,
                        top=0.92, bottom=0.08,
                        left=0.08, right=0.97)

# Colour palette
CLR_ACTUAL   = '#2471A3'   # blue — ground truth
CLR_PREDICT  = '#C0392B'   # red  — model prediction
CLR_REF      = '#7D3C98'   # purple dashed — historical 2012 reference
CLR_BAND     = '#E74C3C'   # light red — confidence band

# Helper: number of display points (≈ 3 days = 432 five-minute intervals)
n_show_a = min(432, len(may2012_df))
n_show_b = min(432, len(may2013_df))


# ---------------------------------------------------------------
# Panel A: May 2012 — Baseline Evaluation
# ---------------------------------------------------------------
ax_a = fig.add_subplot(gs[0, 0])

x_a = np.arange(n_show_a) * 5.0 / 60.0     # 5-min steps → hours

# --- Full-month arrays (metrics use the ENTIRE month, not just the plotted window)
full_actual = may2012_df[_actual_col_2012].values
full_pred   = may2012_df['predicted_mean'].values
mae_a  = float(np.nanmean(np.abs(full_actual - full_pred)))
rmse_a = float(np.sqrt(np.nanmean((full_actual - full_pred) ** 2)))

# --- Sliced arrays for the 72-hour plot window ONLY
actual_a   = full_actual[:n_show_a]
pred_a     = full_pred[:n_show_a]
pred_std_a = may2012_df['predicted_std'].values[:n_show_a] if may2012_has_pred_std else np.full(n_show_a, np.nan)

ax_a.plot(x_a, actual_a, color=CLR_ACTUAL, linewidth=1.5,
          alpha=0.85, label='Ground Truth', zorder=3)
ax_a.plot(x_a, pred_a, color=CLR_PREDICT, linewidth=1.5,
          alpha=0.9, label='Mamba Prediction', zorder=2)

if may2012_has_pred_std:
    ax_a.fill_between(x_a,
                      pred_a - pred_std_a,
                      pred_a + pred_std_a,
                      alpha=0.18, color=CLR_BAND,
                      label='±1 SD', zorder=1)

ax_a.set_xlabel('Hours from Start of May 2012', fontweight='bold', fontsize=10)
ax_a.set_ylabel('Speed (mph)', fontweight='bold', fontsize=10)
ax_a.set_title('A) May 2012 — Baseline Evaluation\n'
               '(Trained on Mar–Apr, tested on May 2012)',
               fontweight='bold', fontsize=11, loc='left')
ax_a.set_xlim(0, n_show_a * 5.0 / 60.0)
ax_a.set_ylim(40, 75)
ax_a.legend(loc='upper right', fontsize=8, framealpha=0.9)
ax_a.tick_params(axis='both', which='major', labelsize=8)
ax_a.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

# MAE annotation (mae_a / rmse_a already calculated from full month above)
ax_a.text(0.02, 0.97,
          f'MAE = {mae_a:.2f} mph  |  RMSE = {rmse_a:.2f} mph',
          transform=ax_a.transAxes, fontsize=9, fontweight='bold',
          verticalalignment='top',
          bbox=dict(boxstyle='round,pad=0.4',
                    facecolor='#D5F5E3',
                    edgecolor='#27AE60', alpha=0.95))


# ---------------------------------------------------------------
# Panel B: May 2013 — Autoregressive Projection
# ---------------------------------------------------------------
ax_b = fig.add_subplot(gs[0, 1])

x_b = np.arange(n_show_b) * 5.0 / 60.0

pred_b   = may2013_df['predicted_mean'].values[:n_show_b]
pred_std_b = may2013_df['predicted_std'].values[:n_show_b] if may2013_has_pred_std else np.full(n_show_b, np.nan)

# Historical reference: reuse May 2012 actual values
# (same calendar window, one year earlier)
ref_b = actual_a[:n_show_b] if len(actual_a) >= n_show_b \
        else np.pad(actual_a, (0, n_show_b - len(actual_a)), mode='edge')

ax_b.plot(x_b, pred_b, color=CLR_PREDICT, linewidth=1.5,
          alpha=0.9, label='May 2013 Predicted', zorder=2)
ax_b.plot(x_b, ref_b, color=CLR_REF, linewidth=1.4, linestyle='--',
          alpha=0.75, label='Historical 2012 Reference', zorder=1)

if may2013_has_pred_std:
    ax_b.fill_between(x_b,
                      pred_b - pred_std_b,
                      pred_b + pred_std_b,
                      alpha=0.15, color=CLR_BAND, zorder=0)

ax_b.set_xlabel('Hours from Start of May 2013', fontweight='bold', fontsize=10)
ax_b.set_ylabel('Speed (mph)', fontweight='bold', fontsize=10)
ax_b.set_title('B) May 2013 — Autoregressive Forecast\n'
               '(Trained on all 2012 data, seed from Apr 30 2012)',
               fontweight='bold', fontsize=11, loc='left')
ax_b.set_xlim(0, n_show_b * 5.0 / 60.0)
ax_b.set_ylim(40, 75)
ax_b.legend(loc='upper right', fontsize=8, framealpha=0.9)
ax_b.tick_params(axis='both', which='major', labelsize=8)
ax_b.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

# Annotation
mean_pred_b  = float(np.nanmean(pred_b))
pred_std_avg = float(np.nanmean(pred_std_b)) if may2013_has_pred_std else np.nan
ref_mean     = float(np.nanmean(ref_b))
ax_b.text(0.02, 0.97,
          f'Mean Predicted: {mean_pred_b:.1f} mph\n'
          f'Reference Mean:  {ref_mean:.1f} mph',
          transform=ax_b.transAxes, fontsize=9, fontweight='bold',
          verticalalignment='top',
          bbox=dict(boxstyle='round,pad=0.4',
                    facecolor='#FDEBD0',
                    edgecolor='#E67E22', alpha=0.95))


# ---------------------------------------------------------------
# Panel C: Error Distribution (May 2012 only)
# ---------------------------------------------------------------
ax_c = fig.add_subplot(gs[1, 0])

errors_a = may2012_df[_actual_col_2012].values - may2012_df['predicted_mean'].values
errors_a = errors_a[~np.isnan(errors_a)]

# Trim > 4σ outliers for a cleaner histogram
mean_err, std_err = np.mean(errors_a), np.std(errors_a)
mask = np.abs(errors_a - mean_err) <= 4 * std_err
errors_trimmed = errors_a[mask]

ax_c.hist(errors_trimmed, bins=60, color=CLR_ACTUAL, alpha=0.65,
          edgecolor='white', linewidth=0.3, density=True)
ax_c.axvline(x=0, color='black', linestyle='-', linewidth=1.2, alpha=0.5)
ax_c.axvline(x=np.mean(errors_trimmed), color=CLR_PREDICT,
             linestyle='--', linewidth=1.5,
             label=f'Mean Error = {np.mean(errors_trimmed):+.2f} mph')

ax_c.set_xlabel('Prediction Error  (Actual − Predicted)  [mph]',
                fontweight='bold', fontsize=10)
ax_c.set_ylabel('Density', fontweight='bold', fontsize=10)
ax_c.set_title('C) Error Distribution — May 2012',
               fontweight='bold', fontsize=11, loc='left')
ax_c.legend(loc='upper left', fontsize=8)
ax_c.tick_params(axis='both', which='major', labelsize=8)
ax_c.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

# Overlay note about 2013
ax_c.text(0.98, 0.95,
          '2013 Error Distribution\nUnavailable (No Ground Truth)',
          transform=ax_c.transAxes, fontsize=8, fontstyle='italic',
          verticalalignment='top', horizontalalignment='right',
          color='#7F8C8D',
          bbox=dict(boxstyle='round,pad=0.4',
                    facecolor='#F8F9FA',
                    edgecolor='#D5D8DC', alpha=0.9))


# =============================================================================
# D) Clean Native Month-Ahead Performance Table Fix
# =============================================================================
ax_d = fig.add_subplot(gs[1, 1])
ax_d.axis('off')  # Strip wireframe coordinates container box
ax_d.set_title('D) Performance & Macro Summaries', fontweight='bold', fontsize=12, loc='left')

df_a = may2012_df; df_b = may2013_df   # local aliases used by table rows

metrics_data = [
    ["Performance Parameter", "Calculated Baseline Values", "Unit Track"],
    ["Mean Absolute Error (MAE)", f"{mae_a:.4f}", "mph"],
    ["Root Mean Squared Error (RMSE)", f"{rmse_a:.4f}", "mph"],
    ["May 2012 Ground-Truth Mean", f"{may2012_df[_actual_col_2012].mean():.2f}", "mph"],
    ["May 2012 Predictive Mean", f"{df_a['predicted_mean'].mean():.2f}", "mph"],
    ["May 2013 Autoregressive Mean", f"{df_b['predicted_mean'].mean():.2f}", "mph"],
    ["Total Forecasting Steps", f"{len(df_b)}", "5-min intervals"]
]

# Generate balanced tabular matrix elements
m_table = ax_d.table(cellText=metrics_data, loc='center', cellLoc='center')
m_table.auto_set_font_size(False)
m_table.set_fontsize(8.5)
m_table.scale(1.0, 1.4)  # Professional tabular scaling ratio

# Apply dark slate corporate profile theme to head columns
for col in range(3):
    h_cell = m_table[0, col]
    h_cell.set_text_props(weight='bold', color='white')
    h_cell.set_facecolor('#2c3e50')


# =============================================================================
# FOOTER VARIABLES (Panel B scope → shared before footer)
# =============================================================================
mean_actual_2012    = float(may2012_df[_actual_col_2012].mean())
mean_predicted_2012 = float(may2012_df['predicted_mean'].mean())
mean_pred_b = float(np.nanmean(may2013_df['predicted_mean'].values))
ref_mean    = float(np.nanmean(actual_a))

# =============================================================================
# GLOBAL TITLE
# =============================================================================
fig.suptitle(
    'Month-Ahead Traffic Forecasting: May 2012 vs May 2013\n'
    'Autoregressive Projection Using Real 2013 Weather Data',
    fontsize=15, fontweight='bold', y=0.99, color='#2C3E50'
)

# =============================================================================
# FOOTER — summary metrics bar
# =============================================================================
footer = (
    f"May 2012  |  MAE: {mae_a:.2f} mph  |  RMSE: {rmse_a:.2f} mph  |  "
    f"Avg Actual: {mean_actual_2012:.1f} mph  |  Avg Predicted: {mean_predicted_2012:.1f} mph        ||        "
    f"May 2013  |  Avg Predicted: {mean_pred_b:.1f} mph  |  Historical Ref: {ref_mean:.1f} mph"
)
fig.text(0.5, 0.01, footer,
         ha='center', va='bottom', fontsize=8, fontfamily='monospace',
         bbox=dict(boxstyle='round,pad=0.5',
                   facecolor='#F0F3F5',
                   edgecolor='#AAB7C4', linewidth=1, alpha=0.95))

# =============================================================================
# SAVE
# =============================================================================
out_path = 'FIGURE4_month_ahead_comparison.png'
fig.savefig(out_path, dpi=200)
plt.close()

print(f"\n  [SAVED] {out_path}")

# =============================================================================
# VERIFICATION
# =============================================================================
if os.path.exists(out_path):
    size_kb = os.path.getsize(out_path) / 1024
    print(f"  File size: {size_kb:.0f} KB")
else:
    print("  ERROR: Output file not created!")
    exit(1)

print("\n" + "=" * 60)
print("FIGURE 4 GENERATION COMPLETE")
print("=" * 60)
print(f"\nPanels generated:")
print(f"  A — May 2012 baseline: Actual vs Predicted (MAE={mae_a:.2f}, RMSE={rmse_a:.2f})")
print(f"  B — May 2013 autoregressive forecast (vs 2012 reference)")
print(f"  C — Error distribution (May 2012 only; 2013 N/A)")
print(f"  D — Statistical summary table")
print(f"\nKey metrics:")
print(f"  May 2012 MAE:     {mae_a:.2f} mph")
print(f"  May 2012 RMSE:    {rmse_a:.2f} mph")
print(f"  May 2013 predicted mean: {mean_pred_b:.2f} mph")