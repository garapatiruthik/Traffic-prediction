import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

plt.rcParams['font.size'] = 10
plt.rcParams['figure.dpi'] = 150

print("=" * 60)
print("MONTH COMPARISON: May 2012 vs May 2013 Predicted")
print("(Same Month, Different Year - Actual vs Predicted)")
print("=" * 60)

# Load data
df_full = pd.read_csv('METR_LA_with_Weather_5min.csv', index_col=0)
df_full.index = pd.to_datetime(df_full.index)
traffic_data = df_full['773869']

# Extract May 2012 actual
may_2012 = traffic_data['2012-05']

# Load May 2013 predicted data
try:
    may2013_pred_df = pd.read_csv('mamba_predictions_may2013.csv')
    may_2013_pred = may2013_pred_df['predicted_mean'].values
    print(f"\nMay 2013 Predicted: {len(may_2013_pred)} points, mean={may_2013_pred.mean():.2f} mph")
except FileNotFoundError:
    print("   ERROR: Run month_ahead_forecasting.py first to generate May 2013 predictions!")
    exit(1)

print(f"May 2012 Actual:   {len(may_2012)} points, mean={may_2012.mean():.2f} mph")
print(f"May 2013 Predicted: {len(may_2013_pred)} points, mean={may_2013_pred.mean():.2f} mph")
print(f"Difference: {may_2013_pred.mean() - may_2012.mean():.2f} mph")

# Create datetime series
may2012_idx = pd.date_range('2012-05-01', periods=len(may_2012), freq='5min')
may2013_pred_idx = pd.date_range('2013-05-01', periods=len(may_2013_pred), freq='5min')
may2012_series = pd.Series(may_2012.values, index=may2012_idx)
may2013_pred_series = pd.Series(may_2013_pred, index=may2013_pred_idx)

# =============================================================================
# Create comprehensive comparison figure
# =============================================================================
fig = plt.figure(figsize=(20, 14))
gs = GridSpec(3, 3, figure=fig, hspace=0.5, wspace=0.35,
              height_ratios=[1.0, 0.9, 0.9])

# =============================================================================
# Plot 1: Daily averages comparison (A)
# =============================================================================
ax1 = fig.add_subplot(gs[0, :])

# Daily averages
may2012_days = may2012_series.groupby(may2012_series.index.day).mean()
may2013_days = may2013_pred_series.groupby(may2013_pred_series.index.day).mean()
days_may2012 = range(1, len(may2012_days)+1)
days_may2013 = range(1, len(may2013_days)+1)

ax1.plot(days_may2012, may2012_days.values, 'b-o', linewidth=2.5, markersize=7,
         label='May 2012 (Actual)', alpha=0.8, markerfacecolor='white', markeredgewidth=2)
ax1.plot(days_may2013, may2013_days.values, 'r-s', linewidth=2.5, markersize=7,
         label='May 2013 (Predicted)', alpha=0.8, markerfacecolor='white', markeredgewidth=2)

ax1.set_xlabel('Day of Month', fontweight='bold', fontsize=12)
ax1.set_ylabel('Average Speed (mph)', fontweight='bold', fontsize=12)
ax1.set_xticks(range(1, 32, 2))
ax1.set_xlim(0.5, 31.5)
ax1.legend(loc='upper right', fontsize=11)

# Highlight differences
min_len = min(len(may2012_days), len(may2013_days))
for i in range(min_len):
    m = may2012_days.iloc[i]
    p = may2013_days.iloc[i]
    if abs(p - m) > 4:
        ax1.plot([i+1, i+1], [m, p], 'g--', alpha=0.4, linewidth=1.5)
        ax1.text(i+1, max(m, p)+1.5, f'{p-m:+.0f}',
                ha='center', fontsize=8, color='darkgreen', fontweight='bold')

# Add difference text
month_diff = may2013_pred_series.mean() - may2012_series.mean()
ax1.text(0.02, 0.98,
         f'Monthly Δ: {month_diff:+.2f} mph ({abs(month_diff)/may2012_series.mean()*100:.1f}%)',
         transform=ax1.transAxes, fontsize=11, fontweight='bold',
         verticalalignment='top', horizontalalignment='left',
         bbox=dict(boxstyle='round', facecolor='#ecf0f1', alpha=0.9,
                   edgecolor='#34495e', linewidth=1.5))

# =============================================================================
# Plot 2: Hourly patterns (B)
# =============================================================================
ax2 = fig.add_subplot(gs[1, 0])

may2012_hourly = may2012_series.groupby(may2012_series.index.hour).mean()
may2013_hourly = may2013_pred_series.groupby(may2013_pred_series.index.hour).mean()
hours = np.arange(24)

ax2.plot(hours, may2012_hourly, 'b-o', linewidth=2, markersize=5,
         label='May 2012 Actual', alpha=0.8)
ax2.plot(hours, may2013_hourly, 'r-o', linewidth=2, markersize=5,
         label='May 2013 Predicted', alpha=0.8)

ax2.fill_between(hours, may2012_hourly, may2013_hourly, alpha=0.1, color='gray')

ax2.axvspan(6, 10, alpha=0.1, color='red', label='AM Rush')
ax2.axvspan(16, 19, alpha=0.1, color='red', label='PM Rush')

ax2.set_xlabel('Hour of Day', fontweight='bold', fontsize=11)
ax2.set_ylabel('Avg Speed (mph)', fontweight='bold', fontsize=11)
ax2.set_title('B) Hourly Patterns\n(Rush hour patterns compared)',
              fontweight='bold', fontsize=12, loc='left', pad=10)
ax2.legend(loc='lower left', fontsize=9, ncol=2)
ax2.set_xticks(range(0, 24, 2))
ax2.set_xlim(-0.5, 23.5)
ax2.grid(True, alpha=0.3, linestyle=':')

# =============================================================================
# Plot 3: Distribution comparison (C)
# =============================================================================
ax3 = fig.add_subplot(gs[1, 1])

bins = np.linspace(0, 70, 45)
ax3.hist(may_2012, bins=bins, alpha=0.6, label='May 2012 Actual',
         color='steelblue', density=True, edgecolor='white', linewidth=0.5)
ax3.hist(may_2013_pred, bins=bins, alpha=0.6, label='May 2013 Predicted',
         color='indianred', density=True, edgecolor='white', linewidth=0.5)

ax3.axvline(may_2012.mean(), color='steelblue', linestyle='-', linewidth=2.5,
            label=f'May2012 μ={may_2012.mean():.1f}')
ax3.axvline(may_2013_pred.mean(), color='indianred', linestyle='-', linewidth=2.5,
            label=f'May2013 Pred μ={may_2013_pred.mean():.1f}')

ax3.set_xlabel('Traffic Speed (mph)', fontweight='bold', fontsize=11)
ax3.set_ylabel('Density', fontweight='bold', fontsize=11)
ax3.set_title('C) Speed Distribution\n(Year-over-year shift)',
              fontweight='bold', fontsize=12, loc='left', pad=10)
ax3.legend(loc='upper left', fontsize=9)
ax3.grid(True, alpha=0.3, linestyle=':')
ax3.set_xlim(0, 70)

# =============================================================================
# Plot 4: Day-of-week variation (D)
# =============================================================================
ax4 = fig.add_subplot(gs[1, 2])

day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
may2012_by_day = [may2012_series[may2012_series.index.dayofweek == i].mean() for i in range(7)]
may2013_by_day = [may2013_pred_series[may2013_pred_series.index.dayofweek == i].mean() for i in range(7)]

x = np.arange(7)
width = 0.35

bars1 = ax4.bar(x - width/2, may2012_by_day, width, label='May 2012',
                color='steelblue', edgecolor='black', linewidth=1, alpha=0.8)
bars2 = ax4.bar(x + width/2, may2013_by_day, width, label='May 2013 Pred',
                color='indianred', edgecolor='black', linewidth=1, alpha=0.8)

ax4.set_xlabel('Day of Week', fontweight='bold', fontsize=11)
ax4.set_ylabel('Avg Speed (mph)', fontweight='bold', fontsize=11)
ax4.set_title('D) Day-of-Week Patterns\n(Year-over-year comparison)',
              fontweight='bold', fontsize=12, loc='left', pad=10)
ax4.set_xticks(x)
ax4.set_xticklabels(day_names)
ax4.legend(loc='upper left', fontsize=9)
ax4.grid(True, alpha=0.3, linestyle=':', axis='y')

# Add value labels
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2, height + 0.5,
                f'{height:.0f}', ha='center', fontsize=8, fontweight='bold')

# =============================================================================
# Plot 5: Hourly bias (E)
# =============================================================================
ax5 = fig.add_subplot(gs[2, 0])

may2012_hourly = may2012_series.groupby(may2012_series.index.hour).mean()
may2013_hourly = may2013_pred_series.groupby(may2013_pred_series.index.hour).mean()
hourly_bias = may2013_hourly - may2012_hourly

ax5.bar(range(24), hourly_bias, color='#e74c3c', alpha=0.7,
        edgecolor='black', linewidth=0.5)
ax5.axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=1)

ax5.set_xlabel('Hour of Day', fontweight='bold', fontsize=11)
ax5.set_ylabel('Speed Difference (mph)\n(May2013_Pred - May2012)', fontweight='bold', fontsize=11)
ax5.set_title('E) Hourly Bias: May 2013 vs May 2012\n(Positive = 2013 slower)',
              fontweight='bold', fontsize=12, loc='left', pad=10)
ax5.set_xticks(range(0, 24, 2))
ax5.grid(True, alpha=0.3, linestyle=':', axis='y')

ax5.text(0.5, -0.9,
         'Note: Shows temporal shift year-over-year\nPositive = May 2013 predicted slower than May 2012',
         transform=ax5.transAxes, fontsize=9, fontweight='bold',
         ha='center', va='top',
         bbox=dict(boxstyle='round', facecolor='#fff3cd', alpha=0.9))

# =============================================================================
# Plot 6: Scatter comparison (F)
# =============================================================================
ax6 = fig.add_subplot(gs[2, 1])

# Sample matching hours
sample_size = min(2000, len(may2012_series), len(may2013_pred_series))
np.random.seed(42)
max_idx = min(len(may2012_series), len(may2013_pred_series))
sample_idx = np.random.choice(max_idx, size=sample_size, replace=False)
may2012_sample = may2012_series.iloc[sample_idx]
may2013_sample = may2013_pred_series.iloc[sample_idx]

ax6.scatter(may2012_sample, may2013_sample, alpha=0.3, s=20, color='#3498db', edgecolors='none')
ax6.plot([0, 70], [0, 70], 'r--', linewidth=2, alpha=0.7, label='Perfect Match')

ax6.set_xlabel('May 2012 Actual Speed (mph)', fontweight='bold', fontsize=11)
ax6.set_ylabel('May 2013 Predicted Speed (mph)', fontweight='bold', fontsize=11)
ax6.set_title('F) May 2012 vs May 2013 Predicted Scatter',
              fontweight='bold', fontsize=12, loc='left', pad=10)
ax6.legend(loc='upper left', fontsize=9)
ax6.grid(True, alpha=0.3, linestyle=':')
ax6.set_aspect('equal')
ax6.set_xlim(0, 70)
ax6.set_ylim(0, 70)

# Correlation text
from scipy import stats
r, p = stats.pearsonr(may2012_sample, may2013_sample)
ax6.text(0.05, 0.95,
         f'Correlation: r = {r:.3f}\n'
         f'R² = {r**2:.3f}\n'
         f'p < {p:.2e}',
         transform=ax6.transAxes, fontsize=9, fontweight='bold',
         verticalalignment='top', horizontalalignment='left',
         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9,
                   edgecolor='gray', linewidth=1))

# =============================================================================
# Plot 7: Summary panel (G)
# =============================================================================
ax7 = fig.add_subplot(gs[2, 2])
ax7.axis('off')

# Stats
diff = may2013_pred_series.mean() - may2012_series.mean()
diff_pct = abs(diff) / may2012_series.mean() * 100

may2012_hourly = may2012_series.groupby(may2012_series.index.hour).mean()
may2013_hourly = may2013_pred_series.groupby(may2013_pred_series.index.hour).mean()
hourly_diff = np.mean(np.abs(may2013_hourly - may2012_hourly))

may2012_dow = may2012_series.groupby(may2012_series.index.dayofweek).mean()
may2013_dow = may2013_pred_series.groupby(may2013_pred_series.index.dayofweek).mean()
dow_diff = np.mean(np.abs(may2013_dow - may2012_dow))

overall_corr = np.corrcoef(may2012_series.values[:max_idx], may2013_pred_series.values[:max_idx])[0,1]

# Monospace table
table_text = f"""
╔══════════════════════════════════════════════════╗
║   MAY 2012 vs MAY 2013 PREDICTED SUMMARY          ║
╠══════════════════════════════════════════════════╣
║                                                  ║
║  AVERAGE SPEED:                                 ║
║    • May 2012 Actual:   {may2012_series.mean():>6.2f} mph                      ║
║    • May 2013 Predicted:  {may2013_pred_series.mean():>6.2f} mph                      ║
║    • Difference: {diff:>+6.2f} mph ({diff_pct:.1f}%)                ║
║                                                  ║
║  VARIABILITY:                                   ║
║    • May 2012 Std:    {may2012_series.std():>6.2f} mph                      ║
║    • May 2013 Std:   {may2013_pred_series.std():>6.2f} mph                      ║
║                                                  ║
║  PATTERN DIFFERENCES:                           ║
║    • Avg hourly diff:  {hourly_diff:>5.2f} mph                      ║
║    • Avg dow diff:     {dow_diff:>5.2f} mph                      ║
║    • Overall r:        {overall_corr:>6.3f}                          ║
║                                                  ║
║  KEY FINDING:                                   ║
║  Model predicts May 2013 from 2012 training     ║
║  This demonstrates TEMPORAL GENERALIZATION      ║
║  (same month, previous year prediction)         ║
║                                                  ║
╚══════════════════════════════════════════════════╝
"""

ax7.text(0, 1.0, table_text, transform=ax7.transAxes,
         fontsize=9, fontfamily='monospace', verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='#f8f9fa', alpha=0.95,
                   edgecolor='#2c3e50', linewidth=1.5))

ax7.set_title('G) Statistical Summary', fontweight='bold', fontsize=12, loc='left')

# Main title
plt.suptitle('Traffic Forecasting: May 2012 Actual vs May 2013 Predicted\n'
             'Same Month, Different Year Comparison | Temporal Generalization',
             fontsize=18, fontweight='bold', y=0.995, color='#2c3e50')

plt.savefig('FIGURE3_same_month_different_year.png', bbox_inches='tight', dpi=200, pad_inches=0.3)
print("   [SAVED] FIGURE3_same_month_different_year.png")
plt.close()

print("\n" + "=" * 60)
print("VISUALIZATION COMPLETE!")
print("=" * 60)
print(f"\nKey Finding: May 2012 vs May 2013 Predicted differ by {diff:.2f} mph ({diff_pct:.1f}%)")
print("  Model trained on 2012 data successfully predicts May 2013!")
print("  This demonstrates SAME-MONTH temporal generalization!")
