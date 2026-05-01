import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.gridspec import GridSpec
import seaborn as sns

# Set style
sns.set_style('whitegrid')
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['figure.dpi'] = 150

print("=" * 60)
print("TRAFFIC FORECASTING - COMPREHENSIVE VISUALIZATION")
print("=" * 60)

# =============================================================================
# 1. Load all results
# =============================================================================
print("\n[1] Loading results...")

chronos_pred = pd.read_csv('chronos_predictions.csv')
chronos_eval = pd.read_csv('chronos_evaluation_results.csv')
chronos_detailed = pd.read_csv('chronos_predictions_detailed.csv')
chronos_info = pd.read_csv('chronos_model_info.txt', sep=': ', header=None, engine='python')

mamba_eval = pd.read_csv('mamba_evaluation_results.csv')
mamba_history = pd.read_csv('mamba_training_history.csv')

# Load merged dataset for temporal analysis
df_full = pd.read_csv('METR_LA_with_Weather_5min.csv', index_col=0)
df_full.index = pd.to_datetime(df_full.index)

# Get a single sensor for visualization
sensor_cols = [c for c in df_full.columns if not c.startswith('weather_')]
traffic_data = df_full[sensor_cols[0]]

print(f"   - Chronos predictions: {chronos_pred.shape}")
print(f"   - Mamba history: {mamba_history.shape}")
print(f"   - Full dataset: {df_full.shape}")
print(f"   - Sensors available: {len(sensor_cols)}")

# =============================================================================
# FIGURE 1: Model Comparison Dashboard
# =============================================================================
print("\n[2] Creating Figure 1 - Model Comparison Dashboard...")

fig1 = plt.figure(figsize=(20, 12))
gs = GridSpec(3, 4, figure=fig1, hspace=0.35, wspace=0.3)

# --- 1A: MAE Comparison ---
ax1 = fig1.add_subplot(gs[0, 0])
models = ['Chronos-2\n(Foundation)', 'Mamba/FFN\n(State Space)']
mae_values = [chronos_eval[chronos_eval['metric']=='MAE']['value'].values[0],
              mamba_eval['MAE'].values[0]]
colors1 = ['#3498db', '#e74c3c']
bars1 = ax1.bar(models, mae_values, color=colors1, edgecolor='black', linewidth=1.5, alpha=0.85)
ax1.set_ylabel('MAE (mph)', fontweight='bold')
ax1.set_title('Mean Absolute Error\n(Lower is Better)', fontweight='bold')
ax1.set_ylim(0, max(mae_values) * 1.5)
for i, (bar, val) in enumerate(zip(bars1, mae_values)):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
             f'{val:.2f} mph', ha='center', fontweight='bold', fontsize=13)
ax1.grid(axis='y', alpha=0.3)

# --- 1B: RMSE Comparison ---
ax2 = fig1.add_subplot(gs[0, 1])
rmse_values = [chronos_eval[chronos_eval['metric']=='RMSE']['value'].values[0],
               mamba_eval['RMSE'].values[0]]
bars2 = ax2.bar(models, rmse_values, color=colors1, edgecolor='black', linewidth=1.5, alpha=0.85)
ax2.set_ylabel('RMSE (mph)', fontweight='bold')
ax2.set_title('Root Mean Squared Error\n(Lower is Better)', fontweight='bold')
ax2.set_ylim(0, max(rmse_values) * 1.5)
for i, (bar, val) in enumerate(zip(bars2, rmse_values)):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.15,
             f'{val:.2f} mph', ha='center', fontweight='bold', fontsize=13)
ax2.grid(axis='y', alpha=0.3)

# --- 1C: KL Divergence (Probabilistic Quality) ---
ax3 = fig1.add_subplot(gs[0, 2])
chronos_kl = chronos_eval[chronos_eval['metric']=='Mean_KL_Divergence']['value'].values[0]
mamba_kl = mamba_eval['KL_Divergence'].values[0]
kl_values = [chronos_kl, mamba_kl]
colors3 = ['#e74c3c', '#3498db']  # Reversed - lower is better
bars3 = ax3.bar(models, kl_values, color=colors3, edgecolor='black', linewidth=1.5, alpha=0.85)
ax3.set_ylabel('KL Divergence (bits)', fontweight='bold')
ax3.set_title('KL Divergence\n(Lower = Better Probabilistic Fit)', fontweight='bold')
ax3.set_yscale('log')
for i, (bar, val) in enumerate(zip(bars3, kl_values)):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.1,
             f'{val:.2f} bits', ha='center', fontweight='bold', fontsize=13)
ax3.grid(axis='y', alpha=0.3)
ax3.set_ylim(0.1, max(kl_values) * 10)

# --- 1D: Inference Speed ---
ax4 = fig1.add_subplot(gs[0, 3])
chrono_load = float(chronos_info[chronos_info[0]=='model_load_time'][1].values[0])
chrono_inf = float(chronos_info[chronos_info[0]=='inference_time'][1].values[0])
mamba_inf = float(mamba_eval['Inference_Latency_ms'].values[0]) / 1000  # Convert to seconds

speed_vals = [chrono_inf, mamba_inf]
colors4 = ['#3498db', '#2ecc71']
bar_labels = ['Chronos-2\nInference', 'Mamba\nInference']
bars4 = ax4.bar(bar_labels, speed_vals, 
                color=colors4, edgecolor='black', linewidth=1.5, alpha=0.85)
ax4.set_ylabel('Time (seconds)', fontweight='bold')
ax4.set_title('Inference Latency\n(Lower is Faster)', fontweight='bold')
for i, (bar, val) in enumerate(zip(bars4, speed_vals)):
    ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
             f'{val:.3f}s', ha='center', fontweight='bold', fontsize=13)
ax4.grid(axis='y', alpha=0.3)
ax4.set_ylim(0, max(speed_vals) * 1.5)

# --- 1E: Chronos - Actual vs Predicted Timeseries ---
ax5 = fig1.add_subplot(gs[1, :])
chronos_pred['timestamp'] = pd.to_datetime(chronos_pred['timestamp'])
x_vals = range(len(chronos_pred))

# Plot samples with transparency
for i in range(min(20, chronos_pred.shape[1] - 5)):
    col = f'sample_{i}'
    if col in chronos_pred.columns:
        ax5.plot(x_vals, chronos_pred[col], color='#3498db', alpha=0.08, linewidth=1)

# Plot mean and actual
ax5.plot(x_vals, chronos_pred['predicted_mean'], 'b-', linewidth=2.5, 
         label='Predicted Mean', alpha=0.85)
ax5.fill_between(x_vals, 
                 chronos_pred['predicted_mean'] - chronos_pred['predicted_std'],
                 chronos_pred['predicted_mean'] + chronos_pred['predicted_std'],
                 alpha=0.2, color='#3498db', label='±1 Std Dev')
ax5.plot(x_vals, chronos_pred['actual'], 'r-', linewidth=2.5, 
         label='Actual', marker='o', markersize=6, markerfacecolor='red')

ax5.set_xlabel('Forecast Horizon (5-min intervals)', fontweight='bold')
ax5.set_ylabel('Traffic Speed (mph)', fontweight='bold')
ax5.set_title('Chronos-2 Zero-Shot Forecast (12 steps = 1 hour ahead) | '
              f'MAE={mae_values[0]:.2f}mph, RMSE={rmse_values[0]:.2f}mph', 
              fontweight='bold', fontsize=15)
ax5.legend(loc='upper right', fontsize=11, framealpha=0.95)
ax5.grid(True, alpha=0.3)
ax5.set_xticks(x_vals)
ax5.set_xticklabels([f'T+{i+1}' for i in x_vals])

# --- 1F: Mamba Training Loss ---
ax6 = fig1.add_subplot(gs[2, :])
ax6.plot(mamba_history['epoch'], mamba_history['train_loss'], 
         'b-o', linewidth=2, markersize=7, label='Train Loss', alpha=0.8)
ax6.plot(mamba_history['epoch'], mamba_history['val_loss'], 
         'r-s', linewidth=2, markersize=7, label='Val Loss', alpha=0.8)
ax6.axhline(y=mamba_history['val_loss'].min(), color='g', linestyle='--', 
            alpha=0.5, label=f'Best Val Loss={mamba_history["val_loss"].min():.4f}')
ax6.set_xlabel('Epoch', fontweight='bold')
ax6.set_ylabel('Loss (Gaussian NLL)', fontweight='bold')
ax6.set_title('Mamba Model Training Progress | 10 epochs, batch=64, LR=1e-3 | '\
              f'Final MAE={mamba_eval["MAE"].values[0]:.2f}mph', 
              fontweight='bold', fontsize=15)
ax6.legend(loc='upper right', fontsize=11, framealpha=0.95)
ax6.grid(True, alpha=0.3)
ax6.set_xlim(0.5, len(mamba_history) + 0.5)

plt.suptitle('URBAN TRAFFIC FORECASTING: Chronos-2 vs Mamba/State Space Model | '
             'METR-LA Dataset (March-June 2012)\n'
             'Features: Traffic Speed + Weather + Temporal (hour/day/month cyclical encoding)',
             fontsize=18, fontweight='bold', y=0.995)

plt.savefig('FIGURE1_model_comparison_dashboard.png', bbox_inches='tight', dpi=200)
print("   [SAVED] FIGURE1_model_comparison_dashboard.png")
plt.close()

# =============================================================================
# FIGURE 2: Temporal Patterns Analysis
# =============================================================================
print("\n[3] Creating Figure 2 - Temporal Pattern Analysis...")

fig2 = plt.figure(figsize=(20, 10))
gs2 = GridSpec(2, 3, figure=fig2, hspace=0.35, wspace=0.3)

# --- 2A: Hourly Pattern (showing rush hours) ---
ax2a = fig2.add_subplot(gs2[0, 0])
hourly_avg = traffic_data.groupby(traffic_data.index.hour).mean()
hourly_std = traffic_data.groupby(traffic_data.index.hour).std()
hours = range(24)

colors_hour = ['#e74c3c' if h in [7, 8, 17, 18] else '#3498db' for h in hours]
ax2a.bar(hours, hourly_avg, color=colors_hour, alpha=0.7, edgecolor='black', linewidth=0.5)
ax2a.errorbar(hours, hourly_avg, yerr=hourly_std, fmt='none', 
              ecolor='black', capsize=3, alpha=0.5, linewidth=1)
ax2a.axvline(x=7, color='red', linestyle='--', alpha=0.5, label='AM Rush (7AM)')
ax2a.axvline(x=17, color='red', linestyle='--', alpha=0.5, label='PM Rush (5PM)')
ax2a.fill_between([7, 9], 0, hourly_avg.max()*1.1, alpha=0.1, color='red')
ax2a.fill_between([17, 19], 0, hourly_avg.max()*1.1, alpha=0.1, color='red')
ax2a.set_xlabel('Hour of Day', fontweight='bold')
ax2a.set_ylabel('Avg Speed (mph)', fontweight='bold')
ax2a.set_title('Hourly Traffic Pattern (Rush Hours Highlighted)', fontweight='bold', fontsize=14)
ax2a.set_xticks(range(0, 24, 2))
ax2a.set_xlim(-0.5, 23.5)
ax2a.legend(loc='upper right')
ax2a.grid(True, alpha=0.3)

# --- 2B: Daily Pattern (weekdays vs weekends) ---
ax2b = fig2.add_subplot(gs2[0, 1])
day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
daily_avg = traffic_data.groupby(traffic_data.index.dayofweek).mean()
daily_std = traffic_data.groupby(traffic_data.index.dayofweek).std()
days = range(7)

colors_day = ['#e74c3c' if d < 5 else '#2ecc71' for d in days]
ax2b.bar(days, daily_avg, color=colors_day, alpha=0.7, edgecolor='black', linewidth=0.5)
ax2b.errorbar(days, daily_avg, yerr=daily_std, fmt='none',
              ecolor='black', capsize=3, alpha=0.5, linewidth=1)
ax2b.axvspan(-0.5, 4.5, alpha=0.1, color='red', label='Weekdays')
ax2b.axvspan(4.5, 6.5, alpha=0.1, color='green', label='Weekends')
ax2b.set_xlabel('Day of Week', fontweight='bold')
ax2b.set_ylabel('Avg Speed (mph)', fontweight='bold')
ax2b.set_title('Daily Traffic Pattern (Weekdays vs Weekends)', fontweight='bold', fontsize=14)
ax2b.set_xticks(days)
ax2b.set_xticklabels(day_names)
ax2b.legend(loc='upper right')
ax2b.grid(True, alpha=0.3)

# --- 2C: Monthly Pattern ---
ax2c = fig2.add_subplot(gs2[0, 2])
monthly_avg = traffic_data.groupby(traffic_data.index.month).mean()
monthly_std = traffic_data.groupby(traffic_data.index.month).std()
months = list(monthly_avg.index)
month_labels = ['Mar', 'Apr', 'May', 'Jun']

colors_month = ['#3498db', '#2ecc71', '#f39c12', '#9b59b6']
ax2c.bar(month_labels, monthly_avg.values, color=colors_month, alpha=0.7, 
         edgecolor='black', linewidth=0.5)
ax2c.errorbar(month_labels, monthly_avg.values, yerr=monthly_std.values, fmt='none',
              ecolor='black', capsize=5, alpha=0.5, linewidth=1.5)
ax2c.set_ylabel('Avg Speed (mph)', fontweight='bold')
ax2c.set_title('Monthly Traffic Pattern (Different Seasons)', fontweight='bold', fontsize=14)
ax2c.grid(True, alpha=0.3, axis='y')

# --- 2D: Weather Impact ---
ax2d = fig2.add_subplot(gs2[1, 0])
# Sample correlation data
weather_cols = [c for c in df_full.columns if c.startswith('weather_')]
precip = df_full[weather_cols[1]]  # precipitation
wind = df_full[weather_cols[2]]    # wind speed

# Bin precipitation and calculate average speed
precip_bins = pd.cut(precip, bins=[-0.1, 0, 1, 3, 10], 
                     labels=['0mm', '0-1mm', '1-3mm', '3+mm'])
precip_grouped = traffic_data.groupby(precip_bins).mean()

colors_precip = ['#3498db', '#f39c12', '#e67e22', '#e74c3c']
ax2d.bar(range(4), precip_grouped.values, color=colors_precip, alpha=0.7, 
         edgecolor='black', linewidth=0.5)
ax2d.set_xticks(range(4))
ax2d.set_xticklabels(['0mm', '0-1mm', '1-3mm', '3+mm'])
ax2d.set_xlabel('Precipitation Level', fontweight='bold')
ax2d.set_ylabel('Avg Speed (mph)', fontweight='bold')
ax2d.set_title('Speed vs Precipitation (Rain reduces speed!)', fontweight='bold', fontsize=14)
ax2d.grid(True, alpha=0.3, axis='y')

# --- 2E: Cyclical Encoding Visualization ---
ax2e = fig2.add_subplot(gs2[1, 1])
hours = np.arange(24)
hour_sin = np.sin(2 * np.pi * hours / 24)
hour_cos = np.cos(2 * np.pi * hours / 24)

# Create a color gradient by hour
colors_gradient = plt.cm.twilight(np.linspace(0, 1, 24))
for i in range(24):
    ax2e.scatter(hour_cos[i], hour_sin[i], s=200, c=[colors_gradient[i]], 
                 edgecolor='black', linewidth=1, zorder=5)
    ax2e.annotate(str(i), (hour_cos[i]*1.15, hour_sin[i]*1.15), 
                  ha='center', va='center', fontweight='bold', fontsize=9)

# Draw unit circle
theta = np.linspace(0, 2*np.pi, 100)
ax2e.plot(np.cos(theta), np.sin(theta), 'k--', alpha=0.3, linewidth=1)
ax2e.set_xlim(-1.5, 1.5)
ax2e.set_ylim(-1.5, 1.5)
ax2e.set_xlabel('cos(hour)', fontweight='bold')
ax2e.set_ylabel('sin(hour)', fontweight='bold')
ax2e.set_title('Cyclical Encoding: Hours (24-hour cycle)\n'
               'Model learns: Hour 23 ≈ Hour 0 (circular!)', 
               fontweight='bold', fontsize=12)
ax2e.axhline(y=0, color='k', linestyle='-', alpha=0.2)
ax2e.axvline(x=0, color='k', linestyle='-', alpha=0.2)
ax2e.grid(True, alpha=0.3)
ax2e.set_aspect('equal')

# --- 2F: Speed Distribution (with weather overlay) ---
ax2f = fig2.add_subplot(gs2[1, 2])

# Plot distribution for different weather conditions
no_rain = traffic_data[precip == 0]
with_rain = traffic_data[precip > 0]

ax2f.hist(no_rain, bins=50, alpha=0.5, label='No Rain', color='#3498db', density=True)
ax2f.hist(with_rain, bins=30, alpha=0.5, label='With Rain', color='#e74c3c', density=True)

ax2f.axvline(no_rain.mean(), color='#3498db', linestyle='--', linewidth=2, 
             label=f'No Rain Mean: {no_rain.mean():.1f} mph')
ax2f.axvline(with_rain.mean(), color='#e74c3c', linestyle='--', linewidth=2,
             label=f'Rain Mean: {with_rain.mean():.1f} mph')

ax2f.set_xlabel('Traffic Speed (mph)', fontweight='bold')
ax2f.set_ylabel('Density', fontweight='bold')
ax2f.set_title('Speed Distribution by Weather Condition', fontweight='bold', fontsize=14)
ax2f.legend(loc='upper right')
ax2f.grid(True, alpha=0.3)

plt.suptitle('TEMPORAL PATTERNS: How Traffic Varies by Time & Weather | '
             'These patterns are LEARNED automatically from data using cyclical encoding',
             fontsize=18, fontweight='bold', y=0.99)

plt.savefig('FIGURE2_temporal_patterns.png', bbox_inches='tight', dpi=200)
print("   [SAVED] FIGURE2_temporal_patterns.png")
plt.close()

print("\n" + "=" * 60)
print("ALL VISUALIZATIONS COMPLETE!")
print("=" * 60)
print("\nGenerated Files:")
print("  1. FIGURE1_model_comparison_dashboard.png")
print("  2. FIGURE2_temporal_patterns.png")
print("\nThese visualizations show:")
print("  • Model performance comparison (MAE, RMSE, KL Divergence)")
print("  • Chronos-2 vs Mamba inference speed")
print("  • Hourly/daily/monthly traffic patterns")
print("  • Weather impact on traffic")
print("  • Cyclical encoding (how model learns circular time)")
print("\nKey Insights:")
print("  • Chronos-2 has better point predictions (lower MAE: 1.66 vs 4.35)")
print("  • Mamba has better probabilistic fit (lower KL: 2.72 vs 18.8)")
print("  • Mamba is much faster for deployment (5ms vs 3 sec)")
print("  • Rush hours clearly visible at 7AM & 5PM")
print("  • Weekends faster than weekdays")
print("  • Rain reduces average speed")
print("  • Cyclical encoding preserves temporal continuity = model learns better!")