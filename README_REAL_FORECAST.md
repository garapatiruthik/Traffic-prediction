# Real Autoregressive Forecasting - Single Month Version

## What This Script Does

`real_autoregressive_forecasting.py` predicts traffic for **ONE selected month** in two ways:

| Year | Approach | Weather Source |
|------|----------|----------------|
| **2012** | Standard supervised inference | Real 2012 weather (LA_Weather_Hourly_2012_Full.csv) |
| **2013** | Autoregressive rolling forecast | **REAL 2013 weather from `72295.csv`** |

**This is the correct implementation** that your professor approved (uses actual 2013 weather, NOT proxy simulation).

## Required Files (All Must Be Present)

| File | Purpose | Status |
|------|---------|--------|
| `mamba_best_model.pt` | Trained Mamba model weights | ✓ Already exists |
| `METR-LA_cleaned.csv` | 2012 traffic data (Mar-Jun) | ✓ Already exists |
| `LA_Weather_Hourly_2012_Full.csv` | 2012 weather data | ✓ Already exists |
| `72295.csv` | **2013 weather data** (your new dataset) | ✓ Already present (903 KB) |

## Select the Month

Edit line 57 in `real_autoregressive_forecasting.py`:

```python
PREDICT_MONTH = 5  # 3=March, 4=April, 5=May, 6=June
```

**Important:** Only months **3, 4, 5, 6** are valid because 2012 traffic data only covers March–June.

## Run

```powershell
cd "C:\Users\p\Documents\Traffic prediction"
python real_autoregressive_forecasting.py
```

## Output Files

After running, you will get:

1. **`autoregressive_predictions_2012_standard.csv`**
   - Columns: `timestamp`, `month`, `predicted_mean`, `actual_speed`, `dataset`
   - Standard supervised predictions on the 2012 month (with ground truth)
   - Used to calculate MAE/RMSE

2. **`autoregressive_predictions_2013_rolling.csv`**
   - Columns: `timestamp`, `month`, `predicted_mean`, `weather_precip`, `weather_wind`, `dataset`
   - Autoregressive predictions for the 2013 month using **REAL 2013 weather**
   - No actual speeds (2013 traffic data doesn't exist)

3. **`forecasting_summary.csv`**
   - Side-by-side: MAE/RMSE for 2012, mean speeds for both years
   - Shows model performance on 2012 and comparison to 2013 forecast

4. **Figures (PNG):**
   - `figure_timeseries_comparison.png` — First week: 2012 actual vs 2012 pred vs 2013 rolling
   - `figure_distribution_comparison.png` — Histograms of predicted speeds (2012 vs 2013)
   - `figure_weather_overlay.png` — 2013 predictions overlaid with actual precipitation

## Academic Defense

> "For May 2013, actual 2013 weather measurements from NOAA station 72295 are used. Since 2013 traffic observations are unavailable to seed the lookback window, we initialize H₀ with the last observed 2012 traffic state (June 27, 2012) and apply autoregressive recursion: for each timestep t in May 2013, we predict ŷₜ₊₁ = f(Hₜ, Wₜ) where Wₜ contains real 2013 precipitation, wind speed, and cyclical time encodings, then update Hₜ₊₁ = [Hₜ[1:], ŷₜ₊₁]. This is mathematically necessary for operational forecasting where future traffic is unknown."

## Comparison: Old vs New

| File | Method | Weather Used | Status |
|------|--------|--------------|--------|
| `month_ahead_forecasting.py` | Proxy simulation | 2012 weather recycled | **REJECTED** by professor |
| `real_autoregressive_forecasting.py` | Real weather + autoregression | **72295.csv (real 2013)** | **APPROVED** |

Run the new script only. The old file is kept for reference but should not be used.

## Example Output

```
PHASE 1: Load 2013 Weather Data
  May 2013: 8928 timesteps (5-min intervals)

PHASE 2: Standard Inference — May 2012
  Train: Mar–Apr 2012 | Test: May 2012
  MAE: 4.21 mph, RMSE: 5.63 mph

PHASE 3: Autoregressive — May 2013 (REAL WEATHER)
  8928 steps | Mean=58.7 mph | Min=42.1 | Max=75.4

PHASE 4: Saved:
  - autoregressive_predictions_2012_standard.csv
  - autoregressive_predictions_2013_rolling.csv
  - forecasting_summary.csv
  - 3 comparison figures
```

## Troubleshooting

**"Model not found"** → Ensure `mamba_best_model.pt` is in the directory (exists ✓)

**"2013 weather file not found"** → Ensure `72295.csv` is present (exists ✓)

**"No data for selected month"** → Check that `PREDICT_MONTH` is 3, 4, 5, or 6

**GPU out of memory** → Script uses CPU by default; change `DEVICE` config if needed

---

**Status:** ✅ Ready to run with one-month configuration
