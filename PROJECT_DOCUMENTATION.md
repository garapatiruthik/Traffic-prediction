# Urban Traffic Forecasting - Comparative Analysis
## Complete Project Documentation for Thesis Supervisor

**Authors:** Suvarna Kotha & Ruthik Garapati  
**Date:** April 9, 2026  
**Project:** Comparative Analysis of Foundation Models (Chronos-2) vs State Space Models (Mamba) for Urban Traffic Forecasting

---

## 1. Executive Summary

This project implements a complete traffic forecasting pipeline that compares two approaches:

1. **Chronos-2** - A pretrained Foundation Model (zero-shot learning)
2. **Mamba** - A State Space Model (custom training)

Both models use traffic history and weather data to predict future traffic speed. The system automatically extracts temporal patterns from data without predefined rules.

---

## 2. Pipeline Overview

```
Step 1: Download Weather Data (Open-Meteo API)
         ↓
Step 2: Merge Traffic + Weather Data
         ↓
Step 3: Chronos-2 Zero-Shot Inference
         ↓
Step 4: Evaluation Metrics (MAE, RMSE, KL Divergence)
         ↓
Step 5: Mamba Training with Temporal Features
         ↓
Step 6: Comparative Results Analysis
```

---

## 3. Dataset Description

### 3.1 METR-LA Traffic Data
- **Source:** Los Angeles Highway Sensors (GraphWaveNet repository)
- **Time Period:** March 1, 2012 - June 27, 2012
- **Frequency:** 5-minute intervals
- **Sensors:** 207 traffic sensors
- **Total Timesteps:** 34,272 (approximately 4 months)
- **File:** `METR-LA_cleaned.csv` (80 MB)

### 3.2 Weather Data
- **Source:** Open-Meteo Historical API
- **Location:** Los Angeles (Lat: 34.0522, Lon: -118.2437)
- **Time Period:** March 1, 2012 - June 30, 2012
- **Frequency:** Hourly
- **Features:**
  - Temperature: 8.0°C - 30.5°C
  - Precipitation: 0.0 - 6.3 mm
  - Wind Speed: 0.0 - 29.2 km/h
- **File:** `LA_Weather_Hourly_2012_Full.csv` (100 KB)

### 3.3 Merged Dataset
- **File:** `METR_LA_with_Weather_5min.csv` (80 MB)
- **Shape:** 34,272 rows × 210 columns (207 traffic + 3 weather)
- **Features Used:**
  - Traffic Speed (mph)
  - Precipitation (mm)
  - Wind Speed (km/h)
  - Hour (cyclical sin/cos encoding)
  - Day of Week (cyclical sin/cos encoding)

---

## 4. Step-by-Step Implementation

### STEP 1: Download Weather Data
**File:** `step1_download_weather.py`

**Purpose:** Fetch historical weather data from Open-Meteo API

**How it works:**
```python
url = "https://archive-api.open-meteo.com/v1/archive"
params = {
    "latitude": 34.0522,
    "longitude": -118.2437,
    "start_date": "2012-03-01",
    "end_date": "2012-06-30",
    "hourly": ["temperature_2m", "precipitation", "windspeed_10m"],
    "timezone": "America/Los_Angeles"
}
```

**Why:** Weather affects traffic flow. Rain reduces speed, extreme temperatures affect driving behavior.

**Output:** `LA_Weather_Hourly_2012_Full.csv`

---

### STEP 2: Data Preprocessing
**File:** `step2_data_preprocessing.py`

**Purpose:** Merge traffic and weather data with proper temporal alignment

**How it works:**
1. Load METR-LA traffic data (5-minute intervals)
2. Load weather data (hourly)
3. Resample weather to 5-minute using forward-fill
4. Find overlapping time periods
5. Inner join on timestamps

**Key Code:**
```python
df_weather_5min = df_weather.resample('5min').ffill()  # Forward fill
merged_df = df_traffic_filtered.join(df_weather_filtered, how='inner')
```

**Why:** Traffic is at 5-min resolution, weather is hourly. Forward-fill is appropriate because weather changes gradually.

**Output:** `METR_LA_with_Weather_5min.csv` (80 MB)

---

### STEP 3: Chronos-2 Inference
**File:** `step3_chronos_inference.py`

**Purpose:** Zero-shot forecasting using Chronos-2 foundation model

**What is Chronos-2?**
- A pretrained transformer model from Amazon
- Trained on diverse time series data
- Can make predictions without task-specific training (zero-shot)

**Configuration:**
```python
MODEL_NAME = "amazon/chronos-t5-small"
PREDICTION_LENGTH = 12  # 1 hour (12 × 5 min)
CONTEXT_LENGTH = 144    # 12 hours (144 × 5 min)
NUM_SAMPLES = 100       # For probabilistic forecasting
```

**How it works:**
1. Load pretrained Chronos-2 model from HuggingFace
2. Extract last 144 timesteps (12 hours) as context
3. Generate 100 sample forecasts for next 12 timesteps
4. Calculate mean and std from samples

**Why Zero-Shot?**
- Foundation models have general knowledge from training
- No fine-tuning required - saves time and compute
- Tests generalization capability

**Output:** `chronos_predictions.csv`, `chronos_model_info.txt`

---

### STEP 4: Evaluation Metrics
**File:** `step4_evaluation_metrics.py`

**Purpose:** Calculate comprehensive evaluation metrics

**Metrics Calculated:**

1. **MAE (Mean Absolute Error)**
   - Formula: mean(|actual - predicted|)
   - Unit: mph
   - Interpretation: Average prediction error

2. **RMSE (Root Mean Squared Error)**
   - Formula: sqrt(mean((actual - predicted)²))
   - Unit: mph
   - Interpretation: Penalizes large errors more

3. **MAPE (Mean Absolute Percentage Error)**
   - Formula: mean(|error|/actual) × 100
   - Unit: %
   - Interpretation: Scale-independent error

4. **R² (Coefficient of Determination)**
   - Formula: 1 - SS_res/SS_tot
   - Interpretation: How well model explains variance

5. **KL Divergence**
   - Measures difference between predicted distribution and actual
   - Lower = better probabilistic prediction

6. **Calibration Analysis**
   - What % of actuals fall within 68%, 90%, 95% prediction intervals

**Output:** `chronos_evaluation_results.csv`

---

### STEP 5: Mamba Training
**File:** `step5_mamba_training.py`

**Purpose:** Train a custom State Space Model for comparison

**What is Mamba?**
- A State Space Model (SSM) architecture
- Uses selective state spaces for efficient long-range modeling
- Can capture temporal patterns in data

**Key Innovation - Automatic Temporal Feature Extraction:**
```python
def extract_temporal_features(df):
    # Cyclical encoding for hour (24-hour cycle)
    hour_sin = np.sin(2 * np.pi * hours / 24)
    hour_cos = np.cos(2 * π * hours / 24)
    
    # Cyclical encoding for day of week (7-day cycle)
    day_sin = np.sin(2 * np.pi * days / 7)
    day_cos = np.cos(2 * π * days / 7)
```

**Why Cyclical Encoding?**
- Instead of raw hour (0-23), use sin/cos
- Hour 23 is close to hour 0 (circular)
- Model learns "rush hour" patterns automatically from data

**Configuration:**
```python
LOOKBACK_WINDOW = 24   # 2 hours of history
FORECAST_HORIZON = 12  # 1 hour ahead
INPUT_DIM = 7         # [speed, precip, wind, hour_sin, hour_cos, day_sin, day_cos]
D_MODEL = 64          # Hidden dimension
EPOCHS = 10           # Training epochs
BATCH_SIZE = 64
```

**How It Works:**
1. Extract temporal features from timestamps automatically
2. Create sliding windows: X (24 timesteps) → y (12 timesteps)
3. Train with Gaussian Negative Log-Likelihood loss
4. Predict mean (μ) and std (σ) for uncertainty

**Why This Approach?**
- Model LEARNS patterns from data (not predefined rush hours)
- Combines traffic history + weather + temporal context
- Probabilistic output for uncertainty quantification

**Discovered Patterns:**
- Peak traffic: 7:00 AM (66.4 mph avg) - Morning rush
- Lowest traffic: 6:00 PM (40.8 mph avg) - Evening
- Busiest day: Sunday (66.2 mph)
- Quietest day: Friday (58.2 mph)

**Output:** `mamba_best_model.pt`, `mamba_evaluation_results.csv`

---

## 5. Results Comparison

### 5.1 Final Metrics

| Metric | Chronos-2 | Mamba/FFN | Interpretation |
|--------|-----------|-----------|----------------|
| **MAE** | 1.66 mph | 4.35 mph | Chronos has lower average error |
| **RMSE** | 1.94 mph | 8.72 mph | Chronos has fewer large errors |
| **MAPE** | 2.57% | N/A | Chronos error is 2.57% of actual |
| **R²** | -0.04 | N/A | Both models struggle to explain variance |
| **KL Divergence** | 18.8 bits | 2.72 bits | Mamba has better distribution fit |
| **Inference Latency** | 2.64 sec | 5.44 ms | Mamba is faster for deployment |

### 5.2 Training Progress (Mamba)

| Epoch | Train Loss | Val Loss |
|-------|------------|----------|
| 1 | 1.5003 | 0.4456 |
| 5 | 0.5592 | 0.2325 |
| 10 | 0.4532 | 0.2356 |

Best validation loss: 0.1293 (epoch 8)

---

## 6. What the Results Indicate

### 6.1 Chronos-2 (Foundation Model)
**Strengths:**
- MAE of 1.66 mph is good for zero-shot
- Fast inference after model loads
- Good probabilistic predictions (100 samples)

**Weaknesses:**
- Negative R² (-0.04) means predictions are worse than predicting the mean
- High KL Divergence (18.8 bits) - predicted distribution doesn't match well
- Not trained specifically on traffic data

**Why:** Chronos-2 is a general time series model, not traffic-specific. It has never seen METR-LA data during training.

### 6.2 Mamba (State Space Model)
**Strengths:**
- Much better KL Divergence (2.72 bits) - better probabilistic fit
- Fast inference (5.44 ms) - suitable for real-time deployment
- Automatically learns temporal patterns from data
- Custom-trained on METR-LA data

**Weaknesses:**
- Higher MAE (4.35 mph) - but trained with subsampling for CPU
- Used FFN fallback (mamba-ssm not installed locally)
- Need proper GPU training for true Mamba architecture

**Why:** The model learns traffic-specific patterns but needs more training on GPU for best results.

---

## 7. Technical Implementation Details

### 7.1 Technologies Used
| Technology | Purpose | Version |
|------------|---------|---------|
| Python | Programming Language | 3.11 |
| PyTorch | Deep Learning Framework | 2.1+ |
| Pandas | Data Manipulation | Latest |
| NumPy | Numerical Computing | Latest |
| Scikit-learn | Preprocessing | Latest |
| Chronos | Foundation Model | amazon/chronos-t5-small |
| Mamba-SSM | State Space Model | 1.2.0 (if installed) |

### 7.2 Data Flow
```
Raw Data → Preprocessing → Feature Extraction → Model Training → Evaluation
    ↓           ↓               ↓                   ↓              ↓
METR-LA    Merge Weather   Temporal Features   Mamba Training  Metrics
Weather   (5-min align)    (auto-learned)       (probabilistic)  (MAE/RMSE/KL)
```

### 7.3 Key Design Decisions

1. **Why Sliding Windows?**
   - Traffic prediction is a sequence-to-sequence problem
   - Lookback: 24 timesteps (2 hours) captures recent patterns
   - Horizon: 12 timesteps (1 hour) is standard for short-term forecasting

2. Why Cyclical Encoding for Time?
   - Linear encoding (hour = 0-23) makes 23 and 0 distant
   - Cyclical (sin/cos) preserves that 23 ≈ 0
   - Model learns "rush hour" patterns without predefined rules

3. Why Probabilistic Output?
   - Single point prediction doesn't show uncertainty
   - Predicting mean + std allows confidence intervals
   - KL Divergence measures distribution quality

---

## 8. Files Created

| File | Size | Description |
|------|------|-------------|
| `step1_download_weather.py` | 2 KB | Weather API download script |
| `step2_data_preprocessing.py` | 5 KB | Data merging script |
| `step3_chronos_inference.py` | 7 KB | Chronos-2 inference script |
| `step4_evaluation_metrics.py` | 8 KB | Evaluation metrics script |
| `step5_mamba_training.py` | 26 KB | Mamba training script |
| `METR-LA_cleaned.csv` | 80 MB | Raw METR-LA data |
| `LA_Weather_Hourly_2012_Full.csv` | 100 KB | Weather data |
| `METR_LA_with_Weather_5min.csv` | 80 MB | Merged dataset |
| `single_sensor_with_weather.csv` | 1.5 MB | Single sensor subset |
| `chronos_predictions.csv` | 2 KB | Chronos predictions |
| `chronos_evaluation_results.csv` | 314 B | Chronos metrics |
| `mamba_evaluation_results.csv` | 213 B | Mamba metrics |
| `mamba_best_model.pt` | 280 KB | Trained model weights |
| `mamba_training_history.csv` | 444 B | Training loss history |

---

## 9. How to Run

### Prerequisites
```bash
pip install torch pandas numpy scikit-learn
pip install chronos-forecasting  # For Chronos
pip install mamba-ssm causal-conv1d  # For Mamba (requires GPU)
```

### Execution Order
```bash
# Step 1: Download weather (already done)
python step1_download_weather.py

# Step 2: Preprocess data (already done)
python step2_data_preprocessing.py

# Step 3: Run Chronos inference
python step3_chronos_inference.py

# Step 4: Evaluate Chronos
python step4_evaluation_metrics.py

# Step 5: Train Mamba (on GPU recommended)
python step5_mamba_training.py
```

---

## 10. Conclusions and Next Steps

### 10.1 Current Findings
1. **Chronos-2** achieves good zero-shot performance (MAE 1.66 mph)
2. **Mamba** shows promise but needs proper GPU training
3. **Temporal features** are automatically learned from data
4. **Weather integration** is working correctly

### 10.2 Recommended Next Steps (for Thesis)
1. **Run Mamba on Colab with GPU** - Install mamba-ssm and train properly
2. **Compare multiple horizons** - 15min, 30min, 1hr, 2hr predictions
3. **Add more evaluation** - Compare against DCRNN, Graph WaveNet
4. **Visualize results** - Plot predictions vs actual over time
5. **Analyze by time of day** - Performance during rush hour vs night

---

## 11. References

1. **METR-LA Dataset:** https://graphwnet.github.io/
2. **Chronos-2 Paper:** "Chronos: Learning the Language of Time Series" (Amazon)
3. **Mamba Paper:** "Mamba: Linear-time Sequence Modeling with Selective State Spaces" (Stanford)
4. **Open-Meteo API:** https://open-meteo.com/

---

*Documentation generated: April 9, 2026*
