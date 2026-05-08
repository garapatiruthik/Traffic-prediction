#!/bin/bash
# VERIFICATION SCRIPT - Check for double-scaling bug (793 mph values)

echo "=========================================================================="
echo "CHECKING FOR DOUBLE-SCALING BUG IN MONTH-AHEAD PREDICTIONS"
echo "=========================================================================="
echo ""

# Function to check if values are realistic (60-80 mph range for LA traffic)
check_realistic() {
    local file=$1
    local label=$2
    
    if [ ! -f "$file" ]; then
        echo "✗ $file: FILE MISSING"
        return 1
    fi
    
    # Get mean from CSV (assuming 'predicted_mean' column exists)
    # We'll use Python to compute
    mean_val=$(python -c "
import pandas as pd
df = pd.read_csv('$file')
if 'predicted_mean' in df.columns:
    print(df['predicted_mean'].mean())
elif 'actual' in df.columns:
    print(df['actual'].mean())
else:
    print('N/A')
" 2>/dev/null)
    
    if [ "$mean_val" = "N/A" ]; then
        echo "? $file: Cannot compute mean"
        return 2
    fi
    
    # Check if value is realistic (between 0 and 150 mph)
    # LA traffic typically 50-70 mph, max 80 mph
    if (( $(echo "$mean_val > 100" | bc -l) )); then
        echo "✗ $file: MEAN = ${mean_val} mph - IMPOSSIBLE! (Double-scaling bug)"
        return 1
    elif (( $(echo "$mean_val < 0" | bc -l) )); then
        echo "✗ $file: MEAN = ${mean_val} mph - NEGATIVE! (Bug)"
        return 1
    else
        echo "✓ $file: Mean = ${mean_val} mph (realistic)"
        return 0
    fi
}

echo "Checking May 2013 predictions..."
check_realistic "mamba_predictions_may2013.csv" "May 2013"

echo ""
echo "Checking June 2013 predictions..."
check_realistic "mamba_predictions_jun2013.csv" "June 2013"

echo ""
echo "Checking May 2012 baseline (should be ~60-65 mph)..."
check_realistic "mamba_predictions_may2012.csv" "May 2012"

echo ""
echo "=========================================================================="
echo "DETAILED VALUE CHECK"
echo "=========================================================================="

if [ -f "month_ahead_comparison.csv" ]; then
    echo ""
    echo "Contents of month_ahead_comparison.csv:"
    cat month_ahead_comparison.csv
    echo ""
    
    # Extract the Mean Actual Speed and Mean Predicted Speed for May_2013_Pred
    may2013_actual=$(python -c "
import pandas as pd
df = pd.read_csv('month_ahead_comparison.csv')
row = df[df['Metric'] == 'Mean Actual Speed']['May_2013_Pred'].values
print(row[0] if len(row) > 0 else 'N/A')
" 2>/dev/null)
    
    may2013_pred=$(python -c "
import pandas as pd
df = pd.read_csv('month_ahead_comparison.csv')
row = df[df['Metric'] == 'Mean Predicted Speed']['May_2013_Pred'].values
print(row[0] if len(row) > 0 else 'N/A')
" 2>/dev/null)
    
    echo "May 2013 row:"
    echo "  Mean Actual Speed: $may2013_actual mph"
    echo "  Mean Predicted Speed: $may2013_pred mph"
    
    # Check for unrealistic values
    if [[ "$may2013_actual" =~ ^[0-9]+$ ]] && [ "$may2013_actual" -gt 100 ]; then
        echo "  ⚠ BUG DETECTED: Actual value > 100 mph (should be ~60)"
    fi
fi

echo ""
echo "=========================================================================="
echo "FIX INSTRUCTIONS (if bug detected):"
echo "=========================================================================="
echo "If you see 'IMPOSSIBLE' or values > 100 mph, the bug is NOT fixed."
echo "Fix: In month_ahead_forecasting.py, find lines:"
echo "  may2013_actual = y_test2 * speed_std + speed_mean  # Line ~457"
echo "  jun2013_actual = y_test3 * speed_std + speed_mean  # Line ~484"
echo ""
echo "Change to:"
echo "  may2013_actual = y_test2  # Already unscaled"
echo "  jun2013_actual = y_test3  # Already unscaled"
echo ""
echo "Then re-run:"
echo "  python month_ahead_forecasting.py"
echo "=========================================================================="
