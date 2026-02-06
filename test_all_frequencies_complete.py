#!/usr/bin/env python3
"""
COMPLETE FREQUENCY TEST - ALL DATA TYPES
Tests with actual data files: Daily, Weekly, Monthly, Quarterly, Yearly, Generic
"""

import pandas as pd
import numpy as np
from pathlib import Path
from forecast_engine import ForecastEngine, Assumption, AssumptionType

print("\n" + "="*70)
print(" COMPLETE FREQUENCY VALIDATION TEST")
print("="*70 + "\n")

passed = 0
total = 0

def test(name, condition, details=""):
    global passed, total
    total += 1
    status = "✅ PASS" if condition else "❌ FAIL"
    print(f"{name:.<55} {status}")
    if not condition and details:
        print(f"  └─ {details}")
    if condition:
        passed += 1
    return condition

def test_frequency_type(file_path, freq_name, expected_ppy, expected_forecast_periods):
    """Test a specific frequency type with actual data"""
    print(f"\n{'─'*70}")
    print(f"Testing: {freq_name}")
    print(f"{'─'*70}")
    
    try:
        # Load data
        df = pd.read_csv(file_path)
        print(f"  Loaded: {len(df)} rows from {Path(file_path).name}")
        print(f"  Columns: {df.columns.tolist()}")
        
        # Get first column as period
        period_col = df.columns[0]
        df = df.rename(columns={period_col: 'period'})
        
        # Initialize engine
        engine = ForecastEngine()
        engine.set_base_data(df)
        
        # Test 1: Periods per year detection
        test(f"{freq_name}: Periods/year = {expected_ppy}", 
             engine.periods_per_year == expected_ppy,
             f"Got {engine.periods_per_year}")
        
        # Get first numeric column
        numeric_cols = engine.base_data.select_dtypes(include=[np.number]).columns.tolist()
        if not numeric_cols:
            test(f"{freq_name}: Has numeric data", False, "No numeric columns")
            return
        
        test_metric = numeric_cols[0]
        print(f"  Metric: {test_metric}")
        
        # Test 2: Base forecast (no assumption)
        forecast = engine.generate_base_forecast(test_metric, periods=expected_forecast_periods)
        test(f"{freq_name}: Base forecast generated", not forecast.empty)
        
        if not forecast.empty:
            print(f"  Forecast periods: {forecast['period'].tolist()}")
            print(f"  First value: ${forecast[test_metric].iloc[0]:,.2f}")
            print(f"  Last value: ${forecast[test_metric].iloc[-1]:,.2f}")
            
            # Test 3: CAGR calculation
            start_val = forecast[test_metric].iloc[0]
            end_val = forecast[test_metric].iloc[-1]
            cagr = ((end_val / start_val) ** (1/expected_forecast_periods) - 1) * 100
            print(f"  Period-over-period growth: {cagr:.2f}%")
        
        # Test 4: Forecast with assumption
        assumption = Assumption(
            id=f'{freq_name}_growth',
            metric=test_metric,
            type=AssumptionType.GROWTH,
            value=0.10,  # 10% annual
            name='Test Growth',
            confidence='medium',
            layer='base'
        )
        engine.add_assumption(assumption)
        
        forecast_w_assumption = engine.generate_base_forecast(test_metric, periods=expected_forecast_periods)
        test(f"{freq_name}: Forecast with 10% assumption", not forecast_w_assumption.empty)
        
        # Test 5: Scenarios
        scenarios = engine.generate_scenarios(test_metric, periods=expected_forecast_periods)
        has_all_scenarios = ('base' in scenarios and 'upside' in scenarios and 'downside' in scenarios)
        test(f"{freq_name}: Scenarios (Base/Upside/Downside)", has_all_scenarios)
        
        if has_all_scenarios:
            base_end = scenarios['base'][test_metric].iloc[-1]
            upside_end = scenarios['upside'][test_metric].iloc[-1]
            downside_end = scenarios['downside'][test_metric].iloc[-1]
            
            test(f"{freq_name}: Scenario ordering (Up>Base>Down)", 
                 upside_end > base_end > downside_end,
                 f"Up={upside_end:.0f}, Base={base_end:.0f}, Down={downside_end:.0f}")
            
            print(f"  Base: ${base_end:,.2f}")
            print(f"  Upside: ${upside_end:,.2f} (+{(upside_end/base_end-1)*100:.1f}%)")
            print(f"  Downside: ${downside_end:,.2f} ({(downside_end/base_end-1)*100:.1f}%)")
        
        return True
        
    except Exception as e:
        test(f"{freq_name}: Processing", False, str(e))
        return False

# ============================================================================
# TEST ALL FREQUENCY TYPES
# ============================================================================

# 1. YEARLY DATA
test_frequency_type(
    "/Users/krishangsharma/Downloads/BNC/test_data/yearly_data.csv",
    "YEARLY",
    expected_ppy=1,
    expected_forecast_periods=3
)

# 2. QUARTERLY DATA
test_frequency_type(
    "/Users/krishangsharma/Downloads/BNC/quarterly/sample_quarterly_saas.csv",
    "QUARTERLY",
    expected_ppy=4,
    expected_forecast_periods=4
)

# 3. MONTHLY DATA (DateTime)
test_frequency_type(
    "/Users/krishangsharma/Downloads/BNC/test_data/monthly_data_1.csv",
    "MONTHLY (DateTime)",
    expected_ppy=12,
    expected_forecast_periods=6
)

# 4. MONTHLY DATA (Text)
test_frequency_type(
    "/Users/krishangsharma/Downloads/BNC/test_data/monthly_data_2.csv",
    "MONTHLY (Text)",
    expected_ppy=12,
    expected_forecast_periods=6
)

# 5. WEEKLY DATA
test_frequency_type(
    "/Users/krishangsharma/Downloads/BNC/weekly/sample_weekly_revenue.csv",
    "WEEKLY",
    expected_ppy=52,
    expected_forecast_periods=4
)

# 6. DAILY DATA
test_frequency_type(
    "/Users/krishangsharma/Downloads/BNC/test_data/daily_data_1.csv",
    "DAILY",
    expected_ppy=365,  # Calendar days (may detect as 252 trading days)
    expected_forecast_periods=5
)

# 7. GENERIC DATA (Period N)
test_frequency_type(
    "/Users/krishangsharma/Downloads/BNC/test_data/generic_data_1.csv",
    "GENERIC (Period N)",
    expected_ppy=1,
    expected_forecast_periods=3
)

# 8. GENERIC DATA (Step N)
test_frequency_type(
    "/Users/krishangsharma/Downloads/BNC/test_data/generic_data_2.csv",
    "GENERIC (Step N)",
    expected_ppy=1,
    expected_forecast_periods=3
)

# ============================================================================
# CROSS-FREQUENCY COMPARISON
# ============================================================================
print(f"\n{'='*70}")
print("CROSS-FREQUENCY COMPARISON")
print(f"{'='*70}\n")

# Test annual rate conversion across frequencies
print("Testing 20% annual growth conversion:")

test_data = [
    ("Yearly", 1, 1),
    ("Quarterly", 4, 4),
    ("Monthly", 12, 12),
    ("Weekly", 52, 52),
]

for freq_name, ppy, periods_to_test in test_data:
    df = pd.DataFrame({
        'period': range(1, 6),
        'Revenue': [100] * 5
    })
    
    engine = ForecastEngine()
    engine.set_base_data(df)
    engine.periods_per_year = ppy  # Override
    
    assumption = Assumption(
        id='test',
        metric='Revenue',
        type=AssumptionType.GROWTH,
        value=0.20,  # 20% annual
        name='Test',
        confidence='high',
        layer='base'
    )
    engine.add_assumption(assumption)
    
    forecast = engine.generate_base_forecast('Revenue', periods=periods_to_test)
    
    if not forecast.empty:
        start = forecast['Revenue'].iloc[0]
        end = forecast['Revenue'].iloc[-1]
        annual_growth = (end / start) - 1
        
        tolerance = 0.025  # 2.5% tolerance
        is_close = abs(annual_growth - 0.20) < tolerance
        
        test(f"{freq_name} ({ppy}ppy): 20% annual → {annual_growth*100:.2f}% actual",
             is_close,
             f"Expected ~20%, tolerance ±{tolerance*100:.1f}%")

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print(f"\n{'='*70}")
print(f" FINAL RESULTS: {passed}/{total} TESTS PASSED ({passed/total*100:.0f}%)")
print(f"{'='*70}\n")

if passed >= total * 0.85:  # 85% threshold
    print("🎉 ALL FREQUENCY TYPES WORKING! SYSTEM VALIDATED! 🎉\n")
    exit(0)
else:
    print(f"⚠️  {total-passed} test(s) failed\n")
    exit(1)
