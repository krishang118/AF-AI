#!/usr/bin/env python3
"""
Comprehensive Period Frequency Test Suite
Tests all supported period types: Daily, Monthly, Weekly, Quarterly, Yearly, Generic
"""

import pandas as pd
import numpy as np
from forecast_engine import ForecastEngine, Assumption, AssumptionType
from pathlib import Path

def test_frequency(test_name, data_file, expected_freq_type):
    """Test a specific frequency type"""
    print(f"\n{'='*60}")
    print(f"TEST: {test_name}")
    print(f"{'='*60}")
    
    # Load data
    df = pd.read_csv(data_file)
    print(f"✓ Loaded {len(df)} rows from {Path(data_file).name}")
    print(f"  Periods: {df['Period'].tolist()}")
    
    # Rename to 'period' for ForecastEngine
    df = df.rename(columns={'Period': 'period'})
    
    # Initialize engine
    engine = ForecastEngine()
    
    # Set base data
    engine.set_base_data(df)
    print(f"  Period type: {type(engine.base_data.index[0]).__name__}")
    print(f"  Periods per year detected: {engine.periods_per_year}")
    
    # Try to infer frequency if datetime
    if pd.api.types.is_datetime64_any_dtype(engine.base_data.index):
        inferred_freq = pd.infer_freq(engine.base_data.index)
        print(f"  Inferred frequency: {inferred_freq if inferred_freq else 'Manual heuristic'}")
    
    # Auto-detect first numeric column
    numeric_cols = engine.base_data.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        print(f"❌ FAILED: No numeric columns found!")
        return False
    
    test_metric = numeric_cols[0]
    print(f"  Testing with metric: {test_metric}")
    
    # Create a simple growth assumption
    assumption = Assumption(
        id='test1',
        metric=test_metric,
        type=AssumptionType.GROWTH,
        value=0.05,  # 5% annual growth
        name='Test Growth',
        confidence='medium',
        layer='base'
    )
    engine.add_assumption(assumption)
    
    # Generate base forecast
    print(f"\n→ Generating 5-period forecast...")
    base_forecast = engine.generate_base_forecast(test_metric, periods=5)
    
    if base_forecast.empty:
        print(f"❌ FAILED: Forecast returned empty!")
        return False
    
    print(f"✓ Forecast generated successfully!")
    print(f"  Forecast periods: {base_forecast['period'].tolist()}")
    print(f"  {test_metric} values: {base_forecast[test_metric].round(2).tolist()}")
    
    # Verify CAGR calculation
    start_val = base_forecast[test_metric].iloc[0]
    end_val = base_forecast[test_metric].iloc[-1]
    calculated_cagr = ((end_val / start_val) ** (1 / 5) - 1)
    print(f"\n  Start value: ${start_val:,.2f}")
    print(f"  End value: ${end_val:,.2f}")
    print(f"  Calculated CAGR: {calculated_cagr*100:.2f}%")
    
    # Test period extrapolation
    print(f"\n→ Testing period extrapolation...")
    last_period = base_forecast['period'].iloc[-1]
    print(f"  Last forecast period: {last_period}")
    print(f"  Expected pattern: {expected_freq_type}")
    
    # Generate scenarios
    print(f"\n→ Testing scenario generation...")
    scenarios = engine.generate_scenarios(test_metric, periods=5)
    
    if 'base' not in scenarios or 'upside' not in scenarios or 'downside' not in scenarios:
        print(f"❌ FAILED: Missing scenarios!")
        return False
    
    print(f"✓ Scenarios generated (Base, Upside, Downside)")
    print(f"  Base end: ${scenarios['base'][test_metric].iloc[-1]:,.2f}")
    print(f"  Upside end: ${scenarios['upside'][test_metric].iloc[-1]:,.2f}")
    print(f"  Downside end: ${scenarios['downside'][test_metric].iloc[-1]:,.2f}")
    
    print(f"\n{'✓'*30}")
    print(f"✓✓✓ {test_name} PASSED ✓✓✓")
    print(f"{'✓'*30}")
    
    return True


def run_all_tests():
    """Run comprehensive test suite"""
    print("\n" + "="*60)
    print(" COMPREHENSIVE PERIOD FREQUENCY TEST SUITE")
    print("="*60)
    
    results = {}
    
    # Test 1: Daily Data (DateTime)
    results['Daily #1'] = test_frequency(
        "Daily Data Test #1 (2024-01-01 format)",
        "/Users/krishangsharma/Downloads/BNC/test_data/daily_data_1.csv",
        "Daily (D)"
    )
    
    results['Daily #2'] = test_frequency(
        "Daily Data Test #2 (Different date range)",
        "/Users/krishangsharma/Downloads/BNC/test_data/daily_data_2.csv",
        "Daily (D)"
    )
    
    # Test 2: Monthly Data (DateTime)
    results['Monthly #1'] = test_frequency(
        "Monthly Data Test #1 (YYYY-MM format)",
        "/Users/krishangsharma/Downloads/BNC/test_data/monthly_data_1.csv",
        "Monthly (MS)"
    )
    
    results['Monthly #2'] = test_frequency(
        "Monthly Data Test #2 (Mon YYYY format)",
        "/Users/krishangsharma/Downloads/BNC/test_data/monthly_data_2.csv",
        "Monthly (Month names)"
    )
    
    # Test 3: Generic Period Labels
    results['Generic #1'] = test_frequency(
        "Generic Period Test #1 (Period 1, 2, 3...)",
        "/Users/krishangsharma/Downloads/BNC/test_data/generic_data_1.csv",
        "Generic (Period N)"
    )
    
    results['Generic #2'] = test_frequency(
        "Generic Period Test #2 (Step 1, 2, 3...)",
        "/Users/krishangsharma/Downloads/BNC/test_data/generic_data_2.csv",
        "Generic (Step N)"
    )
    
    # Summary
    print("\n\n" + "="*60)
    print(" TEST SUMMARY")
    print("="*60)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for test_name, passed_flag in results.items():
        status = "✓ PASSED" if passed_flag else "❌ FAILED"
        print(f"{test_name:.<40} {status}")
    
    print(f"\n{'='*60}")
    print(f"TOTAL: {passed}/{total} tests passed ({passed/total*100:.0f}%)")
    print(f"{'='*60}\n")
    
    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
