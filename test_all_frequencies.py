"""
COMPREHENSIVE FREQUENCY TESTING
Verify annual growth rate standardization works for ALL data frequencies
"""
import pandas as pd
import numpy as np
import sys
sys.path.insert(0, '/Users/krishangsharma/Downloads/BNC')

from forecast_engine import ForecastEngine, Assumption, AssumptionType

print("="*80)
print("COMPREHENSIVE ANNUAL GROWTH RATE STANDARDIZATION TEST")
print("="*80)

# Annual growth rate to test with
TEST_ANNUAL_RATE = 0.20  # 20% annual

test_results = []

def test_frequency(name, data, periods_per_year_expected, metric_col, forecast_periods):
    """Test a specific frequency"""
    print(f"\n{'='*80}")
    print(f"TEST: {name}")
    print(f"{'='*80}")
    
    # Load data
    engine = ForecastEngine()
    engine.set_base_data(data)
    
    print(f"Data: {len(data)} periods")
    print(f"Expected frequency: {periods_per_year_expected} periods/year")
    print(f"Detected frequency: {engine.periods_per_year} periods/year")
    
    # Verify detection
    detection_correct = engine.periods_per_year == periods_per_year_expected
    
    # Add assumption
    assumption = Assumption(
        id=f"test_{name}",
        type=AssumptionType.GROWTH,
        name=f"{name} Growth",
        metric=metric_col,
        value=TEST_ANNUAL_RATE,
        confidence='medium',
        source='test'
    )
    engine.add_assumption(assumption)
    
    # Calculate expected period rate
    expected_period_rate = (1 + TEST_ANNUAL_RATE) ** (1/periods_per_year_expected) - 1
    print(f"\nInput: {TEST_ANNUAL_RATE*100}% annual")
    print(f"Expected period rate: {expected_period_rate*100:.4f}%")
    
    # Generate forecast
    scenarios = engine.generate_scenarios(metric_col, periods=forecast_periods)
    
    start = scenarios['base'][metric_col].iloc[0]
    end = scenarios['base'][metric_col].iloc[-1]
    
    # Calculate annualized CAGR
    # CAGR formula: ((end/start)^(periods_per_year/forecast_periods) - 1) * 100
    annualized_cagr = ((end / start) ** (periods_per_year_expected / forecast_periods) - 1) * 100
    
    print(f"\nForecast Results:")
    print(f"  Period 1: {start:,.2f}")
    print(f"  Period {forecast_periods}: {end:,.2f}")
    print(f"  Total growth: {(end/start - 1)*100:.1f}%")
    print(f"  Annualized CAGR: {annualized_cagr:.1f}%")
    
    # Check if CAGR is within 2% of target
    cagr_correct = abs(annualized_cagr - TEST_ANNUAL_RATE*100) < 2.0
    
    overall_pass = detection_correct and cagr_correct
    
    if overall_pass:
        print(f"\n✅ PASS: {name}")
    else:
        print(f"\n❌ FAIL: {name}")
        if not detection_correct:
            print(f"   - Frequency detection failed: expected {periods_per_year_expected}, got {engine.periods_per_year}")
        if not cagr_correct:
            print(f"   - CAGR off by {annualized_cagr - TEST_ANNUAL_RATE*100:.1f}%")
    
    test_results.append({
        'name': name,
        'detection_correct': detection_correct,
        'cagr_correct': cagr_correct,
        'overall_pass': overall_pass,
        'detected_freq': engine.periods_per_year,
        'expected_freq': periods_per_year_expected,
        'annualized_cagr': annualized_cagr
    })
    
    return overall_pass

# ============================================================================
# TEST 1: WEEKLY DATA (52 periods/year)
# ============================================================================
weekly_df = pd.read_csv('/Users/krishangsharma/Downloads/BNC/sample_weekly_revenue.csv')
test_frequency(
    name="Weekly Data",
    data=weekly_df,
    periods_per_year_expected=52,
    metric_col='Weekly_Revenue',
    forecast_periods=21  # ~5 months
)

# ============================================================================
# TEST 2: QUARTERLY DATA (4 periods/year)
# ============================================================================
quarterly_df = pd.read_excel('/Users/krishangsharma/Downloads/BNC/sample_quarterly_retail.xlsx')
test_frequency(
    name="Quarterly Data",
    data=quarterly_df,
    periods_per_year_expected=4,
    metric_col='Units_Sold',
    forecast_periods=8  # 2 years
)

# ============================================================================
# TEST 3: YEARLY DATA (1 period/year)
# ============================================================================
# Create synthetic yearly data
yearly_data = pd.DataFrame({
    'period': [2018, 2019, 2020, 2021, 2022, 2023],
    'Revenue': [1000000, 1050000, 1100000, 1150000, 1200000, 1250000]
})
test_frequency(
    name="Yearly Data",
    data=yearly_data,
    periods_per_year_expected=1,
    metric_col='Revenue',
    forecast_periods=3  # 3 years
)

# ============================================================================
# TEST 4: MONTHLY DATA (12 periods/year)
# ============================================================================
# Create synthetic monthly data with month labels
monthly_data = pd.DataFrame({
    'period': [f"{year} {month}" for year in [2023, 2024] for month in ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']],
    'Sales': np.array([50000, 52000, 54000, 56000, 58000, 60000, 62000, 64000, 66000, 68000, 70000, 72000,
                       74000, 76000, 78000, 80000, 82000, 84000, 86000, 88000, 90000, 92000, 94000, 96000])
})
test_frequency(
    name="Monthly Data",
    data=monthly_data,
    periods_per_year_expected=12,
    metric_col='Sales',
    forecast_periods=12  # 1 year
)

# ============================================================================
# TEST 5: QUARTERLY DATA (Label-based: Q1, Q2, Q3, Q4)
# ============================================================================
quarterly_labels = pd.DataFrame({
    'period': ['Q1 2022', 'Q2 2022', 'Q3 2022', 'Q4 2022', 'Q1 2023', 'Q2 2023', 'Q3 2023', 'Q4 2023'],
    'Metric': [100, 105, 110, 115, 120, 125, 130, 135]
})
test_frequency(
    name="Quarterly Labels (Q1, Q2)",
    data=quarterly_labels,
    periods_per_year_expected=4,
    metric_col='Metric',
    forecast_periods=4  # 1 year
)

# ============================================================================
# SUMMARY
# ============================================================================
print(f"\n{'='*80}")
print("FINAL SUMMARY")
print(f"{'='*80}\n")

passed = sum(1 for r in test_results if r['overall_pass'])
failed = sum(1 for r in test_results if not r['overall_pass'])
total = len(test_results)

print(f"Total Tests: {total}")
print(f"✅ Passed: {passed}")
print(f"❌ Failed: {failed}")
print(f"\nSuccess Rate: {passed}/{total} ({100*passed/total:.0f}%)")

print(f"\n{'Test Name':<30} {'Detection':<12} {'CAGR Check':<12} {'Result':<8}")
print("-"*80)
for r in test_results:
    detection_mark = "✅" if r['detection_correct'] else "❌"
    cagr_mark = "✅" if r['cagr_correct'] else "❌"
    result = "PASS" if r['overall_pass'] else "FAIL"
    print(f"{r['name']:<30} {detection_mark:<12} {cagr_mark:<12} {result:<8}")

if failed > 0:
    print(f"\n❌ SOME TESTS FAILED")
    for r in test_results:
        if not r['overall_pass']:
            print(f"\n  {r['name']}:")
            if not r['detection_correct']:
                print(f"    Frequency: expected {r['expected_freq']}, got {r['detected_freq']}")
            if not r['cagr_correct']:
                print(f"    CAGR: {r['annualized_cagr']:.1f}% (expected ~{TEST_ANNUAL_RATE*100:.0f}%)")
else:
    print(f"\n🎉 ALL TESTS PASSED!")
    print(f"✅ Annual growth rate standardization working correctly for ALL frequencies!")
