"""
CLEAN TEST: Use synthetic data without any events
"""
import pandas as pd
import numpy as np
import sys
sys.path.insert(0, '/Users/krishangsharma/Downloads/BNC')

from forecast_engine import ForecastEngine, Assumption, AssumptionType

print("="*80)
print("CLEAN FREQUENCY TEST (Synthetic Data, No Events)")
print("="*80)

TEST_ANNUAL_RATE = 0.20  # 20% annual
test_results = []

def test_clean_frequency(name, periods_per_year, num_periods, forecast_periods):
    """Test with clean synthetic data"""
    print(f"\n{'='*80}")
    print(f"TEST: {name} ({periods_per_year} periods/year)")
    print(f"{'='*80}")
    
    # Create clean synthetic data - simple linear values to avoid any implicit growth
    if periods_per_year == 52:
        # Weekly
        period_labels = [f"2024-{i//52 + 1:02d}-{(i%52)*7 + 1:02d}" for i in range(num_periods)]
        df = pd.DataFrame({
            'period': period_labels,
            'Value': [100000] * num_periods  # Flat historical values
        })
    elif periods_per_year == 12:
        # Monthly  
        months = ['January', 'February', 'March', 'April', 'May', 'June',
                 'July', 'August', 'September', 'October', 'November', 'December']
        period_labels = [f"2023 {months[i%12]}" if i < 12 else f"2024 {months[i%12]}" for i in range(num_periods)]
        df = pd.DataFrame({
            'period': period_labels,
            'Value': [50000] * num_periods
        })
    elif periods_per_year == 4:
        # Quarterly
        period_labels = [f"Q{(i%4)+1} {2022 + i//4}" for i in range(num_periods)]
        df = pd.DataFrame({
            'period': period_labels,
            'Value': [1000000] * num_periods
        })
    elif periods_per_year == 1:
        # Yearly
        period_labels = list(range(2018, 2018 + num_periods))
        df = pd.DataFrame({
            'period': period_labels,
            'Value': [500000] * num_periods
        })
    else:
        print(f"❌ Unsupported frequency: {periods_per_year}")
        return False
    
    print(f"Historical data: {num_periods} periods (all values equal to remove noise)")
    
    engine = ForecastEngine()
    engine.set_base_data(df)
    
    print(f"Detected frequency: {engine.periods_per_year} periods/year")
    
    # Verify detection
    if engine.periods_per_year != periods_per_year:
        print(f"❌ FAIL: Detection incorrect (expected {periods_per_year}, got {engine.periods_per_year})")
        return False
    
    # Add growth assumption
    assumption = Assumption(
        id=f"test_{name}",
        type=AssumptionType.GROWTH,
        name="Growth",
        metric='Value',
        value=TEST_ANNUAL_RATE,
        confidence='medium',
        source='test'
    )
    engine.add_assumption(assumption)
    
    # Calculate expected period rate
    expected_period_rate = (1 + TEST_ANNUAL_RATE) ** (1/periods_per_year) - 1
    print(f"\nConversion:")
    print(f"  Input: {TEST_ANNUAL_RATE*100}% annual")
    print(f"  Converted to: {expected_period_rate*100:.4f}% per period")
    
    # Generate forecast
    scenarios = engine.generate_scenarios('Value', periods=forecast_periods)
    
    # Check period-over-period consistency
    base = scenarios['base']['Value']
    
    print(f"\nPeriod-over-period growth rates:")
    period_rates = []
    for i in range(min(5, len(base)-1)):  # Show first 5
        rate = (base.iloc[i+1] / base.iloc[i] - 1) * 100
        period_rates.append(rate)
        print(f"  Period {i+1} → {i+2}: {rate:.3f}%")
    
    # Check consistency
    avg_period_rate = np.mean(period_rates)
    rate_consistent = abs(avg_period_rate - expected_period_rate*100) < 0.01
    
    # Calculate annualized CAGR
    start = base.iloc[0]
    end = base.iloc[-1]
    
    num_years = forecast_periods / periods_per_year
    annualized_cagr = ((end / start) ** (1/num_years) - 1) * 100
    
    print(f"\nCAGR Calculation:")
    print(f"  Period 1: {start:,.0f}")
    print(f"  Period {forecast_periods}: {end:,.0f}")
    print(f"  Forecast span: {forecast_periods} periods = {num_years:.2f} years")
    print(f"  Annualized CAGR: {annualized_cagr:.2f}%")
    print(f"  Target: {TEST_ANNUAL_RATE*100}%")
    print(f"  Difference: {annualized_cagr - TEST_ANNUAL_RATE*100:.2f}%")
    
    cagr_correct = abs(annualized_cagr - TEST_ANNUAL_RATE*100) < 0.5
    
    overall_pass = rate_consistent and cagr_correct
    
    if overall_pass:
        print(f"\n✅ PASS")
    else:
        print(f"\n❌ FAIL")
        if not rate_consistent:
            print(f"   Period rate inconsistent: {avg_period_rate:.4f}% vs expected {expected_period_rate*100:.4f}%")
        if not cagr_correct:
            print(f"   CAGR off by {annualized_cagr - TEST_ANNUAL_RATE*100:.2f}%")
    
    test_results.append({
        'name': name,
        'pass': overall_pass,
        'cagr': annualized_cagr,
        'period_rate': avg_period_rate
    })
    
    return overall_pass

# Run tests
test_clean_frequency("Weekly", 52, 52, 26)     # 1 year history, 6mo forecast
test_clean_frequency("Monthly", 12, 24, 12)    # 2 year history, 1yr forecast
test_clean_frequency("Quarterly", 4, 12, 8)    # 3 year history, 2yr forecast
test_clean_frequency("Yearly", 1, 6, 3)        # 6 year history, 3yr forecast

# Summary
print(f"\n{'='*80}")
print("SUMMARY")
print(f"{'='*80}\n")

passed = sum(1 for r in test_results if r['pass'])
total = len(test_results)

print(f"Tests: {passed}/{total} passed\n")

for r in test_results:
    status = "✅ PASS" if r['pass'] else "❌ FAIL"
    print(f"{r['name']:<15} {status:<10} CAGR: {r['cagr']:.2f}%")

if passed == total:
    print(f"\n🎉 ALL TESTS PASSED!")
    print("Annual growth rate standardization working correctly!")
else:
    print(f"\n❌ {total - passed} test(s) failed")
