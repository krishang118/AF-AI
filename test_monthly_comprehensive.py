"""
Comprehensive Monthly Data Test
Tests all month formats: Jan/Feb/Mar, January/February, Jan 2022, 2022 January
"""
import pandas as pd
import numpy as np
import sys
sys.path.insert(0, '/Users/krishangsharma/Downloads/BNC')

from forecast_engine import ForecastEngine, Assumption, AssumptionType

print("="*70)
print("COMPREHENSIVE MONTHLY DATA TEST")
print("="*70)

def test_monthly_format(label, periods_input):
    """Test a specific monthly format"""
    print(f"\n{label}")
    print("-" * 70)
    
    # Create test data
    data = {
        'period': periods_input,
        'Revenue': [100000 + i*5000 for i in range(len(periods_input))]
    }
    df = pd.DataFrame(data)
    
    print(f"Input periods: {periods_input[:3]}...")
    
    engine = ForecastEngine()
    engine.set_base_data(df)
    
    # Check frequency detection
    print(f"✓ Detected frequency: {engine.periods_per_year} periods/year")
    if engine.periods_per_year != 12:
        print(f"  ❌ FAIL: Expected 12, got {engine.periods_per_year}")
        return False
    
    # Test extrapolation
    extrapolated = engine.extrapolate_periods(periods_input, 3)
    print(f"✓ Extrapolation: {periods_input[-1]} → {extrapolated}")
    
    # Add growth assumption (12% annual)
    assumption = Assumption(
        id="test",
        type=AssumptionType.GROWTH,
        name="Growth",
        metric='Revenue',
        value=0.12,  # 12% annual
        confidence='medium',
        source='test'
    )
    engine.add_assumption(assumption)
    
    # Generate forecast
    scenarios = engine.generate_scenarios('Revenue', periods=3)
    
    # Verify growth rate conversion
    # 12% annual should convert to ~0.949% per month: (1.12)^(1/12) - 1
    expected_monthly = (1.12 ** (1/12) - 1) * 100
    
    last_historical = df['Revenue'].iloc[-1]
    first_forecast = scenarios['base']['Revenue'].iloc[0]
    actual_growth = ((first_forecast / last_historical) - 1) * 100
    
    print(f"✓ Last historical: ${last_historical:,.0f}")
    print(f"✓ First forecast: ${first_forecast:,.0f}")
    print(f"✓ Expected monthly growth: {expected_monthly:.3f}%")
    print(f"✓ Actual growth: {actual_growth:.3f}%")
    
    # Allow small tolerance
    if abs(actual_growth - expected_monthly) > 0.01:
        print(f"  ❌ FAIL: Growth rate mismatch")
        return False
    
    print(f"✅ PASS: Monthly data working correctly!")
    return True

# Test 1: Short month names
test1_passed = test_monthly_format(
    "Test 1: Short Month Names (Jan, Feb, Mar)",
    ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
)

# Test 2: Full month names
test2_passed = test_monthly_format(
    "Test 2: Full Month Names (January, February, March)",
    ['January', 'February', 'March', 'April', 'May', 'June', 
     'July', 'August', 'September', 'October', 'November', 'December']
)

# Test 3: Month + Year
test3_passed = test_monthly_format(
    "Test 3: Month + Year (Jan 2023, Feb 2023)",
    ['Jan 2023', 'Feb 2023', 'Mar 2023', 'Apr 2023', 'May 2023', 'Jun 2023',
     'Jul 2023', 'Aug 2023', 'Sep 2023', 'Oct 2023', 'Nov 2023', 'Dec 2023']
)

# Test 4: Year + Month
test4_passed = test_monthly_format(
    "Test 4: Year + Month (2023 January, 2023 February)",
    ['2023 January', '2023 February', '2023 March', '2023 April', '2023 May', '2023 June',
     '2023 July', '2023 August', '2023 September', '2023 October', '2023 November', '2023 December']
)

# Test 5: Year transitions
print("\nTest 5: Year Transitions (Dec → Jan)")
print("-" * 70)
year_transition = ['Oct 2023', 'Nov 2023', 'Dec 2023']
extrapolated = ForecastEngine().extrapolate_periods(year_transition, 3)
expected = ['Jan 2024', 'Feb 2024', 'Mar 2024']
print(f"Input: {year_transition}")
print(f"Output: {extrapolated}")
print(f"Expected: {expected}")
test5_passed = extrapolated == expected
if test5_passed:
    print("✅ PASS: Year transition working!")
else:
    print(f"❌ FAIL: Expected {expected}, got {extrapolated}")

# Summary
print("\n" + "="*70)
print("TEST SUMMARY")
print("="*70)

all_tests = [
    ("Short Month Names", test1_passed),
    ("Full Month Names", test2_passed),
    ("Month + Year", test3_passed),
    ("Year + Month", test4_passed),
    ("Year Transitions", test5_passed)
]

passed = sum(1 for _, p in all_tests if p)
total = len(all_tests)

for name, result in all_tests:
    status = "✅ PASS" if result else "❌ FAIL"
    print(f"{status}: {name}")

print(f"\nTotal: {passed}/{total} tests passed")

if passed == total:
    print("\n🎉 ALL MONTHLY DATA TESTS PASSED!")
    print("✅ Frequency detection: 12 periods/year")
    print("✅ Period extrapolation: Dec → Jan transitions work")
    print("✅ Growth rate conversion: 12% annual → 0.949% monthly")
else:
    print(f"\n❌ {total - passed} TEST(S) FAILED")
    sys.exit(1)
