"""
COMPREHENSIVE CODE AUDIT SCRIPT
Automated systematic check of all critical functions
"""
import sys
sys.path.insert(0, '/Users/krishangsharma/Downloads/BNC')

from forecast_engine import ForecastEngine
import pandas as pd

print("="*70)
print("COMPREHENSIVE PRODUCTION-READINESS AUDIT")
print("="*70)

audit_results = []

def test(name, condition, details=""):
    """Helper to track audit results"""
    status = "✅ PASS" if condition else "❌ FAIL"
    audit_results.append({'name': name, 'pass': condition, 'details': details})
    print(f"{status}: {name}")
    if details and not condition:
        print(f"  Details: {details}")
    return condition

# Initialize engine
engine = ForecastEngine()

print("\n" + "="*70)
print("1. PERIOD DETECTION ORDER VERIFICATION")
print("="*70)

# Test that quarters come BEFORE pattern matching
test1_input = ['Q1 2022', 'Q2 2022', 'Q3 2022']
test1_result = engine.extrapolate_periods(test1_input, 3)
test1_expected = ['Q4 2022', 'Q1 2023', 'Q2 2023']
test("Quarters not intercepted by pattern matching", 
     test1_result == test1_expected,
     f"Got {test1_result}, expected {test1_expected}")

# Test that months come BEFORE pattern matching  
test2_input = ['Jan 2022', 'Feb 2022', 'Mar 2022']
test2_result = engine.extrapolate_periods(test2_input, 3)
test2_expected = ['Apr 2022', 'May 2022', 'Jun 2022']
test("Months not intercepted by pattern matching",
     test2_result == test2_expected,
     f"Got {test2_result}, expected {test2_expected}")

# Test pattern matching still works for valid cases
test3_input = ['Week 1', 'Week 2', 'Week 3']
test3_result = engine.extrapolate_periods(test3_input, 3)
test3_expected = ['Week 4', 'Week 5', 'Week 6']
test("Pattern matching still functional",
     test3_result == test3_expected,
     f"Got {test3_result}, expected {test3_expected}")

print("\n" + "="*70)
print("2. EDGE CASES VERIFICATION")
print("="*70)

# Quarter year transition
test4_input = ['Q3 2024', 'Q4 2024']
test4_result = engine.extrapolate_periods(test4_input, 2)
test4_expected = ['Q1 2025', 'Q2 2025']
test("Quarter year transition (Q4 → Q1)",
     test4_result == test4_expected,
     f"Got {test4_result}, expected {test4_expected}")

# Month year transition
test5_input = ['November 2024', 'December 2024']
test5_result = engine.extrapolate_periods(test5_input, 2)
test5_expected = ['January 2025', 'February 2025']
test("Month year transition (Dec → Jan)",
     test5_result == test5_expected,
     f"Got {test5_result}, expected {test5_expected}")

# Year-first formats
test6_input = ['2022 Q1', '2022 Q2']
test6_result = engine.extrapolate_periods(test6_input, 2)
test6_expected = ['2022 Q3', '2022 Q4']
test("Year-first quarter format",
     test6_result == test6_expected,
     f"Got {test6_result}, expected {test6_expected}")

test7_input = ['2022 January', '2022 February']
test7_result = engine.extrapolate_periods(test7_input, 2)
test7_expected = ['2022 March', '2022 April']
test("Year-first month format",
     test7_result == test7_expected,
     f"Got {test7_result}, expected {test7_expected}")

print("\n" + "="*70)
print("3. DATA TYPE HANDLING")
print("="*70)

# String years
test8_input = ['2020', '2021', '2022']
test8_result = engine.extrapolate_periods(test8_input, 2)
test8_expected = ['2023', '2024']
test("String year handling",
     test8_result == test8_expected,
     f"Got {test8_result}, expected {test8_expected}")

# Integer years  
test9_input = [2020, 2021, 2022]
test9_result = engine.extrapolate_periods(test9_input, 2)
# Should work even if returns strings
test("Integer year handling",
     len(test9_result) == 2,
     f"Got {test9_result}")

print("\n" + "="*70)
print("4. EVENT VALIDATION")
print("="*70)

from forecast_engine import Event, EventType, Assumption, AssumptionType

# Create test event with quarter format
test_event = Event(
    id="test",
    event_type=EventType.PRODUCT_LAUNCH,
    name="Test",
    metric="Revenue",
    date="Q4 2023",
    impact_multiplier=1.1,
    decay_periods=0
)

# This should NOT raise an error
try:
    validation = engine.validate_event(test_event)  # Use validate_event for Event objects
    test("Event validation accepts quarter formats",
         'errors' in validation and len(validation.get('errors', [])) == 0,
         f"Validation: {validation}")
except Exception as e:
    test("Event validation accepts quarter formats", False, str(e))

print("\n" + "="*70)
print("5. SCENARIO GENERATION (NO EXPONENTIAL BUG)")
print("="*70)

# Load test data
df = pd.read_excel('/Users/krishangsharma/Downloads/BNC/sample_quarterly_retail.xlsx')
engine.set_base_data(df)

# Add 9% growth assumption
assumption = Assumption(
    id="test",
    type=AssumptionType.GROWTH,
    name="Growth",
    metric='Units_Sold',
    value=0.09,  # 9%
    confidence='medium',
    source='analyst'
)
engine.add_assumption(assumption)

# Generate scenarios
scenarios = engine.generate_scenarios('Units_Sold', periods=5)

# Check CAGR is reasonable (not 140% or 1000%)
start = scenarios['base']['Units_Sold'].iloc[0]
end = scenarios['base']['Units_Sold'].iloc[-1]
cagr = ((end / start) ** (1/5) - 1) * 100

test("CAGR within reasonable range (5-12%)",
     5 <= cagr <= 12,
     f"CAGR: {cagr:.1f}%")

test("No exponential explosion (forecast < 10M)",
     end < 10_000_000,
     f"End value: {end:,.0f}")

print("\n" + "="*70)
print("AUDIT SUMMARY")
print("="*70)

passed = sum(1 for r in audit_results if r['pass'])
failed = sum(1 for r in audit_results if not r['pass'])
total = len(audit_results)

print(f"\nTotal Tests: {total}")
print(f"✅ Passed: {passed}")
print(f"❌ Failed: {failed}")

if failed > 0:
    print(f"\n❌ FAILED TESTS:")
    for r in audit_results:
        if not r['pass']:
            print(f"  - {r['name']}")
            if r['details']:
                print(f"    {r['details']}")
    print(f"\n⚠️ CODE NOT READY FOR PRODUCTION")
else:
    print(f"\n🎉 ALL TESTS PASSED - CODE READY FOR PRODUCTION")

print(f"\nSuccess Rate: {passed}/{total} ({100*passed/total:.1f}%)")
