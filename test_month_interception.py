"""
Test: Check if month formats are intercepted by pattern matching
"""
import sys
sys.path.insert(0, '/Users/krishangsharma/Downloads/BNC')

from forecast_engine import ForecastEngine

engine = ForecastEngine()

print("="*70)
print("TESTING MONTH FORMAT INTERCEPTION")
print("="*70)

# Test 1: Simple month names (should work - no numbers)
test1 = ['Jan', 'Feb', 'Mar']
result1 = engine.extrapolate_periods(test1, 3)
print(f"\nTest 1: Simple months")
print(f"Input: {test1}")
print(f"Output: {result1}")
print(f"Expected: ['Apr', 'May', 'Jun']")
if result1 == ['Apr', 'May', 'Jun']:
    print("✅ PASS")
else:
    print("⚠️ ISSUE DETECTED")

# Test 2: Months with years (RISKY - has numbers!)
test2 = ['Jan 2022', 'Feb 2022', 'Mar 2022']
result2 = engine.extrapolate_periods(test2, 3)
print(f"\nTest 2: Months with years")
print(f"Input: {test2}")
print(f"Output: {result2}")
print(f"Expected: ['Apr 2022', 'May 2022', 'Jun 2022']")
if result2 == ['Apr 2022', 'May 2022', 'Jun 2022']:
    print("✅ PASS")
else:
    print("❌ BUG: Month + year format intercepted by pattern matcher!")

# Test 3: Year-first months
test3 = ['2022 January', '2022 February', '2022 March']
result3 = engine.extrapolate_periods(test3, 3)
print(f"\nTest 3: Year-first months")
print(f"Input: {test3}")
print(f"Output: {result3}")
print(f"Expected: ['2022 April', '2022 May', '2022 June']")
if result3 == ['2022 April', '2022 May', '2022 June']:
    print("✅ PASS")
else:
    print("❌ BUG: Year + month format intercepted!")

# Test 4: Week formats
test4 = ['Week 1', 'Week 2', 'Week 3']
result4 = engine.extrapolate_periods(test4, 3)
print(f"\nTest 4: Week numbers")
print(f"Input: {test4}")
print(f"Output: {result4}")
print(f"Expected: ['Week 4', 'Week 5', 'Week 6']")
if result4 == ['Week 4', 'Week 5', 'Week 6']:
    print("✅ PASS (pattern matcher should handle this)")
else:
    print("⚠️ Unexpected result")

# Test 5: Day names (no specific handler)
test5 = ['Monday', 'Tuesday', 'Wednesday']
result5 = engine.extrapolate_periods(test5, 3)
print(f"\nTest 5: Day names")
print(f"Input: {test5}")
print(f"Output: {result5}")
print(f"Expected: ['Thursday', 'Friday', 'Saturday'] OR fallback")
# This might not work - no handler for day names

print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print("If ANY tests show issues, month logic needs to be moved before pattern matching!")
