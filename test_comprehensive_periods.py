"""
COMPREHENSIVE EDGE CASE TEST
Testing all discovered and fixed issues
"""
import sys
sys.path.insert(0, '/Users/krishangsharma/Downloads/BNC')

from forecast_engine import ForecastEngine

engine = ForecastEngine()

print("="*70)
print("COMPREHENSIVE PERIOD FORMAT EDGE CASE TESTING")
print("="*70)

tests = [
    # Quarters (previously broken)
    {
        'name': 'Quarters - Simple Cycle',
        'input': ['Q1', 'Q2', 'Q3'],
        'expected': ['Q4', 'Q1', 'Q2']
    },
    {
        'name': 'Quarters - With Year',
        'input': ['Q1 2022', 'Q2 2022', 'Q3 2022'],
        'expected': ['Q4 2022', 'Q1 2023', 'Q2 2023']
    },
    {
        'name': 'Quarters - Year First',
        'input': ['2022 Q1', '2022 Q2', '2022 Q3'],
        'expected': ['2022 Q4', '2023 Q1', '2023 Q2']
    },
    {
        'name': 'Quarters - Year Transition',
        'input': ['Q3 2024', 'Q4 2024'],
        'expected': ['Q1 2025', 'Q2 2025', 'Q3 2025']
    },
    
    # Months (previously broken with years)
    {
        'name': 'Months - Simple',
        'input': ['Jan', 'Feb', 'Mar'],
        'expected': ['Apr', 'May', 'Jun']
    },
    {
        'name': 'Months - With Year',
        'input': ['Jan 2022', 'Feb 2022', 'Mar 2022'],
        'expected': ['Apr 2022', 'May 2022', 'Jun 2022']
    },
    {
        'name': 'Months - Year First',
        'input': ['2022 January', '2022 February'],
        'expected': ['2022 March', '2022 April', '2022 May']
    },
    {
        'name': 'Months - Year Transition',
        'input': ['November', 'December'],
        'expected': ['January', 'February', 'March']
    },
    
    # Years
    {
        'name': 'Years - String',
        'input': ['2020', '2021', '2022'],
        'expected': ['2023', '2024', '2025']
    },
    {
        'name': 'Years - Integer',
        'input': [2020, 2021, 2022],
        'expected': ['2023', '2024', '2025']  # Returns as strings
    },
    
    # Pattern matching (should still work)
    {
        'name': 'Week Numbers',
        'input': ['Week 1', 'Week 2', 'Week 3'],
        'expected': ['Week 4', 'Week 5', 'Week 6']
    },
    {
        'name': 'Period Numbers',
        'input': ['Period-1', 'Period-2', 'Period-3'],
        'expected': ['Period-4', 'Period-5', 'Period-6']
    },
    {
        'name': 'Step Numbers',
        'input': ['Step 1', 'Step 2', 'Step 3'],
        'expected': ['Step 4', 'Step 5', 'Step 6']
    },
]

results = {'passed': 0, 'failed': 0, 'failures': []}

for test in tests:
    result = engine.extrapolate_periods(test['input'], 3)
    
    if result == test['expected']:
        print(f"✅ {test['name']}: PASS")
        results['passed'] += 1
    else:
        print(f"❌ {test['name']}: FAIL")
        print(f"   Input: {test['input']}")
        print(f"   Expected: {test['expected']}")
        print(f"   Got: {result}")
        results['failed'] += 1
        results['failures'].append(test['name'])

print("\n" + "="*70)
print("FINAL RESULTS")
print("="*70)
print(f"Passed: {results['passed']}/{len(tests)}")
print(f"Failed: {results['failed']}/{len(tests)}")

if results['failed'] > 0:
    print(f"\n❌ FAILURES:")
    for failure in results['failures']:
        print(f"  - {failure}")
else:
    print(f"\n🎉 ALL {len(tests)} TESTS PASSED!")
    print("✅ Quarter formats work")
    print("✅ Month formats work") 
    print("✅ Year formats work")
    print("✅ Pattern matching still works")
    print("✅ No regressions detected")
