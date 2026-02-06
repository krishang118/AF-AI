"""
Comprehensive Quarter Format Test
Tests all three quarter formats the user mentioned
"""
import pandas as pd
import sys
sys.path.insert(0, '/Users/krishangsharma/Downloads/BNC')

from forecast_engine import ForecastEngine, Assumption, AssumptionType

def test_quarter_format(name, quarters, expected_next):
    """Test a specific quarter format"""
    print(f"\n{'='*60}")
    print(f"Testing: {name}")
    print(f"{'='*60}")
    
    print(f"Input periods: {quarters}")
    print(f"Expected next: {expected_next}")
    
    # Create test data
    df = pd.DataFrame({
        'period': quarters,
        'Revenue': [100, 110, 121, 133, 146][:len(quarters)]
    })
    
    try:
        # Initialize engine
        engine = ForecastEngine()
        engine.set_base_data(df)
        
        # Add assumption
        assumption = Assumption(
            id="test_q",
            type=AssumptionType.GROWTH,
            name="Growth",
            metric='Revenue',
            value=0.10,
            confidence='medium',
            source='analyst'
        )
        engine.add_assumption(assumption)
        
        # Generate forecast
        scenarios = engine.generate_scenarios('Revenue', periods=3)
        
        # Get generated periods
        forecast_periods = scenarios['base']['period'].tolist()
        print(f"✓ Generated: {forecast_periods}")
        
        # Verify format matches
        success = True
        for i, (actual, expected) in enumerate(zip(forecast_periods, expected_next)):
            actual_str = str(actual).strip()
            if actual_str != expected:
                print(f"  ⚠️ Period {i+1}: Expected '{expected}', got '{actual_str}'")
                success = False
        
        if success:
            print(f"\n✅ SUCCESS: {name} format works perfectly!")
            return True
        else:
            print(f"\n⚠️ PARTIAL: {name} generated periods but format differs")
            return False
        
    except Exception as e:
        print(f"\n❌ FAILED: {name}")
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

# =============================================================================
# TEST 1: Simple Cycle (Q1, Q2, Q3, Q4, Q1, Q2...)
# =============================================================================
quarters_simple = ['Q1', 'Q2', 'Q3', 'Q4', 'Q1']
expected_simple = ['Q2', 'Q3', 'Q4']  # Next 3 after Q1
result1 = test_quarter_format(
    "Simple Cycle (Q1, Q2, Q3, Q4)",
    quarters_simple,
    expected_simple
)

# =============================================================================
# TEST 2: With Year (Q1 2022, Q2 2022, Q3 2022, Q4 2022, Q1 2023...)
# =============================================================================
quarters_with_year = ['Q1 2022', 'Q2 2022', 'Q3 2022', 'Q4 2022', 'Q1 2023']
expected_with_year = ['Q2 2023', 'Q3 2023', 'Q4 2023']  # Next 3 after Q1 2023
result2 = test_quarter_format(
    "With Year (Q1 2022, Q2 2022...)",
    quarters_with_year,
    expected_with_year
)

# =============================================================================
# TEST 3: Pure Years (2020, 2021, 2022...)
# =============================================================================
years_only = [2020, 2021, 2022, 2023, 2024]
expected_years = [2025, 2026, 2027]  # Next 3 years
result3 = test_quarter_format(
    "Pure Years (2020, 2021, 2022...)",
    years_only,
    [str(y) for y in expected_years]  # Engine returns strings
)

# =============================================================================
# TEST 4: Edge Case - Year before Q (2022 Q1, 2022 Q2...)
# =============================================================================
quarters_year_first = ['2022 Q1', '2022 Q2', '2022 Q3', '2022 Q4', '2023 Q1']
expected_year_first = ['2023 Q2', '2023 Q3', '2023 Q4']
result4 = test_quarter_format(
    "Year First (2022 Q1, 2022 Q2...)",
    quarters_year_first,
    expected_year_first
)

# =============================================================================
# TEST 5: Edge Case - Transition across year (Q3 2024 -> Q4 2024 -> Q1 2025)
# =============================================================================
quarters_transition = ['Q2 2024', 'Q3 2024', 'Q4 2024']
expected_transition = ['Q1 2025', 'Q2 2025', 'Q3 2025']
result5 = test_quarter_format(
    "Year Transition (Q4 2024 -> Q1 2025)",
    quarters_transition,
    expected_transition
)

# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "="*60)
print("SUMMARY")
print("="*60)

results = [
    ("Simple Cycle (Q1, Q2...)", result1),
    ("With Year (Q1 2022...)", result2),
    ("Pure Years (2020, 2021...)", result3),
    ("Year First (2022 Q1...)", result4),
    ("Year Transition (Q4 -> Q1)", result5),
]

passed = sum(1 for _, r in results if r)
total = len(results)

for name, result in results:
    status = "✅ PASS" if result else "❌ FAIL"
    print(f"{status}: {name}")

print(f"\n{passed}/{total} tests passed")

if passed == total:
    print("\n🎉 ALL QUARTER FORMATS WORK PERFECTLY!")
else:
    print(f"\n⚠️ {total - passed} issue(s) detected")

sys.exit(0 if passed == total else 1)
