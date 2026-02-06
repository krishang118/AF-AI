"""
Comprehensive Test: Verify ALL period types work seamlessly
Tests: Yearly, Daily, Monthly, Quarterly, Generic (Period-1, Period-2...)
"""
import pandas as pd
import numpy as np
import sys
sys.path.insert(0, '/Users/krishangsharma/Downloads/BNC')

from forecast_engine import ForecastEngine, Assumption, AssumptionType

def test_period_type(name, df, metric='Revenue'):
    """Test a specific period type end-to-end"""
    print(f"\n{'='*60}")
    print(f"Testing: {name}")
    print(f"{'='*60}")
    
    print(f"Data shape: {df.shape}")
    print(f"Period column type: {df['period'].dtype}")
    print(f"Sample periods: {df['period'].head(3).tolist()}")
    
    try:
        # Initialize engine
        engine = ForecastEngine()
        engine.set_base_data(df)
        print("✓ Data loaded")
        
        # Add assumption
        assumption = Assumption(
            id="test_001",
            type=AssumptionType.GROWTH,
            name="Base Growth",
            metric=metric,
            value=0.10,
            confidence='medium',
            source='analyst'
        )
        engine.add_assumption(assumption)
        print("✓ Assumption added")
        
        # Validate data quality
        quality = engine.validate_data_quality(metric)
        print(f"✓ Data quality: {quality['quality_score'].upper()}")
        
        # Generate forecast
        scenarios = engine.generate_scenarios(metric, periods=5)
        print(f"✓ Forecast generated: {len(scenarios['base'])} periods")
        
        # Verify future periods
        forecast_periods = scenarios['base']['period'].tolist()
        print(f"✓ Future periods: {forecast_periods}")
        
        print(f"\n✅ SUCCESS: {name} works perfectly!")
        return True
        
    except Exception as e:
        print(f"\n❌ FAILED: {name}")
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

# =============================================================================
# TEST 1: Yearly Data (Strings)
# =============================================================================
df_yearly_str = pd.DataFrame({
    'period': ['2020', '2021', '2022', '2023', '2024'],
    'Revenue': [100, 110, 121, 133, 146]
})
result1 = test_period_type("Yearly (Strings)", df_yearly_str)

# =============================================================================
# TEST 2: Yearly Data (Integers)
# =============================================================================
df_yearly_int = pd.DataFrame({
    'period': [2020, 2021, 2022, 2023, 2024],
    'Revenue': [100, 110, 121, 133, 146]
})
result2 = test_period_type("Yearly (Integers)", df_yearly_int)

# =============================================================================
# TEST 3: Daily Data (Datetime)
# =============================================================================
df_daily = pd.DataFrame({
    'period': pd.date_range('2024-01-01', periods=10, freq='D'),
    'Revenue': np.random.randint(1000, 2000, 10)
})
result3 = test_period_type("Daily (Datetime)", df_daily)

# =============================================================================
# TEST 4: Monthly Data (Datetime)
# =============================================================================
df_monthly = pd.DataFrame({
    'period': pd.date_range('2024-01-01', periods=12, freq='MS'),
    'Revenue': np.random.randint(5000, 10000, 12)
})
result4 = test_period_type("Monthly (Datetime)", df_monthly)

# =============================================================================
# TEST 5: Quarterly Data (Strings)
# =============================================================================
df_quarterly = pd.DataFrame({
    'period': ['Q1 2024', 'Q2 2024', 'Q3 2024', 'Q4 2024'],
    'Revenue': [100, 110, 121, 133]
})
result5 = test_period_type("Quarterly (Strings)", df_quarterly)

# =============================================================================
# TEST 6: Generic Periods (Period-1, Period-2...)
# =============================================================================
df_generic = pd.DataFrame({
    'period': ['Period-1', 'Period-2', 'Period-3', 'Period-4', 'Period-5'],
    'Revenue': [100, 110, 121, 133, 146]
})
result6 = test_period_type("Generic (Period-N)", df_generic)

# =============================================================================
# TEST 7: Generic Steps (Step 1, Step 2...)
# =============================================================================
df_steps = pd.DataFrame({
    'period': ['Step 1', 'Step 2', 'Step 3', 'Step 4', 'Step 5'],
    'Revenue': [100, 110, 121, 133, 146]
})
result7 = test_period_type("Generic (Step N)", df_steps)

# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "="*60)
print("SUMMARY")
print("="*60)

results = [
    ("Yearly (Strings)", result1),
    ("Yearly (Integers)", result2),
    ("Daily (Datetime)", result3),
    ("Monthly (Datetime)", result4),
    ("Quarterly (Strings)", result5),
    ("Generic (Period-N)", result6),
    ("Generic (Step N)", result7),
]

passed = sum(1 for _, r in results if r)
total = len(results)

for name, result in results:
    status = "✅ PASS" if result else "❌ FAIL"
    print(f"{status}: {name}")

print(f"\n{passed}/{total} tests passed")

if passed == total:
    print("\n🎉 ALL PERIOD TYPES WORK SEAMLESSLY!")
else:
    print(f"\n⚠️ {total - passed} issue(s) need fixing")

sys.exit(0 if passed == total else 1)
