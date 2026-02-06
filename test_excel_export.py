"""
Test Excel Export with Quarterly Data (the bug case)
"""
import pandas as pd
import sys
sys.path.insert(0, '/Users/krishangsharma/Downloads/BNC')

from forecast_engine import ForecastEngine, Assumption, AssumptionType

print("="*70)
print("TEST: Excel Export with Quarterly Data (String Periods)")
print("="*70)

# Load quarterly data
df = pd.read_excel('/Users/krishangsharma/Downloads/BNC/quarterly/sample_quarterly_retail.xlsx')
print(f"\nData loaded: {len(df)} quarters")
print(f"Period column type: {df['Period'].dtype}")
print(f"Sample periods: {df['Period'].head(3).tolist()}")

engine = ForecastEngine()
engine.set_base_data(df)

# Add assumption
assumption = Assumption(
    id="test",
    type=AssumptionType.GROWTH,
    name="Growth",
    metric='Units_Sold',
    value=0.09,
    confidence='medium',
    source='test'
)
engine.add_assumption(assumption)

# Generate scenarios
print("\nGenerating scenarios...")
scenarios = engine.generate_scenarios('Units_Sold', periods=5)

print(f"Scenarios generated successfully!")
print(f"Base forecast shape: {scenarios['base'].shape}")
print(f"Period column in forecast: {scenarios['base']['period'].dtype}")

# Try to export to Excel
print("\nTesting Excel export...")
try:
    from forecast_engine import ExcelOutputGenerator
    
    generator = ExcelOutputGenerator()  # No arguments needed
    output_path = '/tmp/test_quarterly_export.xlsx'
    generator.generate_excel_output(scenarios, 'Units_Sold', output_path)
    
    print(f"✅ SUCCESS: Excel file created at {output_path}")
    
    # Verify file exists and is readable
    import os
    if os.path.exists(output_path):
        file_size = os.path.getsize(output_path)
        print(f"   File size: {file_size:,} bytes")
        
        # Try to read it back
        test_read = pd.read_excel(output_path, sheet_name='Scenario_Comparison')
        print(f"   Verified: File readable, {len(test_read)} rows")
        print(f"\n🎉 EXCEL EXPORT WORKING PERFECTLY!")
    else:
        print(f"❌ FAIL: File not created")
        
except Exception as e:
    print(f"❌ FAIL: {str(e)}")
    import traceback
    traceback.print_exc()

print("="*70)
