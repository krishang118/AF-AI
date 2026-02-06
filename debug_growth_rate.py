"""
Debug: Check if 9% growth is being interpreted correctly
"""
import pandas as pd
import sys
sys.path.insert(0, '/Users/krishangsharma/Downloads/BNC')

from forecast_engine import ForecastEngine, Assumption, AssumptionType

# Load the actual quarterly data
df = pd.read_excel('sample_quarterly_retail.xlsx')
print("Data loaded:")
print(df.head())
print(f"\nLast value: {df['Units_Sold'].iloc[-1]:,.0f}")

# Initialize engine
engine = ForecastEngine()
engine.set_base_data(df)

# Add 9% growth assumption (AS DECIMAL)
assumption = Assumption(
    id="test_growth",
    type=AssumptionType.GROWTH,
    name="Base Growth Rate",
    metric='Units_Sold',
    value=0.09,  # 9% as decimal
    confidence='medium',
    source='analyst'
)
engine.add_assumption(assumption)

# Generate  forecast for 5 periods
forecast = engine.generate_base_forecast('Units_Sold', periods=5)

print("\nForecast generated:")
print(forecast[['period', 'Units_Sold']])

# Check the math
start = df['Units_Sold'].iloc[-1]
end = forecast['Units_Sold'].iloc[-1]
total_growth = (end / start - 1) * 100
print(f"\nStart: {start:,.0f}")
print(f"End (period 5): {end:,.0f}")
print(f"Total growth: {total_growth:.1f}%")
print(f"Expected with 9% growth: {start * (1.09**5):,.0f}")

# Now test with 9.0 instead of 0.09 (USER ERROR scenario)
print("\n" + "="*60)
print("Testing with 9.0 instead of 0.09 (possible bug):")
print("="*60)

engine2 = ForecastEngine()
engine2.set_base_data(df)

assumption2 = Assumption(
    id="test_growth2",
    type=AssumptionType.GROWTH,
    name="Base Growth Rate",
    metric='Units_Sold',
    value=9.0,  # WRONG: 9.0 instead of 0.09
    confidence='medium',
    source='analyst'
)
engine2.add_assumption(assumption2)

forecast2 = engine2.generate_base_forecast('Units_Sold', periods=5)
print(forecast2[['period', 'Units_Sold']])

end2 = forecast2['Units_Sold'].iloc[-1]
total_growth2 = (end2 / start - 1) * 100
print(f"\nStart: {start:,.0f}")
print(f"End (period 5): {end2:,.0f}")
print(f"Total growth: {total_growth2:.1f}%")
print(f"Expected with 900% growth: {start * (10**5):,.0f}")
