"""
Debug: Check what's happening with quarterly data
"""
import pandas as pd
import sys
sys.path.insert(0, '/Users/krishangsharma/Downloads/BNC')

from forecast_engine import ForecastEngine, Assumption, AssumptionType

# Load quarterly data
df = pd.read_excel('/Users/krishangsharma/Downloads/BNC/sample_quarterly_retail.xlsx')
print("Historical data (last 5 periods):")
print(df[['Period', 'Units_Sold']].tail(5))

engine = ForecastEngine()
engine.set_base_data(df)

print(f"\nDetected frequency: {engine.periods_per_year} periods/year")

# Add 20% annual growth
annual_rate = 0.20
assumption = Assumption(
    id="test",
    type=AssumptionType.GROWTH,
    name="Growth",
    metric='Units_Sold',
    value=annual_rate,
    confidence='medium',
    source='test'
)
engine.add_assumption(assumption)

# Expected: convert 20% annual to quarterly
# Formula: (1.20)^(1/4) - 1 = 4.66%
expected_quarterly_rate = (1 + annual_rate) ** (1/4) - 1
print(f"\n20% annual should convert to: {expected_quarterly_rate*100:.3f}% per quarter")

# Generate 8-period forecast (2 years)
scenarios = engine.generate_scenarios('Units_Sold', periods=8)

base = scenarios['base']['Units_Sold']
print(f"\nForecast values:")
for i, val in enumerate(base, 1):
    print(f"  Period {i}: {val:,.0f}")

# Check period-over-period growth
print(f"\nPeriod-over-period growth rates:")
for i in range(1, len(base)):
    period_growth = (base.iloc[i] / base.iloc[i-1] - 1) * 100
    print(f"  Period {i} → {i+1}: {period_growth:.2f}%")

# Calculate CAGR from period 1 to period 8
start_val = base.iloc[0]
end_val = base.iloc[-1]

# CAGR formula: (end/start)^(1/num_years) - 1
# For 8 quarters = 2 years
num_years = 8 / 4  # 8 quarters = 2 years
cagr = (end_val / start_val) ** (1/num_years) - 1

print(f"\nCAGR Calculation:")
print(f"  Start (period 1): {start_val:,.0f}")
print(f"  End (period 8): {end_val:,.0f}")
print(f"  Periods: 8 quarters = {num_years} years")
print(f"  CAGR: {cagr*100:.2f}%")
print(f"  Expected: 20%")
print(f"  Difference: {(cagr*100 - 20):.2f}%")

# Check if formula is correct
# With 4.66% quarterly for 8 periods:
expected_period_1 = base.iloc[0]
expected_period_8 = expected_period_1 * ((1 + expected_quarterly_rate) ** 8)
print(f"\nManual calculation check:")
print(f"  Expected period 8 value: {expected_period_8:,.0f}")
print(f"  Actual period 8 value: {end_val:,.0f}")
print(f"  Match: {abs(expected_period_8 - end_val) < 1}")
