"""
Test: Annual growth rate standardization with weekly data
"""
import pandas as pd
import sys
sys.path.insert(0, '/Users/krishangsharma/Downloads/BNC')

from forecast_engine import ForecastEngine, Assumption, AssumptionType

print("="*70)
print("TEST: Annual Growth Rate Standardization")
print("="*70)

# Load weekly data (104 periods)
df = pd.read_csv('/Users/krishangsharma/Downloads/BNC/sample_weekly_revenue.csv')
print(f"\nData loaded: {len(df)} periods (weekly)")
print(f"Last 3 periods:")
print(df[['Week_Ending', 'Weekly_Revenue']].tail(3))

engine = ForecastEngine()
engine.set_base_data(df)

print(f"\nDetected frequency: {engine.periods_per_year} periods/year")

freq_map = {1: "Yearly", 4: "Quarterly", 12: "Monthly", 52: "Weekly", 252: "Daily"}
print(f"Interpretation: {freq_map.get(engine.periods_per_year, 'Unknown')}")

# Add 20% ANNUAL growth assumption
annual_rate = 0.20  # 20% per year
assumption = Assumption(
    id="test",
    type=AssumptionType.GROWTH,
    name="Annual Growth",
    metric='Weekly_Revenue',
    value=annual_rate,  # 20% annual
    confidence='medium',
    source='analyst'
)
engine.add_assumption(assumption)

print(f"\nAssumption: {annual_rate*100}% ANNUAL growth")

# Expected period rate: (1.20)^(1/52) - 1 = 0.35% per week
expected_period_rate = (1 + annual_rate) ** (1/engine.periods_per_year) - 1
print(f"Expected period rate: {expected_period_rate*100:.3f}% per period")

# Generate forecast
scenarios = engine.generate_scenarios('Weekly_Revenue', periods=21)  # 21 weeks

start = scenarios['base']['Weekly_Revenue'].iloc[0]
end = scenarios['base']['Weekly_Revenue'].iloc[-1]

print(f"\nForecast Results:")
print(f"  Week 1: ${start:,.0f}")
print(f"  Week 21: ${end:,.0f}")
print(f"  Total growth: {(end/start - 1)*100:.1f}%")

# Calculate annualized CAGR
# For 21 weeks: CAGR = ((end/start)^(52/21) - 1) * 100
periods_per_year = 52
forecast_periods = 21
annualized_cagr = ((end / start) ** (periods_per_year / forecast_periods) - 1) * 100

print(f"  Annualized CAGR: {annualized_cagr:.1f}%")

print(f"\n" + "="*70)
if abs(annualized_cagr - 20.0) < 2:
    print(f"✅ SUCCESS: Annualized CAGR ({annualized_cagr:.1f}%) matches input (20%)")
    print(f"   Weekly data now correctly interprets 20% as ANNUAL rate!")
else:
    print(f"❌ ISSUE: Annualized CAGR ({annualized_cagr:.1f}%) doesn't match input (20%)")
    print(f"   Difference: {annualized_cagr - 20:.1f}%")
