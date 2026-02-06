"""
End-to-end test simulating the EXACT user workflow
"""
import pandas as pd
import sys
sys.path.insert(0, '/Users/krishangsharma/Downloads/BNC')

from forecast_engine import ForecastEngine, Assumption, AssumptionType, Event, EventType

# Load quarterly data
df = pd.read_excel('sample_quarterly_retail.xlsx')
print("="*70)
print("SIMULATING USER'S EXACT WORKFLOW")
print("="*70)
print(f"Last historical value: {df['Units_Sold'].iloc[-1]:,.0f}")

# Initialize engine (simulating app startup)
engine = ForecastEngine()
engine.set_base_data(df)

# User adds 9% growth via slider (slider value = 9, divided by 100 = 0.09)
slider_value = 9
growth_rate = slider_value / 100  # This is what happens in app.py line 698

print(f"\nUser sets slider to: {slider_value}%")
print(f"Stored as: {growth_rate} (decimal)")

# Create assumption (simulating form submission)
assumption = Assumption(
    id="test001",
    type=AssumptionType.GROWTH,
    name="Base Growth Rate",
    metric='Units_Sold',
    value=growth_rate,  # 0.09
    confidence='medium',
    source='analyst'
)
engine.add_assumption(assumption)

# User adds events (Product Launches)
event1 = Event(
    id="evt001",
    name="Product Launch",
    event_type=EventType.PRODUCT_LAUNCH,
    metric="Units_Sold",
    date="Q3 2022",
    impact_multiplier=0.90,  # -10%
    decay_periods=0,
    description="Q3 2022 launch"
)
engine.add_event(event1)

event2 = Event(
    id="evt002",
    name="Product Launch",
    event_type=EventType.PRODUCT_LAUNCH,
    metric="Units_Sold",
    date="Q3 2027",
    impact_multiplier=0.70,  # -30%
    decay_periods=7,
    description="Q3 2027 launch"
)
engine.add_event(event2)

print(f"\nAdded 2 events:")
print(f"  1. Q3 2022: -10% impact, no decay")
print(f"  2. Q3 2027: -30% impact, 7-period decay")

# Generate base forecast (simulating clicking "Generate Forecast")
print(f"\nGenerating 5-period forecast...")
base_forecast = engine.generate_base_forecast('Units_Sold', periods=5)

print("\nBase Forecast:")
print(base_forecast[['period', 'Units_Sold']])

# Calculate CAGR
start_val = df['Units_Sold'].iloc[-1]
end_val = base_forecast['Units_Sold'].iloc[-1]
total_growth_pct = (end_val / start_val - 1) * 100
cagr = ((end_val / start_val) ** (1/5) - 1) * 100

print(f"\nResults:")
print(f"  Start: {start_val:,.0f}")
print(f"  End (Period 5): {end_val:,.0f}")
print(f"  Total Growth: {total_growth_pct:.1f}%")
print(f"  CAGR: {cagr:.1f}%")

print(f"\nExpected:")
print(f"  With 9% growth: {start_val * (1.09**5):,.0f}")
print(f"  Expected CAGR: ~9%")

# Now generate SCENARIOS (this is what shows in the UI)
print("\n" + "="*70)
print("GENERATING SCENARIOS (what user sees in UI)")
print("="*70)

scenarios = engine.generate_scenarios('Units_Sold', periods=5)

print("\nBase Scenario:")
print(scenarios['base'][['period', 'Units_Sold']])

base_start = scenarios['base']['Units_Sold'].iloc[0]
base_end = scenarios['base']['Units_Sold'].iloc[-1]
scenario_cagr = ((base_end / base_start) ** (1/5) - 1) * 100

print(f"\nScenario CAGR: {scenario_cagr:.1f}%")

# Check if there's a discrepancy
if abs(scenario_cagr - 9.0) > 20:
    print(f"\n❌ BUG DETECTED: CAGR is {scenario_cagr:.1f}% instead of ~9%")
else:
    print(f"\n✅ LOOKS GOOD: CAGR is {scenario_cagr:.1f}% (close to expected 9%)")
