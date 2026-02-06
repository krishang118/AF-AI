"""
SIMPLIFIED: Test just the CAGR calculation with scenarios
"""
import pandas as pd
import sys
sys.path.insert(0, '/Users/krishangsharma/Downloads/BNC')

from forecast_engine import ForecastEngine, Assumption, AssumptionType

# Load quarterly data
df = pd.read_excel('sample_quarterly_retail.xlsx')

print("="*70)
print("TESTING: Does 9% growth produce ~9% CAGR in scenarios?")
print("="*70)
print(f"Last historical Units_Sold: {df['Units_Sold'].iloc[-1]:,.0f}\n")

# Initialize engine
engine = ForecastEngine()
engine.set_base_data(df)

# Add 9% growth assumption (slider value 9 → stored as 0.09)
growth_rate = 9 / 100  # 0.09

assumption = Assumption(
    id="test001",
    type=AssumptionType.GROWTH,
    name="Base Growth Rate",
    metric='Units_Sold',
    value=growth_rate,
    confidence='medium',
    source='analyst'
)
engine.add_assumption(assumption)

print(f"Assumption created:")
print(f"  Slider value: 9%")
print(f"  Stored value: {growth_rate}")
print(f"  Display format (.1%): {growth_rate:.1%}\n")

# Generate scenarios (this is what the UI shows)
print("Generating scenarios for 5 periods...\n")
scenarios = engine.generate_scenarios('Units_Sold', periods=5)

# Analyze base scenario
base_start = scenarios['base']['Units_Sold'].iloc[0]
base_end = scenarios['base']['Units_Sold'].iloc[-1]
base_cagr = ((base_end / base_start) ** (1/5) - 1) * 100

print("BASE SCENARIO Results:")
print(f"  Period 1: {base_start:,.0f}")
print(f"  Period 5: {base_end:,.0f}")
print(f"  CAGR: {base_cagr:.1f}%")
print(f"  Expected: ~9.0%")

# Check if bug exists
if abs(base_cagr - 9.0) > 5:
    print(f"\n❌ BUG FOUND: CAGR is {base_cagr:.1f}% instead of ~9%")
    print(f"   Difference: {base_cagr - 9.0:+.1f}%")
    
    # Diagnose
    print(f"\n�� Diagnosis:")
    print(f"  If stored as 0.09: CAGR should be ~9%")
    print(f"  If stored as 0.9: CAGR would be ~{((base_start * (1.9**5) / base_start)**(1/5) - 1)*100:.1f}%")
    print(f"  If stored as 9.0: CAGR would be ~{((base_start * (10**5) / base_start)**(1/5) - 1)*100:.1f}%")
else:
    print(f"\n✅ PASS: CAGR is close to expected 9%")

# Also check upside/downside
upside_start = scenarios['upside']['Units_Sold'].iloc[0]
upside_end = scenarios['upside']['Units_Sold'].iloc[-1]
upside_cagr = ((upside_end / upside_start) ** (1/5) - 1) * 100

downside_start = scenarios['downside']['Units_Sold'].iloc[0]
downside_end = scenarios['downside']['Units_Sold'].iloc[-1]
downside_cagr = ((downside_end / downside_start) ** (1/5) - 1) * 100

print(f"\nUPSIDE CAGR: {upside_cagr:.1f}% (Expected: ~9.9%)")
print(f"DOWNSIDE CAGR: {downside_cagr:.1f}% (Expected: ~8.1%)")
