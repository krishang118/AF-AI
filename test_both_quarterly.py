"""
Test: Upload both quarterly files and check for CAGR issues
"""
import pandas as pd
import sys
sys.path.insert(0, '/Users/krishangsharma/Downloads/BNC')

from forecast_engine import ForecastEngine, Assumption, AssumptionType

print("="*70)
print("TEST 1: Quarterly Retail (Units_Sold)")
print("="*70)

df_retail = pd.read_excel('sample_quarterly_retail.xlsx')
print(f"Last 3 periods:")
print(df_retail[['Period', 'Units_Sold']].tail(3))
print(f"\nLast value: {df_retail['Units_Sold'].iloc[-1]:,.0f}")

engine1 = ForecastEngine()
engine1.set_base_data(df_retail)

# 9% growth  
assumption1 = Assumption(
    id="test1",
    type=AssumptionType.GROWTH,
    name="Growth",
    metric='Units_Sold',
    value=0.09,
    confidence='medium',
    source='analyst'
)
engine1.add_assumption(assumption1)

scenarios1 = engine1.generate_scenarios('Units_Sold', periods=5)
base_start1 = scenarios1['base']['Units_Sold'].iloc[0]
base_end1 = scenarios1['base']['Units_Sold'].iloc[-1]
cagr1 = ((base_end1 / base_start1) ** (1/5) - 1) * 100

print(f"\nScenario Results:")
print(f"  Start: {base_start1:,.0f}")
print(f"  End: {base_end1:,.0f}")
print(f"  CAGR: {cagr1:.1f}%")

if abs(cagr1 - 9.0) > 30:
    print(f"  ❌ BUG: CAGR is {cagr1:.1f}% (expected ~9%)")
else:
    print(f"  ✅ PASS")

print("\n" + "="*70)
print("TEST 2: Quarterly SaaS (Quarterly_Revenue)")
print("="*70)

df_saas = pd.read_csv('sample_quarterly_saas.csv')
print(f"Last 3 periods:")
print(df_saas[['Quarter', 'Quarterly_Revenue']].tail(3))
print(f"\nLast value: {df_saas['Quarterly_Revenue'].iloc[-1]:,.0f}")

# Rename Quarter to period for consistency
df_saas = df_saas.rename(columns={'Quarter': 'period'})

engine2 = ForecastEngine()
engine2.set_base_data(df_saas)

assumption2 = Assumption(
    id="test2",
    type=AssumptionType.GROWTH,
    name="Growth",
    metric='Quarterly_Revenue',
    value=0.09,
    confidence='medium',
    source='analyst'
)
engine2.add_assumption(assumption2)

scenarios2 = engine2.generate_scenarios('Quarterly_Revenue', periods=5)
base_start2 = scenarios2['base']['Quarterly_Revenue'].iloc[0]
base_end2 = scenarios2['base']['Quarterly_Revenue'].iloc[-1]
cagr2 = ((base_end2 / base_start2) ** (1/5) - 1) * 100

print(f"\nScenario Results:")
print(f"  Start: {base_start2:,.0f}")
print(f"  End: {base_end2:,.0f}")
print(f"  CAGR: {cagr2:.1f}%")

if abs(cagr2 - 9.0) > 30:
    print(f"  ❌ BUG: CAGR is {cagr2:.1f}% (expected ~9%)")
else:
    print(f"  ✅ PASS")

print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print(f"Retail CAGR: {cagr1:.1f}%")
print(f"SaaS CAGR: {cagr2:.1f}%")

if abs(cagr1 - 9.0) > 30 or abs(cagr2 - 9.0) > 30:
    print("\n❌ At least one test shows abnormal CAGR")
else:
    print("\n✅ Both tests show reasonable CAGR")
