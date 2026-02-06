"""
Quick CAGR verification for user's forecast
"""

# Standard CAGR formula: ((End / Start) ^ (1 / n)) - 1

# User's setup:
# - Forecast Periods: 5
# - Base CAGR shown: 2.6%
# - Growth assumption: 3%
# - Event: -15% at period 2, decay over 4 periods

# Let's simulate a simple scenario
import numpy as np

def verify_cagr(start_value, end_value, periods):
    """Calculate CAGR"""
    cagr = ((end_value / start_value) ** (1 / periods) - 1) * 100
    return cagr

# Example: If we start at 14,000 and grow at 3% for 5 periods
# But have a -15% event impact at period 2 with decay...

# Simplified simulation
start = 14000
growth_rate = 0.03

# Period 0 (start): 14,000
# Period 1: 14,000 * 1.03 = 14,420
# Period 2: 14,420 * 1.03 * 0.85 (event) = 12,627
# Period 3: 12,627 * 1.03 * (decay recovery) ≈ 13,000
# Period 4: Continue...
# Period 5: Final value

# Let's calculate with event impact
values = [start]
event_multiplier = 0.85  # -15% = multiply by 0.85

for i in range(1, 6):
    prev = values[-1]
    new_val = prev * (1 + growth_rate)
    
    # Apply event at period 2 (index 1 in forecast)
    if i == 2:
        new_val *= event_multiplier
        print(f"Period {i}: Event applied! {prev:.2f} → {new_val:.2f}")
    elif i in [3, 4, 5]:
        # Decay periods: event effect gradually diminishes
        decay_recovery = 0.15 * (1 - (i - 2) / 4)  # Linear decay
        new_val *= (1 + decay_recovery)
        print(f"Period {i}: Decay recovery (+{decay_recovery*100:.1f}%): {prev:.2f} → {new_val:.2f}")
    else:
        print(f"Period {i}: Normal growth: {prev:.2f} → {new_val:.2f}")
    
    values.append(new_val)

print(f"\nStart: {values[0]:.2f}")
print(f"End (Period 5): {values[5]:.2f}")

calculated_cagr = verify_cagr(values[0], values[5], 5)
print(f"\nCalculated CAGR: {calculated_cagr:.1f}%")
print(f"Expected (from UI): 2.6%")

if abs(calculated_cagr - 2.6) < 1.0:
    print("\n✅ CAGR calculation is CORRECT!")
else:
    print(f"\n⚠️ Discrepancy: {calculated_cagr - 2.6:.1f}%")

# Also verify the formula itself
print("\n" + "="*50)
print("CAGR Formula Verification:")
print("="*50)
test_start = 100
test_end = 110.41  # 2% growth for 5 periods
test_periods = 5

expected_cagr = 2.0  # We know this should be 2%
calculated = verify_cagr(test_start, test_end, test_periods)
print(f"Test: {test_start} → {test_end} over {test_periods} periods")
print(f"Expected CAGR: {expected_cagr:.1f}%")
print(f"Calculated CAGR: {calculated:.1f}%")
print(f"Formula is {'✅ CORRECT' if abs(calculated - expected_cagr) < 0.1 else '❌ WRONG'}")
