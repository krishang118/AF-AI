"""
Quick debug test for quarter extrapolation
"""
import sys
sys.path.insert(0, '/Users/krishangsharma/Downloads/BNC')

from forecast_engine import ForecastEngine

# Test directly
engine = ForecastEngine()

# Test case 1: Simple quarters
history1 = ['Q1', 'Q2', 'Q3', 'Q4', 'Q1']
result1 = engine.extrapolate_periods(history1, 3)
print(f"Test 1 - Simple quarters")
print(f"Input: {history1}")
print(f"Output: {result1}")
print(f"Expected: ['Q2', 'Q3', 'Q4']")
print()

# Test case 2: Quarters with year
history2 = ['Q1 2022', 'Q2 2022', 'Q3 2022', 'Q4 2022', 'Q1 2023']
result2 = engine.extrapolate_periods(history2, 3)
print(f"Test 2 - Quarters with year")
print(f"Input: {history2}")
print(f"Output: {result2}")
print(f"Expected: ['Q2 2023', 'Q3 2023', 'Q4 2023']")
print()

# Test case 3: Year transition
history3 = ['Q2 2024', 'Q3 2024', 'Q4 2024']
result3 = engine.extrapolate_periods(history3, 3)
print(f"Test 3 - Year transition")
print(f"Input: {history3}")
print(f"Output: {result3}")
print(f"Expected: ['Q1 2025', 'Q2 2025', 'Q3 2025']")
