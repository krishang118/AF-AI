
import pandas as pd
from forecast_engine import ForecastEngine, Event, EventType

def test_event_application():
    print("--- Starting Reproduction Test ---")
    
    # 1. Setup Engine
    engine = ForecastEngine()
    
    # 2. Mock Data
    dates = pd.date_range(start='2020-01-01', periods=5, freq='Y')
    df = pd.DataFrame({'Revenue': [100, 110, 120, 130, 140]}, index=dates)
    # Ensure 'period' column exists if engine expects it, or index is enough
    # The set_base_data uses the dataframe as is if no 'period' col, 
    # but my recent fix EXPECTS valid index.
    # Let's verify how set_base_data handles this.
    # If passed data has index, it uses it.
    engine.set_base_data(df)
    
    metric = "Revenue"
    
    # 3. Add Event
    # Impact: -15% (0.85 multiplier)
    # Date: "1" (First forecast period)
    print(f"Adding Event: Product Launch, Metric={metric}, Date='1', Impact=0.85")
    event = Event(
        id="test_event_1",
        event_type=EventType.PRODUCT_LAUNCH,
        name="Product Launch Failure",
        metric=metric,
        date="1", # String "1" matching app.py logic
        impact_multiplier=0.85, # -15%
        decay_periods=0
    )
    engine.add_event(event) # IMPORTANT: Add to engine
    
    # 4. Generate Forecast
    print("Generating Forecast for 5 periods...")
    forecast_df = engine.generate_base_forecast(metric, periods=5)
    
    # 5. Inspect Results
    print("\nForecast Results:")
    print(forecast_df[[metric, 'type', 'base_contribution']])
    
    # Check if value dropped
    # Base growth based on historical CAGR?
    # 100->140 in 4 steps. Growth ~10%.
    # Next value should be ~154.
    # With event (0.85), should be ~130.
    
    val_1 = forecast_df.iloc[0][metric]
    base_1 = forecast_df.iloc[0]['base_contribution']
    
    print(f"\nPeriod 1 Base (calc): {base_1}")
    print(f"Period 1 Final: {val_1}")
    
    if val_1 < base_1:
        print("SUCCESS: Event impact detected (Final < Base)")
    else:
        print("FAILURE: Event ignored (Final >= Base)")

    # Check breakdown columns
    print("\nColumns in result:", forecast_df.columns.tolist())
    if "event_test_event_1" in forecast_df.columns:
        print(f"Event Contribution Col Value: {forecast_df.iloc[0]['event_test_event_1']}")
    print(f"Event Contribution Col Value: {forecast_df.iloc[0]['event_test_event_1']}")

    # 6. Test Scenarios (The User's specific complaint)
    print("\nGenerating Scenarios...")
    scenarios = engine.generate_scenarios(metric, periods=5)
    
    upside_df = scenarios['upside']
    print("\nUpside Scenario Results:")
    print(upside_df[[metric]])
    
    val_upside_1 = upside_df.iloc[0][metric]
    # Upside should be roughly (Base * 1.05) = (129 * 1.05) ~ 135.
    # If Event ignored, it would be (152 * 1.05) ~ 160.
    
    print(f"Period 1 Upside: {val_upside_1}")
    if val_upside_1 < 150:
         print("SUCCESS: Upside scenario includes Event Impact")
    else:
         print("FAILURE: Upside scenario ignores Event Impact (Value too high)")

if __name__ == "__main__":
    test_event_application()
