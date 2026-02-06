import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_samples():
    print("Generating realistic weekly sample data...")
    
    # Common Date Range (2 years of weekly data)
    start_date = datetime(2024, 1, 1)
    periods = 104 # 52 weeks * 2 years
    dates = pd.date_range(start=start_date, periods=periods, freq='W-MON')
    
    # ---------------------------------------------------------
    # 1. Weekly Revenue (CSV) - Retail style
    # ---------------------------------------------------------
    # Trend + Seasonality (Higher in Q4) + Noise
    t = np.linspace(0, 4*np.pi, periods)
    trend = np.linspace(50000, 120000, periods) # Growth from 50k to 120k
    seasonality = 10000 * np.sin(t) # Simple sine wave
    
    # Add Q4 bumps (Nov-Dec)
    q4_bump = np.zeros(periods)
    for i, d in enumerate(dates):
        if d.month in [11, 12]:
            q4_bump[i] = 20000
            
    noise = np.random.normal(0, 5000, periods)
    revenue = trend + seasonality + q4_bump + noise
    
    df_rev = pd.DataFrame({
        'Week_Ending': dates,
        'Weekly_Revenue': revenue.astype(int),
        'Orders': (revenue / 50).astype(int) # Approx $50 AOV
    })
    
    df_rev.to_csv('sample_weekly_revenue.csv', index=False)
    print("✅ Created 'sample_weekly_revenue.csv'")

    # ---------------------------------------------------------
    # 2. Weekly Marketing Spend (XLSX) - SaaS/Digital style
    # ---------------------------------------------------------
    # Correlated with revenue but leading? Let's just make it step-wise growth
    spend_base = np.linspace(5000, 15000, periods)
    spend_noise = np.random.normal(0, 1000, periods)
    spend = spend_base + spend_noise
    
    # Clicks correlated to spend
    cpc = 2.5
    clicks = (spend / cpc) * np.random.uniform(0.9, 1.1, periods)
    
    df_mkt = pd.DataFrame({
        'Date': dates,
        'Ad_Spend': spend,
        'Clicks': clicks.astype(int),
        'Impressions': (clicks * 25).astype(int) # ~4% CTR
    })
    
    # Save as Excel
    df_mkt.to_excel('sample_weekly_marketing.xlsx', index=False)
    print("✅ Created 'sample_weekly_marketing.xlsx'")

    # ---------------------------------------------------------
    # 3. Weekly Active Users (CSV) - App Growth style
    # ---------------------------------------------------------
    # Logistic Growth (S-Curve)
    x = np.linspace(-6, 6, periods)
    # Sigmoid function: L / (1 + e^-k(x-x0))
    L = 500000 # Max users
    k = 0.1 # Steepness
    wau = L / (1 + np.exp(-x)) 
    
    # Add some weekly jitter
    wau_jitter = wau * np.random.uniform(0.98, 1.02, periods)
    
    df_users = pd.DataFrame({
        'Period': dates,
        'WAU': wau_jitter.astype(int),
        'New_Signups': np.gradient(wau_jitter).astype(int)
    })
    
    df_users.to_csv('sample_weekly_users.csv', index=False)
    print("✅ Created 'sample_weekly_users.csv'")
    print("\nDone! 3 sample files ready for upload.")

if __name__ == "__main__":
    generate_samples()
