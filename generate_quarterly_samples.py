import pandas as pd
import numpy as np

def generate_quarterly_samples():
    print("Generating realistic quarterly sample data...")
    
    # ---------------------------------------------------------
    # 1. Quarterly Revenue (CSV) - SaaS Company style
    # ---------------------------------------------------------
    # 3 years of quarterly data (12 quarters)
    quarters = [
        'Q1 2022', 'Q2 2022', 'Q3 2022', 'Q4 2022',
        'Q1 2023', 'Q2 2023', 'Q3 2023', 'Q4 2023',
        'Q1 2024', 'Q2 2024', 'Q3 2024', 'Q4 2024'
    ]
    
    # Realistic SaaS growth: strong growth with seasonal patterns
    # Q4 typically strongest for enterprise SaaS
    base_revenue = np.array([
        2.5, 2.8, 3.1, 3.8,  # 2022 - Starting year
        4.2, 4.6, 5.1, 6.2,  # 2023 - Growth year
        6.8, 7.4, 8.2, 9.8   # 2024 - Scaling
    ]) * 1_000_000  # In millions
    
    # Add some realistic noise
    revenue_noise = np.random.normal(0, 50000, len(quarters))
    revenue = base_revenue + revenue_noise
    
    # ARR (Annual Recurring Revenue) - typically 4x Q4 MRR for SaaS
    arr = revenue * 4.2
    
    # Customer counts
    customers = (revenue / 25000).astype(int)  # ~$25k ACV
    
    # Churn rate (decreases as company matures)
    churn_rate = np.linspace(8.5, 4.2, len(quarters))
    
    df_saas = pd.DataFrame({
        'Quarter': quarters,
        'Quarterly_Revenue': revenue.astype(int),
        'ARR': arr.astype(int),
        'Total_Customers': customers,
        'Churn_Rate_Pct': churn_rate.round(1)
    })
    
    df_saas.to_csv('sample_quarterly_saas.csv', index=False)
    print("✅ Created 'sample_quarterly_saas.csv'")
    print(f"   Sample: Q4 2024 Revenue = ${df_saas.iloc[-1]['Quarterly_Revenue']:,.0f}")
    
    # ---------------------------------------------------------
    # 2. Quarterly Operations (XLSX) - Retail/E-commerce style
    # ---------------------------------------------------------
    # Same quarters
    
    # Retail has strong Q4 seasonality (holiday shopping)
    base_sales = np.array([
        850, 920, 980, 1450,   # 2022 - Holiday bump in Q4
        1100, 1180, 1260, 1820, # 2023 - Growth + holiday
        1380, 1490, 1580, 2280  # 2024 - Continued growth
    ]) * 1000  # In thousands of units
    
    # Units sold
    units_sold = base_sales + np.random.normal(0, 10000, len(quarters))
    
    # Average order value (increasing slightly over time)
    aov = np.linspace(85, 110, len(quarters))
    
    # Revenue = units * AOV
    retail_revenue = units_sold * aov
    
    # Operating expenses (as % of revenue, improving over time)
    opex_pct = np.linspace(72, 58, len(quarters))
    operating_expenses = retail_revenue * (opex_pct / 100)
    
    # Gross margin
    gross_margin_pct = np.linspace(38, 45, len(quarters))
    
    df_retail = pd.DataFrame({
        'Period': quarters,
        'Units_Sold': units_sold.astype(int),
        'Avg_Order_Value': aov.round(2),
        'Total_Revenue': retail_revenue.astype(int),
        'Operating_Expenses': operating_expenses.astype(int),
        'Gross_Margin_Pct': gross_margin_pct.round(1)
    })
    
    # Save as Excel
    df_retail.to_excel('sample_quarterly_retail.xlsx', index=False)
    print("✅ Created 'sample_quarterly_retail.xlsx'")
    print(f"   Sample: Q4 2024 Revenue = ${df_retail.iloc[-1]['Total_Revenue']:,.0f}")
    
    print("\nDone! 2 quarterly sample files ready for upload.")

if __name__ == "__main__":
    generate_quarterly_samples()
