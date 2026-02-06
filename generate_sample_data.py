"""
Sample Data Generator
Creates rich, extensive example Excel files for testing the forecasting system.
Expanded schema: 15+ years of data, detailed granular metrics, operational KPIs.
Designed to provide clear signals for AI analysis.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from openpyxl.styles import Font, PatternFill

def generate_financials():
    # 16 years of data (2010-2025)
    years = list(range(2010, 2026))
    data = []
    
    # Base params (Starting 2010)
    active_customers = 500
    churn_rate = 0.20
    arpu_sub = 1000  # Subscription ARPU
    arpu_svc = 200   # Services ARPU (Implementation fees etc)
    
    marketing_spend = 150000
    sales_headcount = 2
    eng_headcount = 3
    
    # Historical logic
    for year in years:
        # Contextual Market Events (Simulation)
        market_sentiment = 1.0
        if year == 2020: market_sentiment = 0.8  # Covid impact
        if year == 2021: market_sentiment = 1.2  # Tech boom
        
        # 1. Operational Drivers
        marketing_efficiency = 800 if year < 2015 else 1200 # CAC gets expensive
        new_customers = int((marketing_spend / marketing_efficiency) * market_sentiment)
        lost_customers = int(active_customers * churn_rate)
        
        active_customers = max(100, active_customers + new_customers - lost_customers)
        
        # 2. Revenue Breakdown
        sub_revenue = active_customers * arpu_sub
        svc_revenue = active_customers * arpu_svc * 0.2 # Only 20% attach rate for services
        total_revenue = sub_revenue + svc_revenue
        
        # 3. Cost Breakdown
        # COGS
        hosting_costs = int(total_revenue * 0.12)
        support_headcount = int(active_customers / 500) + 1
        support_costs = support_headcount * 80000
        total_cogs = hosting_costs + support_costs
        
        gross_profit = total_revenue - total_cogs
        
        # OpEx
        # R&D
        eng_headcount = int(eng_headcount * 1.15) # Growing eng team
        rd_spend = eng_headcount * 150000 + (total_revenue * 0.05) # Salaries + Infra
        
        # S&M
        sales_headcount = int(sales_headcount * 1.2)
        sales_spend = sales_headcount * 120000 + (new_customers * 50) # Commissions
        total_marketing = marketing_spend + sales_spend
        
        # G&A
        ga_spend = total_revenue * 0.12 # Scale efficiency? No, fixed ratio for now
        
        total_opex = rd_spend + total_marketing + ga_spend
        ebitda = gross_profit - total_opex
        
        # Net Income (Tax simplified)
        tax = max(0, ebitda * 0.21)
        net_income = ebitda - tax
        
        # Add entry
        data.append({
            'Year': year,
            # Top Line
            'Total_Revenue': int(total_revenue),
            'Subscription_Revenue': int(sub_revenue),
            'Services_Revenue': int(svc_revenue),
            # COGS
            'COGS': int(total_cogs),
            'Hosting_Costs': int(hosting_costs),
            'Support_Costs': int(support_costs),
            'Gross_Profit': int(gross_profit),
            # OpEx
            'Total_OpEx': int(total_opex),
            'Marketing_Spend': int(marketing_spend),
            'Sales_Spend': int(sales_spend),
            'R_and_D_Spend': int(rd_spend),
            'G_and_A_Spend': int(ga_spend),
            # Bottom Line
            'EBITDA': int(ebitda),
            'Net_Income': int(net_income),
            # KPIs
            'Active_Customers': int(active_customers),
            'New_Customers': int(new_customers),
            'Churn_Rate': float(f"{churn_rate:.3f}"),
            'ARPU_Blended': int((sub_revenue + svc_revenue) / active_customers),
            'Headcount_Total': support_headcount + eng_headcount + sales_headcount + 5, # +5 Execs
            'Headcount_Eng': eng_headcount,
            'Headcount_Sales': sales_headcount
        })
        
        # Evolve drivers for next year
        marketing_spend *= 1.15
        arpu_sub *= 1.02 # Infation
        churn_rate = max(0.05, churn_rate * 0.92) # Retention improving logic
        
    return pd.DataFrame(data)

def generate_market_data():
    years = list(range(2010, 2026))
    data = []
    
    market_size_tam = 100000000 # 100M start
    
    for year in years:
        # Market growth cycle
        growth_rate = 0.08
        if 2015 <= year <= 2018: growth_rate = 0.12 # Growth spurt
        if year == 2020: growth_rate = 0.02 # Covid slow
        if year >= 2023: growth_rate = 0.06 # Maturity
        
        market_size_tam *= (1 + growth_rate)
        
        som_share = 0.01 + ((year - 2010) * 0.005) # Slow capture
        som_size = market_size_tam * som_share
        
        data.append({
            'Period': year,
            'Global_TAM': int(market_size_tam),
            'Served_Market_SAM': int(market_size_tam * 0.4), # 40% addressable
            'Market_Share_SOM': float(f"{som_share:.4f}"),
            'Competitor_A_Share': float(f"{0.30 - (som_share * 0.5):.4f}"), # We take share from A
            'Competitor_B_Share': 0.15, # Steady
            'Inflation_Index': round(1.0 + (year-2010)*0.02, 3), # Macro indicator
            'Industry_Growth_Rate': float(f"{growth_rate:.3f}")
        })
        
    return pd.DataFrame(data)

def create_analyst_notes():
    return """
FORECAST VALIDATION NOTES - 2025

HISTORICAL CONTEXT:
- 2010-2015: bootstrapping phase. High churn, low efficiency.
- 2016-2019: Series A growth. Sales headcount expansion.
- 2020: Covid stagnated new logos but retention held strong.
- 2021-2024: Scale-up phase. Margins improved via hosting optimization.

KEY DRIVERS FOR 2026+:
1. Headcount Efficiency: Engineering leverage is kicking in. R&D % of rev should drop.
2. Services Attach: We are pushing services revenue up to 25% of mix.
3. Churn: Best-in-class at <6%. Maintain this.

RISKS:
- Competitor A is aggressive on price (check Price/Volume variance).
- Tech debt from 2016 era might increase Hosting Costs.
"""

def generate_all_samples():
    output_dir = Path("sample_data")
    output_dir.mkdir(exist_ok=True)
    
    print("Generating expanded dataset (2010-2025)...")
    
    # 1. Financials
    fin_df = generate_financials()
    fin_path = output_dir / "financial_history.xlsx"
    
    with pd.ExcelWriter(fin_path, engine='openpyxl') as writer:
        fin_df.to_excel(writer, sheet_name='Financials', index=False)
        # Apply strict formatting
        for cell in writer.sheets['Financials'][1]:
            cell.font = Font(bold=True, color='FFFFFF')
            cell.fill = PatternFill(start_color='366092', end_color='366092', fill_type='solid')
            
    print(f"Created: {fin_path} ({len(fin_df)} rows, {len(fin_df.columns)} cols)")
    
    # 2. Market
    mkt_df = generate_market_data()
    mkt_path = output_dir / "market_data.csv"
    mkt_df.to_csv(mkt_path, index=False)
    print(f"Created: {mkt_path} ({len(mkt_df)} rows)")
    
    # 3. Notes
    notes_path = output_dir / "analyst_notes.txt"
    with open(notes_path, 'w') as f:
        f.write(create_analyst_notes())
    print(f"Created: {notes_path}")
    
    print(f"\nExpanded samples files generated in: {output_dir.absolute()}")
    print("Use these files in the 'Data Upload' tab.")

if __name__ == "__main__":
    generate_all_samples()
