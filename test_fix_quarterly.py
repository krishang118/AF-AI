#!/usr/bin/env python3
"""
Test the fix with ACTUAL quarterly and yearly data
"""

import pandas as pd
from data_joiner import DataJoiner
from excel_formula_exporter import export_combined_excel_with_formulas
import tempfile

print("="*80)
print("TESTING FIX WITH ACTUAL QUARTERLY DATA")
print("="*80)

# Load quarterly data
df_quarterly = pd.read_csv('quarterly/sample_quarterly_saas.csv')
print(f"\nQuarterly data loaded: {len(df_quarterly)} rows")
print(df_quarterly.head(3))

# Split into 2 sheets to test join
df_q1 = df_quarterly[['Quarter', 'Quarterly_Revenue', 'ARR']].copy()
df_q2 = df_quarterly[['Quarter', 'Total_Customers', 'Churn_Rate_Pct']].copy()

print(f"\nSheet 1: {df_q1.columns.tolist()}")
print(f"Sheet 2: {df_q2.columns.tolist()}")

# Join them
joiner = DataJoiner()
combined = joiner.join_on_column({'Revenue_Data': df_q1, 'Customer_Data': df_q2}, 'Quarter', mode='reference')

print(f"\nCombined data: {len(combined)} rows × {len(combined.columns)} columns")
print(combined.head())

# Export
output_file = '/tmp/quarterly_test.xlsx'
export_combined_excel_with_formulas(
    joined_df=combined,
    source_mapping=joiner.get_source_mapping(),
    source_dataframes={'Revenue_Data': df_q1, 'Customer_Data': df_q2},
    output_path=output_file,
    join_type='column_join'
)

# Read back
df_readback = pd.read_excel(output_file, sheet_name='Combined_Data')
print(f"\nRead back from Excel:")
print(df_readback.head())

# Check if values are correct
if (df_readback.select_dtypes(include=['number']) == 0).all().all():
    print("\n❌ STILL BROKEN - Values are zero")
elif df_readback.empty or len(df_readback) == 0:
    print("\n❌ STILL BROKEN - Empty DataFrame")
else:
    print(f"\n✅ FIX CONFIRMED! {len(df_readback)} rows with actual values")
    print(f"Sample values: Revenue={df_readback['Quarterly_Revenue'].iloc[0]}, Customers={df_readback['Total_Customers'].iloc[0]}")

print("\n" + "="*80)
print(f"File saved at: {output_file}")
print("Open in Excel to verify!")
