#!/usr/bin/env python3
"""
CRITICAL BUG REPRODUCTION
User reports all values are 0 after column join
Testing with yearly and quarterly data
"""

import pandas as pd
from data_joiner import DataJoiner

print("="*80)
print("REPRODUCING ALL ZEROS BUG")
print("="*80)

# Test 1: Yearly data join
print("\n1. Testing YEARLY data join...")
try:
    df_yearly = pd.read_csv('test_data/yearly_data.csv')
    print(f"Loaded yearly data: {len(df_yearly)} rows")
    print(df_yearly.head())
    print(f"\nData types:\n{df_yearly.dtypes}")
    print(f"\nSample values:\n{df_yearly.iloc[0]}")
except Exception as e:
    print(f"Error loading yearly: {e}")

# Test 2: Quarterly data join
print("\n\n2. Testing QUARTERLY data join...")
try:
    df_quarterly = pd.read_csv('quarterly/sample_quarterly_saas.csv')
    print(f"Loaded quarterly data: {len(df_quarterly)} rows")
    print(df_quarterly.head())
    print(f"\nData types:\n{df_quarterly.dtypes}")
    print(f"\nSample values:\n{df_quarterly.iloc[0]}")
except Exception as e:
    print(f"Error loading quarterly: {e}")

# Test 3: Try a simple column join
print("\n\n3. Testing COLUMN JOIN...")
try:
    # Create two simple dataframes
    df1 = pd.DataFrame({
        'period': ['2020', '2021', '2022'],
        'Revenue': [100, 110, 121]
    })
    df2 = pd.DataFrame({
        'period': ['2020', '2021', '2022'],
        'Cost': [40, 45, 50]
    })
    
    print("DF1 before join:")
    print(df1)
    print(f"\nDF1 dtypes: {df1.dtypes}")
    
    print("\nDF2 before join:")
    print(df2)
    print(f"\nDF2 dtypes: {df2.dtypes}")
    
    joiner = DataJoiner()
    result = joiner.join_on_column({'Sheet1': df1, 'Sheet2': df2}, 'period', mode='reference')
    
    print("\nJOINED RESULT:")
    print(result)
    print(f"\nResult dtypes: {result.dtypes}")
    print(f"\nResult values:\n{result.values}")
    
    # Check if values are actually zero
    if (result.select_dtypes(include=['number']) == 0).all().all():
        print("\n🚨 BUG CONFIRMED: ALL NUMERIC VALUES ARE ZERO!")
    else:
        print("\n✅ Values look correct")
        
except Exception as e:
    print(f"Error during join: {e}")
    import traceback
    traceback.print_exc()

# Test 4: Test Excel export
print("\n\n4. Testing EXCEL EXPORT...")
try:
    import tempfile
    output_file = tempfile.mktemp(suffix='.xlsx')
    result.to_excel(output_file, index=False)
    
    # Read it back
    df_read = pd.read_excel(output_file)
    print("Data read back from Excel:")
    print(df_read)
    
    if (df_read.select_dtypes(include=['number']) == 0).all().all():
        print("\n🚨 BUG IN EXCEL EXPORT: ALL VALUES ARE ZERO!")
    else:
        print("\n✅ Excel export looks correct")
        
except Exception as e:
    print(f"Error during Excel export: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*80)
