"""
Excel Formula Exporter with Cached Values
Uses xlsxwriter to write formulas WITH calculated values
This ensures both formula references AND pandas readability
"""

import pandas as pd
import xlsxwriter
from typing import Dict, List, Any
import re


def sanitize_sheet_name(name: str) -> str:
    """Make sheet name Excel-safe"""
    name = re.sub(r'[\[\]:*?/\\]', '_', name)
    name = name[:31]
    name = name.strip("'")
    if not name:
        name = "Sheet"
    return name


def export_combined_excel_with_formulas(
    joined_df: pd.DataFrame,
    source_mapping: Dict,
    source_dataframes: Dict[str, pd.DataFrame],
    output_path: str,
    join_type: str = 'column_join'
):
    """
    Export Excel with formulas AND cached values using xlsxwriter
    
    Args:
        joined_df: Combined DataFrame
        source_mapping: Mapping from DataJoiner
        source_dataframes: Dict of {sheet_name: DataFrame}
        output_path: Output file path
        join_type: 'column_join' or 'row_append'
    """
    
    # Create workbook
    workbook = xlsxwriter.Workbook(output_path)
    
    # Formats
    header_format = workbook.add_format({
        'bold': True,
        'bg_color': '#D9E1F2',
        'border': 1
    })
    
    # Sanitize sheet names
    sanitized_names = {}
    for original_name in source_dataframes.keys():
        san = sanitize_sheet_name(original_name)
        base = san
        counter = 1
        while san in sanitized_names.values():
            san = f"{base}_{counter}"
            counter += 1
        sanitized_names[original_name] = san
    
    # Write source sheets
    for original_name, df in source_dataframes.items():
        sheet_name = sanitized_names[original_name]
        worksheet = workbook.add_worksheet(sheet_name)
        
        # Write headers
        for col_idx, col_name in enumerate(df.columns):
            worksheet.write(0, col_idx, str(col_name), header_format)
        
        # Write data
        for row_idx in range(len(df)):
            for col_idx, col_name in enumerate(df.columns):
                value = df.iloc[row_idx, col_idx]
                if pd.isna(value):
                    worksheet.write(row_idx + 1, col_idx, None)
                else:
                    worksheet.write(row_idx + 1, col_idx, value)
        
        # Auto-size columns to prevent #### issue
        for col_idx, col_name in enumerate(df.columns):
            # Calculate max width needed
            max_len = len(str(col_name))  # Header length
            for val in df.iloc[:, col_idx]:
                if pd.notna(val):
                    max_len = max(max_len, len(str(val)))
            # Set width with some padding (minimum 12 for dates)
            worksheet.set_column(col_idx, col_idx, max(12, min(max_len + 2, 50)))
    
    # Create Combined_Data sheet
    combined_sheet = workbook.add_worksheet('Combined_Data')
    
    # Write headers
    for col_idx, col_name in enumerate(joined_df.columns):
        combined_sheet.write(0, col_idx, str(col_name), header_format)
    
    # Pre-calculate column widths for Combined_Data
    col_widths = []
    for col_idx, col_name in enumerate(joined_df.columns):
        max_len = len(str(col_name))
        for val in joined_df.iloc[:, col_idx]:
            if pd.notna(val):
                max_len = max(max_len, len(str(val)))
        col_widths.append(max(12, min(max_len + 2, 50)))
    
    # Write data with formulas
    if join_type == 'column_join':
        # Update mapping with sanitized names
        updated_mapping = {
            col: sanitized_names.get(sheet, sheet)
            for col, sheet in source_mapping.items()
        }
        updated_source_dfs = {
            sanitized_names.get(name, name): df
            for name, df in source_dataframes.items()
        }
        
        for row_idx in range(len(joined_df)):
            excel_row = row_idx + 1  # +1 for header
            
            for col_idx, col_name in enumerate(joined_df.columns):
                # Get actual value
                actual_value = joined_df.iloc[row_idx, col_idx]
                
                # Get source info
                source_sheet = updated_mapping.get(col_name)
                
                if source_sheet and source_sheet in updated_source_dfs:
                    source_df = updated_source_dfs[source_sheet]
                    
                    # Find column index in source
                    try:
                        source_col_idx = source_df.columns.get_loc(col_name)
                        source_row = row_idx + 1  # Same row alignment
                        
                        # Create formula with cached value
                        # xlsxwriter syntax: write_formula(row, col, formula, format, value)
                        formula = f"='{source_sheet}'!{xlsxwriter.utility.xl_col_to_name(source_col_idx)}{source_row + 1}"
                        
                        if pd.isna(actual_value):
                            combined_sheet.write_formula(excel_row, col_idx, formula, None, None)
                        else:
                            combined_sheet.write_formula(excel_row, col_idx, formula, None, actual_value)
                    except (KeyError, ValueError):
                        # Column not in source, write value
                        if pd.isna(actual_value):
                            combined_sheet.write(excel_row, col_idx, None)
                        else:
                            combined_sheet.write(excel_row, col_idx, actual_value)
                else:
                    # No mapping, write value
                    if pd.isna(actual_value):
                        combined_sheet.write(excel_row, col_idx, None)
                    else:
                        combined_sheet.write(excel_row, col_idx, actual_value)
    
    else:  # row_append
        # Update mapping with sanitized names
        updated_mapping = {
            row_idx: {
                'sheet': sanitized_names.get(info['sheet'], info['sheet']),
                'source_row': info['source_row']
            }
            for row_idx, info in source_mapping.items()
        }
        updated_source_dfs = {
            sanitized_names.get(name, name): df
            for name, df in source_dataframes.items()
        }
        
        for row_idx in range(len(joined_df)):
            excel_row = row_idx + 1
            
            row_mapping = updated_mapping.get(row_idx, {})
            source_sheet = row_mapping.get('sheet')
            source_row_idx = row_mapping.get('source_row')
            
            if source_sheet and source_sheet in updated_source_dfs:
                source_df = updated_source_dfs[source_sheet]
                
                for col_idx, col_name in enumerate(joined_df.columns):
                    actual_value = joined_df.iloc[row_idx, col_idx]
                    
                    try:
                        source_col_idx = source_df.columns.get_loc(col_name)
                        formula = f"='{source_sheet}'!{xlsxwriter.utility.xl_col_to_name(source_col_idx)}{source_row_idx + 2}"
                        
                        if pd.isna(actual_value):
                            combined_sheet.write_formula(excel_row, col_idx, formula, None, None)
                        else:
                            combined_sheet.write_formula(excel_row, col_idx, formula, None, actual_value)
                    except (KeyError, ValueError):
                        if pd.isna(actual_value):
                            combined_sheet.write(excel_row, col_idx, None)
                        else:
                            combined_sheet.write(excel_row, col_idx, actual_value)
            else:
                # No mapping, write values
                for col_idx, col_name in enumerate(joined_df.columns):
                    actual_value = joined_df.iloc[row_idx, col_idx]
                    if pd.isna(actual_value):
                        combined_sheet.write(excel_row, col_idx, None)
                    else:
                        combined_sheet.write(excel_row, col_idx, actual_value)
    
    # Apply column widths to Combined_Data to prevent #### issue
    for col_idx, width in enumerate(col_widths):
        combined_sheet.set_column(col_idx, col_idx, width)
    
    workbook.close()
    return output_path
