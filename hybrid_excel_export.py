"""
Hybrid Excel Export - Formulas for unedited cells, values for edited cells
Uses xlsxwriter for formula+cached value support
"""

import pandas as pd
import xlsxwriter
from typing import Dict, Set, Any
import re


def sanitize_sheet_name(name: str) -> str:
    """Make sheet name Excel-safe"""
    name = re.sub(r'[\[\]:*?/\\]', '_', name)
    name = name[:31]
    name = name.strip("'")
    if not name:
        name = "Sheet"
    return name


def export_hybrid_excel(
    combined_df: pd.DataFrame,
    affected_cells: Set[tuple],
    source_mapping: Dict,
    source_dataframes: Dict[str, pd.DataFrame],
    output_path: str,
    join_type: str = 'column_join'
):
    """
    Hybrid export: formulas for unchanged cells, values for edited cells
    
    Args:
        combined_df: Combined DataFrame (possibly with edits)
        affected_cells: Set of (row_idx, col_name) tuples for edited cells
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
    
    edited_format = workbook.add_format({
        'bg_color': '#FFF2CC'  # Light yellow for edited cells
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
            max_len = len(str(col_name))
            for val in df.iloc[:, col_idx]:
                if pd.notna(val):
                    max_len = max(max_len, len(str(val)))
            worksheet.set_column(col_idx, col_idx, max(12, min(max_len + 2, 50)))
    
    # Create Combined_Data sheet
    combined_sheet = workbook.add_worksheet('Combined_Data')
    
    # Write headers
    for col_idx, col_name in enumerate(combined_df.columns):
        combined_sheet.write(0, col_idx, str(col_name), header_format)
    
    # Pre-calculate column widths
    col_widths = []
    for col_idx, col_name in enumerate(combined_df.columns):
        max_len = len(str(col_name))
        for val in combined_df.iloc[:, col_idx]:
            if pd.notna(val):
                max_len = max(max_len, len(str(val)))
        col_widths.append(max(12, min(max_len + 2, 50)))
    
    # Write data - hybrid approach
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
        
        for row_idx in range(len(combined_df)):
            excel_row = row_idx + 1
            
            for col_idx, col_name in enumerate(combined_df.columns):
                actual_value = combined_df.iloc[row_idx, col_idx]
                
                # Check if this cell was edited
                is_edited = (row_idx, col_name) in affected_cells
                
                if is_edited:
                    # Write value only (with highlight)
                    if pd.isna(actual_value):
                        combined_sheet.write(excel_row, col_idx, None, edited_format)
                    else:
                        combined_sheet.write(excel_row, col_idx, actual_value, edited_format)
                else:
                    # Write formula with cached value
                    source_sheet = updated_mapping.get(col_name)
                    
                    if source_sheet and source_sheet in updated_source_dfs:
                        source_df = updated_source_dfs[source_sheet]
                        
                        try:
                            source_col_idx = source_df.columns.get_loc(col_name)
                            source_row = row_idx  # 0-indexed row in source
                            
                            # Check if this row exists in the source (for new rows added after join)
                            if source_row >= len(source_df):
                                # This is a NEW row that doesn't exist in source - write value only
                                if pd.isna(actual_value):
                                    combined_sheet.write(excel_row, col_idx, None)
                                else:
                                    combined_sheet.write(excel_row, col_idx, actual_value)
                                continue  # Skip formula generation
                            
                            formula = f"='{source_sheet}'!{xlsxwriter.utility.xl_col_to_name(source_col_idx)}{source_row + 2}"
                            
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
                        if pd.isna(actual_value):
                            combined_sheet.write(excel_row, col_idx, None)
                        else:
                            combined_sheet.write(excel_row, col_idx, actual_value)
    
    else:  # row_append
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
        
        for row_idx in range(len(combined_df)):
            excel_row = row_idx + 1
            
            row_mapping = updated_mapping.get(row_idx, {})
            source_sheet = row_mapping.get('sheet')
            source_row_idx = row_mapping.get('source_row')
            
            for col_idx, col_name in enumerate(combined_df.columns):
                actual_value = combined_df.iloc[row_idx, col_idx]
                is_edited = (row_idx, col_name) in affected_cells
                
                if is_edited:
                    if pd.isna(actual_value):
                        combined_sheet.write(excel_row, col_idx, None, edited_format)
                    else:
                        combined_sheet.write(excel_row, col_idx, actual_value, edited_format)
                else:
                    if source_sheet and source_sheet in updated_source_dfs:
                        source_df = updated_source_dfs[source_sheet]
                        
                        try:
                            source_col_idx = source_df.columns.get_loc(col_name)
                            
                            # Check if source_row_idx exists in source (for new rows)
                            if source_row_idx is None or source_row_idx >= len(source_df):
                                # New row, write value only
                                if pd.isna(actual_value):
                                    combined_sheet.write(excel_row, col_idx, None)
                                else:
                                    combined_sheet.write(excel_row, col_idx, actual_value)
                                continue  # Skip formula
                            
                            formula = f"='{ source_sheet}'!{xlsxwriter.utility.xl_col_to_name(source_col_idx)}{source_row_idx + 2}"
                            
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
                        if pd.isna(actual_value):
                            combined_sheet.write(excel_row, col_idx, None)
                        else:
                            combined_sheet.write(excel_row, col_idx, actual_value)
    
    # Apply column widths to Combined_Data
    for col_idx, width in enumerate(col_widths):
        combined_sheet.set_column(col_idx, col_idx, width)
    
    workbook.close()
    return output_path
