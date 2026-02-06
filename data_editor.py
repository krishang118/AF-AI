"""
Data Editor Module
SQL-like operations for editing combined datasets
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Union, Optional


class DataEditor:
    """Provides SQL-like editing operations on DataFrames"""
    
    def __init__(self, df: pd.DataFrame):
        """Initialize with a dataframe to edit"""
        self.df = df.copy()  # Work on a copy
        self.edit_history: List[Dict[str, Any]] = []
        self.affected_cells: set = set()  # Track which cells were edited: {(row_idx, col_name), ...}
        
    def delete_columns(self, column_names: List[str]) -> pd.DataFrame:
        """
        Delete specified columns
        
        Args:
            column_names: List of column names to delete
        
        Returns:
            Modified DataFrame
        """
        existing_cols = [c for c in column_names if c in self.df.columns]
        
        if existing_cols:
            self.df = self.df.drop(columns=existing_cols)
            self.edit_history.append({
                'operation': 'delete_columns',
                'columns': existing_cols
            })
        
        return self.df
    
    def delete_rows(self, row_indices: List[int]) -> pd.DataFrame:
        """
        Delete specified rows by index
        
        Args:
            row_indices: List of row indices to delete (0-based)
        
        Returns:
            Modified DataFrame
        """
        # Filter out invalid indices
        valid_indices = [i for i in row_indices if 0 <= i < len(self.df)]
        
        if valid_indices:
            self.df = self.df.drop(self.df.index[valid_indices])
            self.df = self.df.reset_index(drop=True)
            self.edit_history.append({
                'operation': 'delete_rows',
                'indices': valid_indices
            })
        
        return self.df
    
    def replace_values(self, find: Any, replace: Any, 
                      columns: Optional[List[str]] = None,
                      case_sensitive: bool = True,
                      regex: bool = False) -> pd.DataFrame:
        """
        Find and replace values in the dataframe
        
        Args:
            find: Value to find
            replace: Value to replace with
            columns: Specific columns to search (None = all columns)
            case_sensitive: Whether string matching should be case-sensitive
            regex: Whether to use regex patterns
        
        Returns:
            Modified DataFrame
        """
        target_cols = columns if columns else self.df.columns.tolist()
        target_cols = [c for c in target_cols if c in self.df.columns]
        
        if not target_cols:
            return self.df
        
        for col in target_cols:
            # Store original values to track which cells actually changed
            original_values = self.df[col].copy()
            
            if pd.api.types.is_string_dtype(self.df[col]) or pd.api.types.is_object_dtype(self.df[col]):
                # String replacement
                if regex:
                    self.df[col] = self.df[col].str.replace(
                        str(find), str(replace), 
                        case=case_sensitive, 
                        regex=True
                    )
                else:
                    self.df[col] = self.df[col].replace(find, replace)
            else:
                # Numeric or other type replacement
                # Try to cast 'find' and 'replace' to match column type if they are strings
                current_find = find
                current_replace = replace
                
                if isinstance(find, str):
                    try:
                        if pd.api.types.is_integer_dtype(self.df[col]):
                            current_find = int(find)
                        elif pd.api.types.is_float_dtype(self.df[col]):
                            current_find = float(find)
                    except:
                        pass # Keep as string if cast fails
                
                if isinstance(replace, str):
                    try:
                        if pd.api.types.is_integer_dtype(self.df[col]):
                            current_replace = int(replace)
                        elif pd.api.types.is_float_dtype(self.df[col]):
                            current_replace = float(replace)
                    except:
                        pass
                        
                self.df[col] = self.df[col].replace(current_find, current_replace)
            
            # Track which cells actually changed
            for row_idx in range(len(self.df)):
                old_val = original_values.iloc[row_idx]
                new_val = self.df[col].iloc[row_idx]
                # Handle NaN comparison
                if pd.isna(old_val) and pd.isna(new_val):
                    continue  # Both NaN, no change
                elif old_val != new_val:
                    self.affected_cells.add((row_idx, col))
        
        self.edit_history.append({
            'operation': 'replace_values',
            'find': find,
            'replace': replace,
            'columns': target_cols
        })
        
        return self.df
    
    def add_column(self, name: str, values: Optional[List[Any]] = None,
                  constant: Any = None, position: int = -1) -> pd.DataFrame:
        """
        Add a new column
        
        Args:
            name: Column name
            values: List of values (must match row count), or None
            constant: Constant value for all rows, or None
            position: Column position (-1 = end)
        
        Returns:
            Modified DataFrame
        """
        if values is not None:
            # Use provided values
            if len(values) != len(self.df):
                raise ValueError(f"Values length ({len(values)}) doesn't match row count ({len(self.df)})")
            new_col_data = values
        elif constant is not None:
            # Use constant value
            new_col_data = [constant] * len(self.df)
        else:
            # Default to NA
            new_col_data = [pd.NA] * len(self.df)
        
        # Insert column
        if position == -1 or position >= len(self.df.columns):
            # Add at end
            self.df[name] = new_col_data
        else:
            # Insert at position
            self.df.insert(position, name, new_col_data)
        
        # Track all cells in the new column as affected
        for row_idx in range(len(self.df)):
            self.affected_cells.add((row_idx, name))
        
        self.edit_history.append({
            'operation': 'add_column',
            'name': name,
            'position': position
        })
        
        return self.df
    
    def add_row(self, values: Dict[str, Any] = None, position: int = -1) -> pd.DataFrame:
        """
        Add a new row
        
        Args:
            values: Dict of {column: value} for the new row
            position: Row position (-1 = end)
        
        Returns:
            Modified DataFrame
        """
        if values is None:
            values = {}
        
        # Create new row with NA for missing columns
        new_row = {col: values.get(col, pd.NA) for col in self.df.columns}
        new_row_df = pd.DataFrame([new_row])
        
        if position == -1 or position >= len(self.df):
            # Append at end
            self.df = pd.concat([self.df, new_row_df], ignore_index=True)
        else:
            # Insert at position
            top = self.df.iloc[:position]
            bottom = self.df.iloc[position:]
            self.df = pd.concat([top, new_row_df, bottom], ignore_index=True)
        
        self.edit_history.append({
            'operation': 'add_row',
            'values': values,
            'position': position
        })
        
        return self.df
    
    def mark_as_na(self, row_indices: List[int] = None, 
                   column_names: List[str] = None) -> pd.DataFrame:
        """
        Mark specific cells as N.A.
        
        Args:
            row_indices: List of row indices (None = all rows)
            column_names: List of column names (None = all columns)
        
        Returns:
            Modified DataFrame
        """
        # Default to all rows/columns if not specified
        target_rows = row_indices if row_indices is not None else list(range(len(self.df)))
        target_cols = column_names if column_names is not None else self.df.columns.tolist()
        
        # Filter valid indices/columns
        target_rows = [i for i in target_rows if 0 <= i < len(self.df)]
        target_cols = [c for c in target_cols if c in self.df.columns]
        
        # Set to NA
        for row_idx in target_rows:
            for col in target_cols:
                self.df.at[row_idx, col] = pd.NA
                # Track this specific cell as affected
                self.affected_cells.add((row_idx, col))
        
        self.edit_history.append({
            'operation': 'mark_as_na',
            'rows': target_rows,
            'columns': target_cols
        })
        
        return self.df
    
    def duplicate_column(self, source_col: str, new_name: str) -> pd.DataFrame:
        """
        Duplicate an existing column
        
        Args:
            source_col: Column to duplicate
            new_name: Name for the new column
        
        Returns:
            Modified DataFrame
        """
        if source_col not in self.df.columns:
            raise ValueError(f"Column '{source_col}' not found")
        
        self.df[new_name] = self.df[source_col].copy()
        
        self.edit_history.append({
            'operation': 'duplicate_column',
            'source': source_col,
            'new_name': new_name
        })
        
        return self.df
    
    def rename_column(self, old_name: str, new_name: str) -> pd.DataFrame:
        """
        Rename a column
        
        Args:
            old_name: Current column name
            new_name: New column name
        
        Returns:
            Modified DataFrame
        """
        if old_name not in self.df.columns:
            raise ValueError(f"Column '{old_name}' not found")
        
        self.df = self.df.rename(columns={old_name: new_name})
        
        self.edit_history.append({
            'operation': 'rename_column',
            'old_name': old_name,
            'new_name': new_name
        })
        
        return self.df
    
    def get_dataframe(self) -> pd.DataFrame:
        """Return the current state of the dataframe"""
        return self.df.copy()
    
    def get_edit_history(self) -> List[Dict[str, Any]]:
        """Return the history of all edits"""
        return self.edit_history.copy()
    
    def get_affected_cells(self) -> set:
        """Return set of cells that were modified: {(row_idx, col_name), ...}"""
        return self.affected_cells.copy()
    
    def undo_last_edit(self) -> pd.DataFrame:
        """Undo the last edit (not fully implemented - would need state snapshots)"""
        # This is a placeholder - true undo would require storing dataframe states
        if self.edit_history:
            self.edit_history.pop()
        return self.df
