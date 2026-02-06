"""
Data Joiner Module
Handles column-based joining and row-based appending with copy/reference modes
"""

import pandas as pd
from typing import List, Dict, Optional, Any
from pathlib import Path


class DataJoiner:
    """Performs smart data joining operations with multiple modes"""
    
    def __init__(self):
        self.joined_df: Optional[pd.DataFrame] = None
        self.source_mapping: Dict[str, Any] = {}  # Track cell origins for reference mode
        
    def join_on_column(self, dataframes: Dict[str, pd.DataFrame], join_key: str, 
                      mode: str = 'copy') -> pd.DataFrame:
        """
        Join multiple dataframes on a common column
        
        Args:
            dataframes: Dict of {sheet_name: DataFrame}
            join_key: Column name to join on
            mode: 'copy' (values) or 'reference' (formulas)
        
        Returns:
            Joined DataFrame
        """
        if not dataframes or len(dataframes) < 2:
            raise ValueError("Need at least 2 dataframes to join")
            
        # Validate join key exists
        for sheet_name, df in dataframes.items():
            if join_key not in df.columns:
                raise ValueError(f"Join key '{join_key}' not found in sheet '{sheet_name}'")
        
        # Validate and Normalize Join Keys
        processed_dfs = []
        is_datetime_target = False
        
        # 1. First pass: Detect if any DF has datetime key
        for df in dataframes.values():
            if pd.api.types.is_datetime64_any_dtype(df[join_key]):
                is_datetime_target = True
                break
        
        # 2. Second pass: Normalize
        for df in dataframes.values():
            df_copy = df.copy()
            if is_datetime_target:
                try:
                    # Try to coerce to datetime
                    df_copy[join_key] = pd.to_datetime(df_copy[join_key], errors='coerce')
                except:
                    # Fallback to string if coercion fails heavily (shouldn't happen with coerce)
                    df_copy[join_key] = df_copy[join_key].astype(str)
            else:
                # Ensure all are strings if not datetime
                df_copy[join_key] = df_copy[join_key].astype(str)
                
            processed_dfs.append(df_copy)
            
        # Perform outer join
        from functools import reduce
        
        # Use processed DFs for merging
        if not processed_dfs:
            processed_dfs = list(dataframes.values())

        joined = reduce(
            lambda left, right: pd.merge(left, right, on=join_key, how='outer'),
            processed_dfs
        )
        
        # Track source mapping for reference mode
        if mode == 'reference':
            self.source_mapping = self._build_join_mapping(dataframes, join_key, joined)
        
        self.joined_df = joined
        return joined
    
    def append_rows(self, dataframes: Dict[str, pd.DataFrame], 
                   align_columns: List[str] = None,
                   include_non_common: bool = False,
                   mode: str = 'copy') -> pd.DataFrame:
        """
        Append/stack rows from multiple dataframes
        
        Args:
            dataframes: Dict of {sheet_name: DataFrame}
            align_columns: List of columns to align on (None = use all)
            include_non_common: Include non-aligned columns with NA
            mode: 'copy' (values) or 'reference' (formulas)
        
        Returns:
            Appended DataFrame
        """
        if not dataframes:
            raise ValueError("No dataframes provided")
        
        selected_dfs = []
        
        if align_columns:
            # Use only specified columns
            for sheet_name, df in dataframes.items():
                if include_non_common:
                    # Keep all columns
                    selected_dfs.append(df)
                else:
                    # Only keep aligned columns that exist
                    available_cols = [c for c in align_columns if c in df.columns]
                    if available_cols:
                        selected_dfs.append(df[available_cols])
        else:
            # Use all columns
            selected_dfs = list(dataframes.values())
        
        # Stack rows
        appended = pd.concat(selected_dfs, ignore_index=True, sort=False)
        
        # Track source mapping for reference mode
        if mode == 'reference':
            self.source_mapping = self._build_append_mapping(dataframes, appended, selected_dfs)
        
        self.joined_df = appended
        return appended
    
    def _build_join_mapping(self, dataframes: Dict[str, pd.DataFrame], 
                           join_key: str, result_df: pd.DataFrame) -> Dict:
        """Build mapping of result cells to source sheets (for reference mode)"""
        mapping = {}
        
        # For each column in result, track which source sheet it came from
        sheet_names = list(dataframes.keys())
        
        for col in result_df.columns:
            if col == join_key:
                # Join key comes from first sheet
                mapping[col] = sheet_names[0]
            else:
                # Find which sheet this column came from
                for sheet_name, df in dataframes.items():
                    if col in df.columns and col != join_key:
                        mapping[col] = sheet_name
                        break
        
        return mapping
    
    def _build_append_mapping(self, dataframes: Dict[str, pd.DataFrame],
                             result_df: pd.DataFrame, source_dfs: List[pd.DataFrame]) -> Dict:
        """Build mapping of result rows to source sheets (for reference mode)"""
        mapping = {}
        
        row_offset = 0
        sheet_names = list(dataframes.keys())
        
        for i, (sheet_name, source_df) in enumerate(zip(sheet_names, source_dfs)):
            num_rows = len(source_df)
            # Map result rows to source sheet and row
            for j in range(num_rows):
                mapping[row_offset + j] = {
                    'sheet': sheet_name,
                    'source_row': j
                }
            row_offset += num_rows
        
        return mapping
    
    def get_source_mapping(self) -> Dict:
        """Return the source mapping for reference mode"""
        return self.source_mapping
