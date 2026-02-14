"""
Data Upload Wizard Module
Contains the new multi-step data preparation wizard
"""

import streamlit as st
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any
from data_joiner import DataJoiner
from data_editor import DataEditor
from excel_builder import ExcelReferenceBuilder, create_copy_table


def init_wizard_state():
    """Initialize wizard session state variables"""
    if 'wizard_step' not in st.session_state:
        st.session_state.wizard_step = 1
    if 'uploaded_temp_paths' not in st.session_state:
        st.session_state.uploaded_temp_paths = []
    if 'selected_sheets' not in st.session_state:
        st.session_state.selected_sheets = []
    if 'join_strategy' not in st.session_state:
        st.session_state.join_strategy = None
    if 'join_key' not in st.session_state:
        st.session_state.join_key = None
    if 'output_mode' not in st.session_state:
        st.session_state.output_mode = 'copy'
    if 'combined_df' not in st.session_state:
        st.session_state.combined_df = None
    if 'data_editor' not in st.session_state:
        st.session_state.data_editor = None


def data_upload_wizard():
    """Main wizard entry point - replaces old data_upload_page"""
    
    init_wizard_state()
    
    st.header("Data Preparation")
    
    # Progress indicator (4 steps now - removed Output Mode)
    steps = ["Upload", "Join Strategy", "Edit (Optional)", "Finalize"]
    cols = st.columns(4)
    for i, (col, step_name) in enumerate(zip(cols, steps), start=1):
        with col:
            if i < st.session_state.wizard_step:
                st.markdown(f"**{step_name}** (Done)")
            elif i == st.session_state.wizard_step:
                st.markdown(f"**{step_name}** (Current)")
            else:
                st.markdown(f"{step_name}")
    
    st.divider()
    
    # Route to appropriate step
    if st.session_state.wizard_step == 1:
        step_1_upload()
    elif st.session_state.wizard_step == 2:
        step_2_join_strategy()
    elif st.session_state.wizard_step == 3:
        step_3_edit_mode()  # Renumbered from 4
    elif st.session_state.wizard_step == 4:
        step_4_finalize()  # Renumbered from 5


def step_1_upload():
    """Step 1: File Upload"""
    st.subheader("Step 1: Upload Your Data Files")
    
    st.markdown("""
    Upload CSV, Excel (XLSX/XLS), or Word documents (if they contain table data).
    You can upload multiple files to combine them in the next step.
    """)
    
    uploaded_files = st.file_uploader(
        "Choose files",
        type=['xlsx', 'xls', 'csv', 'docx'],
        accept_multiple_files=True,
        key="wizard_uploader"
    )
    
    if uploaded_files:
        # Save to temp and split multi-sheet Excel files
        temp_paths = []
        
        for uploaded_file in uploaded_files:
            # Save original file first
            temp_path = Path(f"/tmp/{uploaded_file.name}")
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # Check for multi-sheet Excel
            if uploaded_file.name.endswith(('.xlsx', '.xls')):
                try:
                    excel_file = pd.ExcelFile(temp_path)
                    
                    if len(excel_file.sheet_names) > 1:
                        # Multi-sheet detected - split into individual files
                        st.info(f"Detected {len(excel_file.sheet_names)} sheets in **{uploaded_file.name}**. Splitting into individual files...")
                        
                        base_name = temp_path.stem  # filename without extension
                        
                        for sheet_name in excel_file.sheet_names:
                            # Read individual sheet
                            df = pd.read_excel(excel_file, sheet_name=sheet_name)
                            
                            # Create split file path with sanitized sheet name
                            safe_sheet_name = sheet_name.replace('/', '_').replace('\\', '_')
                            split_path = Path(f"/tmp/{base_name}_{safe_sheet_name}.xlsx")
                            
                            # Save as individual Excel file
                            df.to_excel(split_path, index=False, sheet_name=sheet_name)
                            
                            temp_paths.append(split_path)
                        
                        # Remove original multi-sheet file
                        temp_path.unlink()
                        
                        # Show split files
                        with st.expander(f"Split files from {uploaded_file.name}"):
                            for path in temp_paths[-len(excel_file.sheet_names):]:
                                st.caption(f"• {path.name}")
                    else:
                        # Single sheet - keep as-is
                        temp_paths.append(temp_path)
                except Exception as e:
                    # If reading fails, keep original file
                    st.warning(f"Could not process **{uploaded_file.name}**: {str(e)}")
                    temp_paths.append(temp_path)
            else:
                # CSV or non-Excel file - keep as-is
                temp_paths.append(temp_path)
        
        st.session_state.uploaded_temp_paths = temp_paths
        
        # Summary message
        if len(temp_paths) != len(uploaded_files):
            st.success(f"Processed {len(uploaded_files)} uploaded file(s) → {len(temp_paths)} file(s) ready for analysis")
        
        # Load files into normalizer
        with st.spinner("Loading files..."):
            warnings = st.session_state.normalizer.load_files(temp_paths)
            
            if warnings['errors']:
                st.error("**Errors:**")
                for error in warnings['errors']:
                    st.error(f"- {error}")
                return
            
            if warnings['info']:
                st.success("**Loaded:**")
                for info in warnings['info']:
                    st.success(f"- {info}")
        
        # Show preview of each loaded sheet
        st.subheader("File Previews")
        for sheet_name, df in st.session_state.normalizer.data_frames.items():
            with st.expander(f"{sheet_name} ({len(df)} rows, {len(df.columns)} columns)"):
                st.dataframe(df.head(10), use_container_width=True)
        
        # Progress to next step
        st.divider()
        col1, col2 = st.columns([1, 4])
        with col2:
            if st.button("Next: Choose Join Strategy →", type="primary", use_container_width=True):
                st.session_state.wizard_step = 2
                st.rerun()


def step_2_join_strategy():
    """Step 2: Choose Join/Append Strategy"""
    st.subheader("Step 2: How Should We Combine Your Data?")
    
    num_sheets = len(st.session_state.normalizer.data_frames)
    
    if num_sheets == 1:
        st.info("You uploaded only one file. You can skip directly to the next step.")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("← Back", use_container_width=True):
                st.session_state.wizard_step = 1
                st.rerun()
        with col2:
            if st.button("Skip →", type="primary", use_container_width=True):
                # Use the single sheet as-is
                sheet_name = list(st.session_state.normalizer.data_frames.keys())[0]
                df = st.session_state.normalizer.data_frames[sheet_name].copy()
                
                # ESSENTIAL: Normalize date column to 'period' like unify_data does
                date_col = st.session_state.normalizer.detect_date_column(df)
                if date_col:
                    df = df.rename(columns={date_col: 'period'})
                    
                    # Smart date handling (same as multi-file mode):
                    # - Excel date serials (e.g., 45292) → convert to dates
                    # - Years (2010, 2011) → keep as integers
                    # - String dates → keep as strings (no time added)
                    if pd.api.types.is_numeric_dtype(df['period']):
                        min_val = df['period'].min()
                        max_val = df['period'].max()
                        
                        # Excel date serial numbers (typically 25000-60000 for 1968-2064)
                        if min_val > 25000 and max_val < 60000:
                            # Convert Excel serial dates to actual dates
                            df['period'] = pd.to_datetime(df['period'], unit='D', origin='1899-12-30')
                            # Format as date-only strings (no time)
                            df['period'] = df['period'].dt.strftime('%Y-%m-%d')
                        # Years (1900-2100) - keep as integers
                        elif min_val > 1900 and max_val < 2100:
                            pass  # Keep as integers
                
                st.session_state.combined_df = df
                st.session_state.join_strategy = 'skip'
                st.session_state.wizard_step = 3  # Go to Edit step instead of Finalize
                st.rerun()
        return
    
    # Strategy selection
    strategy = st.radio(
        "Choose combination strategy:",
        ["Join Columns (Merge by Key)", "Append Rows (Stack Vertically)"],
        help="Join: Combines different metrics for the same periods. Append: Stacks datasets with similar columns."
    )
    
    st.session_state.join_strategy = 'join' if 'Join' in strategy else 'append'
    
    # Sheet selection
    all_sheets = list(st.session_state.normalizer.data_frames.keys())
    st.session_state.selected_sheets = st.multiselect(
        "Select sheets to combine:",
        all_sheets,
        default=all_sheets
    )
    
    if len(st.session_state.selected_sheets) < 2 and st.session_state.join_strategy != 'skip':
        st.warning("Select at least 2 sheets to combine.")
        return
    
    st.divider()
    
    # JOIN COLUMNS MODE
    if st.session_state.join_strategy == 'join':
        st.markdown("### Column Join Settings")
        
        # Detect common columns
        common_cols = st.session_state.normalizer.detect_common_columns(st.session_state.selected_sheets)
        join_candidates = st.session_state.normalizer.get_join_candidates(st.session_state.selected_sheets)
        
        # Manual or Auto selection mode
        use_manual = False
        if not common_cols:
            st.warning("**No exact column name matches found across selected sheets.**")
            use_manual = st.checkbox(
                "Enable Manual Column Selection",
                value=True,
                help="Manually specify which columns to use as join keys"
            )
        else:
            st.success(f"Found {len(common_cols)} common columns: {', '.join(common_cols)}")
            use_manual = st.checkbox(
                "Use Manual Column Selection Instead",
                value=False,
                help="Override auto-detection and manually specify join columns"
            )
        
        # Manual mode: Let user pick column from each sheet
        if use_manual:
            st.info("**Manual Join Mode**: Select the column from each sheet that represents the same data (e.g., 'Year' in one file and 'Period' in another)")
            
            join_mapping = {}
            cols_display = st.columns(2)
            
            for idx, sheet_name in enumerate(st.session_state.selected_sheets):
                df = st.session_state.normalizer.data_frames[sheet_name]
                with cols_display[idx % 2]:
                    selected_col = st.selectbox(
                        f"Join column in **{sheet_name}**:",
                        df.columns.tolist(),
                        key=f"manual_join_{sheet_name}"
                    )
                    join_mapping[sheet_name] = selected_col
            
            st.session_state.manual_join_mapping = join_mapping
            
            # Show what will be matched
            st.caption("**Columns that will be used for joining:**")
            for sheet, col in join_mapping.items():
                st.caption(f"- {sheet}: `{col}`")
            
            # Use first sheet's column as the canonical join key name
            first_sheet = st.session_state.selected_sheets[0]
            st.session_state.join_key = join_mapping[first_sheet]
            
        else:
            # Auto mode: Use common columns
            if not common_cols:
                st.error("No common columns found. Please enable manual selection or choose different sheets.")
                return
            
            # Let user select join key from common columns
            st.session_state.join_key = st.selectbox(
                "Select join key (column to match on):",
                join_candidates if join_candidates else common_cols,
                help="The column used to align rows across sheets"
            )
            st.session_state.manual_join_mapping = None
        
        # Show preview (only in auto mode)
        if st.session_state.join_key and not use_manual:
            with st.expander("Preview Join Result"):
                preview = st.session_state.normalizer.preview_join(
                    st.session_state.selected_sheets,
                    st.session_state.join_key,
                    max_rows=5
                )
                
                if 'error' in preview:
                    st.error(preview['error'])
                else:
                    st.caption(f"Preview: {preview['total_rows']} rows × {preview['total_columns']} columns")
                    st.dataframe(preview['preview'], use_container_width=True)
    
    # APPEND ROWS MODE
    else:
        st.markdown("### Row Append Settings")
        
        common_cols = st.session_state.normalizer.detect_common_columns(st.session_state.selected_sheets)
        
        if common_cols:
            st.success(f"Found {len(common_cols)} common columns")
            
            align_cols = st.multiselect(
                "Select columns to align:",
                common_cols,
                default=common_cols,
                help="Only these columns will be included (unless you enable non-common columns below)"
            )
            
            include_non_common = st.checkbox(
                "Include non-common columns (will add NAs)",
                value=False,
                help="Include columns that don't exist in all sheets (missing values will be NA)"
            )
            
            # Show preview
            with st.expander("Preview Append Result"):
                preview = st.session_state.normalizer.preview_append(
                    st.session_state.selected_sheets,
                    align_columns=align_cols,
                    include_non_common=include_non_common,
                    max_rows=5
                )
                
                if 'error' in preview:
                    st.error(preview['error'])
                else:
                    st.caption(f"Preview: {preview['total_rows']} rows × {preview['total_columns']} columns")
                    if preview.get('has_nulls'):
                        st.warning("Result will contain some NA values")
                    st.dataframe(preview['preview'], use_container_width=True)
            
            # Store settings
            st.session_state.align_columns = align_cols
            st.session_state.include_non_common = include_non_common
        else:
            st.warning("No common columns found. The append will combine all columns with many NAs.")
    
    # Navigation
    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        if st.button("← Back to Upload", use_container_width=True):
            st.session_state.wizard_step = 1
            st.rerun()
    with col2:
        can_proceed = True
        if st.session_state.join_strategy == 'join' and not st.session_state.join_key:
            can_proceed = False
        
        if can_proceed:
            if st.button("Next: Output Mode →", type="primary", use_container_width=True):
                # Perform the actual join/append
                with st.spinner("Combining data..."):
                    try:
                        joiner = DataJoiner()
                        dataframes = {name: st.session_state.normalizer.data_frames[name].copy() 
                                    for name in st.session_state.selected_sheets}
                        
                        if st.session_state.join_strategy == 'join':
                            # Handle manual column mapping if enabled
                            if hasattr(st.session_state, 'manual_join_mapping') and st.session_state.manual_join_mapping:
                                # Rename columns in each dataframe to standardize join key
                                canonical_key = st.session_state.join_key  # Already set to first sheet's column
                                
                                for sheet_name, col_name in st.session_state.manual_join_mapping.items():
                                    if col_name != canonical_key and sheet_name in dataframes:
                                        # Rename the selected column to match canonical key
                                        dataframes[sheet_name] = dataframes[sheet_name].rename(columns={col_name: canonical_key})
                            
                            # Now all dataframes have the same join key column name
                            combined = joiner.join_on_column(
                                dataframes,
                                st.session_state.join_key,
                                mode='reference'  # Use reference mode to build source mapping
                            )
                        else:
                            combined = joiner.append_rows(
                                dataframes,
                                align_columns=st.session_state.get('align_columns'),
                                include_non_common=st.session_state.get('include_non_common', False),
                                mode='reference'  # Use reference mode to build source mapping
                            )
                        
                        st.session_state.combined_df = combined
                        st.session_state.data_joiner = joiner
                        st.session_state.wizard_step = 3
                        st.rerun()
                    
                    except Exception as e:
                        st.error(f"Failed to combine data: {str(e)}")
                        import traceback
                        st.error(traceback.format_exc())



def step_3_edit_mode():
    """Step 3: Optional SQL-like Editing"""
    st.subheader("Step 3: Edit Combined Data (Optional)")
    
    st.markdown("""
    Make additional edits to the combined data.
    """)
    
    if st.session_state.data_editor is None:
        st.session_state.data_editor = DataEditor(st.session_state.combined_df)
    
    # Edit operations panel
    operation = st.selectbox(
        "Choose operation:",
        ["None", "Delete Columns", "Delete Rows", "Replace Values", "Add Column", "Add Row", "Mark as N.A."]
    )
    
    current_df = st.session_state.data_editor.get_dataframe()
    
    if operation == "Delete Columns":
        cols_to_delete = st.multiselect("Select columns to delete:", current_df.columns.tolist())
        if st.button("Delete Selected Columns") and cols_to_delete:
            st.session_state.data_editor.delete_columns(cols_to_delete)
            st.success(f"Deleted {len(cols_to_delete)} columns")
            st.rerun()
    
    elif operation == "Delete Rows":
        row_indices = st.multiselect(
            "Select row indices to delete:",
            list(range(len(current_df))),
            format_func=lambda x: f"Row {x}"
        )
        if st.button("Delete Selected Rows") and row_indices:
            st.session_state.data_editor.delete_rows(row_indices)
            st.success(f"Deleted {len(row_indices)} rows")
            st.rerun()
    
    elif operation == "Replace Values":
        col1, col2 = st.columns(2)
        with col1:
            find_val = st.text_input("Find:")
        with col2:
            replace_val = st.text_input("Replace with:")
        
        target_cols = st.multiselect("In columns (empty = all):", current_df.columns.tolist())
        
        if st.button("Replace") and find_val:
            st.session_state.data_editor.replace_values(
                find_val, replace_val,
                columns=target_cols if target_cols else None
            )
            st.success("Replacement complete")
            st.rerun()
    
    elif operation == "Add Column":
        col1, col2 = st.columns(2)
        with col1:
            new_col_name = st.text_input("Column name:")
        with col2:
            constant_val = st.text_input("Constant value (or leave empty for NA):")
        
        if st.button("Add Column") and new_col_name:
            const = constant_val if constant_val else None
            st.session_state.data_editor.add_column(new_col_name, constant=const)
            st.success(f"Added column '{new_col_name}'")
            st.rerun()
    
    elif operation == "Add Row":
        st.info("Adding empty row at the end")
        if st.button("Add Row"):
            st.session_state.data_editor.add_row()
            st.success("Added  new row")
            st.rerun()
    
    elif operation == "Mark as N.A.":
        row_indices = st.multiselect(
            "Select rows:",
            list(range(len(current_df))),
            format_func=lambda x: f"Row {x}"
        )
        target_cols = st.multiselect("Select columns:", current_df.columns.tolist())
        
        if st.button("Mark as N.A.") and (row_indices or target_cols):
            st.session_state.data_editor.mark_as_na(row_indices if row_indices else None,
                                                    target_cols if target_cols else None)
            st.success("Marked cells as N.A.")
            st.rerun()
    
    # Show current state with Interactive Grid
    st.divider()
    st.subheader("Interactive Data Editor")
    st.caption("Double-click any cell to edit. Changes are saved automatically.")
    
    # Use standard Streamlit Data Editor
    edited_df = st.data_editor(
        current_df,
        use_container_width=True,
        num_rows="dynamic",
        key="data_editor_grid"
    )
    
    # Sync manual edits back to wrapper if changed
    if not current_df.equals(edited_df):
        # Track which cells were manually edited
        for row_idx in range(min(len(current_df), len(edited_df))):
            for col_name in current_df.columns:
                if col_name in edited_df.columns:
                    old_val = current_df.iloc[row_idx][col_name]
                    new_val = edited_df.iloc[row_idx][col_name]
                    
                    # Handle NA values safely to avoid "boolean value of NA is ambiguous" error
                    old_is_na = pd.isna(old_val)
                    new_is_na = pd.isna(new_val)
                    
                    if old_is_na and new_is_na:
                        continue  # Both are NA, no change
                    elif old_is_na or new_is_na:
                        # One is NA, one isn't - this is a change
                        st.session_state.data_editor.affected_cells.add((row_idx, col_name))
                    elif old_val != new_val:
                        # Neither is NA, safe to compare normally
                        st.session_state.data_editor.affected_cells.add((row_idx, col_name))
        
        # Update the dataframe
        st.session_state.data_editor.df = edited_df
        
        # Add to edit history
        st.session_state.data_editor.edit_history.append({
            'operation': 'manual_cell_edit',
            'description': 'Manual edits via interactive grid'
        })
        
        st.rerun()
        
    st.caption(f"{len(current_df)} rows × {len(current_df.columns)} columns")
    
    # Show edit history
    if st.session_state.data_editor.get_edit_history():
        with st.expander("Edit History"):
            for edit in st.session_state.data_editor.get_edit_history():
                st.caption(f"- {edit['operation']}")
    
    # Navigation
    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        if st.button("← Back to Join Strategy", use_container_width=True):
            st.session_state.wizard_step = 2
            st.rerun()
    with col2:
        if st.button("Next: Finalize →", type="primary", use_container_width=True):
            # Update combined_df with edited version
            st.session_state.combined_df = st.session_state.data_editor.get_dataframe()
            st.session_state.wizard_step = 4
            st.rerun()


def step_4_finalize():
    """Step 4: Finalize and Export"""
    st.subheader("Step 4: Finalize Your Data")
    
    st.success("Data preparation complete.")
    
    # Final preview
    st.subheader("Final Combined Data")
    st.dataframe(st.session_state.combined_df.head(20), use_container_width=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Rows", len(st.session_state.combined_df))
    with col2:
        st.metric("Total Columns", len(st.session_state.combined_df.columns))
    
    # Download option
    st.divider()
    
    # Back button
    if st.button("Back to Edit Data", use_container_width=True):
        st.session_state.wizard_step = 3
        st.rerun()
    
    st.subheader("Download Combined File")
    
    if st.button("Generate Excel File", type="primary"):
        with st.spinner("Creating Excel file..."):
            output_path = Path("/tmp/combined_data.xlsx")
            try:
                import tempfile
                from excel_formula_exporter import export_combined_excel_with_formulas
                
                output_path = tempfile.mktemp(suffix='.xlsx')
                
                # Check if any edits were made
                has_edits = (st.session_state.data_editor is not None and 
                           len(st.session_state.data_editor.get_edit_history()) > 0)
                
                affected_cells = set()
                if has_edits:
                    affected_cells = st.session_state.data_editor.get_affected_cells()
                    
                # Get source dataframes and convert dates properly
                source_dfs = {}
                for name, df in st.session_state.normalizer.data_frames.items():
                    if name in st.session_state.selected_sheets:
                        df_copy = df.copy()
                        
                        # Convert ALL date-like columns to avoid Excel serial numbers
                        for col in df_copy.columns:
                            # Check if it's already a datetime column
                            if pd.api.types.is_datetime64_any_dtype(df_copy[col]):
                                # Convert datetime to string to prevent Excel serial conversion
                                df_copy[col] = df_copy[col].dt.strftime('%Y-%m-%d')
                            
                            # Check if it's numeric and might be Excel serial dates
                            elif pd.api.types.is_numeric_dtype(df_copy[col]):
                                min_val = df_copy[col].min()
                                max_val = df_copy[col].max()
                                
                                # Excel date serial numbers (typically 25000-60000 for 1968-2064)
                                if min_val > 25000 and max_val < 60000:
                                    # Convert Excel serial dates to datetime then string
                                    df_copy[col] = pd.to_datetime(df_copy[col], unit='D', origin='1899-12-30')
                                    df_copy[col] = df_copy[col].dt.strftime('%Y-%m-%d')
                        
                        source_dfs[name] = df_copy
                
                # Check if this is multi-file mode
                is_multifile = (hasattr(st.session_state, 'data_joiner') and 
                               st.session_state.data_joiner and 
                               len(source_dfs) > 1)
                
                # Decide export strategy - prioritize multi-file scenarios
                if is_multifile:
                    # Multi-file mode: Use formulas (hybrid if edits, pure if no edits)
                    join_type = 'column_join' if st.session_state.join_strategy == 'join' else 'row_append'
                    source_mapping = st.session_state.data_joiner.get_source_mapping()
                    
                    if has_edits:
                        # Hybrid: formulas for unedited cells, values for edited cells
                        from hybrid_excel_export import export_hybrid_excel
                        
                        excel_path = export_hybrid_excel(
                            combined_df=st.session_state.combined_df,
                            affected_cells=affected_cells,
                            source_mapping=source_mapping,
                            source_dataframes=source_dfs,
                            output_path=output_path,
                            join_type=join_type
                        )
                    else:
                        # Pure formulas: all cells reference source sheets
                        excel_path = export_combined_excel_with_formulas(
                            joined_df=st.session_state.combined_df,
                            source_mapping=source_mapping,
                            source_dataframes=source_dfs,
                            output_path=output_path,
                            join_type=join_type
                        )
                else:
                    # Single file mode: Just export the dataframe
                    st.session_state.combined_df.to_excel(output_path, index=False)
                    excel_path = output_path
                
                # Read back for download
                with open(excel_path, 'rb') as f:
                    st.download_button(
                        "Download Excel",
                        data=f,
                        file_name="combined_data.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
            except Exception as e:
                st.error(f"Failed to generate Excel file: {e}")
                import traceback
                st.error(traceback.format_exc())
            
            st.success("Excel file ready.")
    
    # Continue to other features
    st.divider()
    st.subheader("What's Next?")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("Use for Forecasting", use_container_width=True):
            # Set master data for forecasting
            st.session_state.master_df = st.session_state.combined_df
            
            # CRITICAL FIX: Also update the normalizer's internal ref so get_available_metrics() works
            st.session_state.normalizer.master_df = st.session_state.combined_df
            
            st.session_state.forecast_engine.set_base_data(
                st.session_state.combined_df.set_index('period') if 'period' in st.session_state.combined_df.columns 
                else st.session_state.combined_df
            )
            st.success("Data loaded into forecasting engine.")
            st.info("Navigate to 'Assumptions' tab to begin")
    
    with col2:
        if st.button("Ask AI Assistant", use_container_width=True):
            st.session_state.master_df = st.session_state.combined_df
            st.success("Data loaded for AI analysis.")
            st.info("Navigate to 'Chat Assistant' tab")
    
    with col3:
        if st.button("Start Over", use_container_width=True):
            # Reset wizard
            st.session_state.wizard_step = 1
            st.session_state.combined_df = None
            st.session_state.data_editor = None
            st.rerun()
