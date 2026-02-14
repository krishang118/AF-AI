"""
Streamlit Frontend for Assumption-Driven Forecasting
Main application interface for analysts.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import json
from datetime import datetime
import uuid

from forecast_engine import (
    DataNormalizer, ForecastEngine, ExcelOutputGenerator,
    Assumption, Event, AssumptionType, EventType, AssumptionLayer
)
from llm_helper import LLMHelper, AssumptionBuilder, EventBuilder


# Page config
st.set_page_config(
    page_title="Assumption-Driven Forecasting AI",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    .stApp {
        background: linear-gradient(135deg, #000000 0%, #0a0a0a 100%);
        background-attachment: fixed;
    }
    
    .stApp::before {
        content: '';
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background-image: 
            radial-gradient(2px 2px at 20% 30%, rgba(255, 255, 255, 0.03), transparent),
            radial-gradient(2px 2px at 60% 70%, rgba(255, 255, 255, 0.03), transparent),
            radial-gradient(1px 1px at 50% 50%, rgba(255, 255, 255, 0.02), transparent);
        background-size: 200% 200%;
        background-position: 0% 0%;
        animation: drift 20s ease infinite;
        pointer-events: none;
        z-index: 0;
    }
    
    @keyframes drift {
        0%, 100% { background-position: 0% 0%; }
        50% { background-position: 100% 100%; }
    }
    
    h1, h2, h3, h4, h5, h6 {
        color: #e0e0e0 !important;
        font-weight: 600;
    }
    
    h1 {
        font-weight: 700;
        letter-spacing: -1px;
    }
    
    p, li, label, div {
        color: #b0b0b0 !important;
    }
    
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0a0a0a 0%, #000000 100%);
        border-right: 1px solid rgba(255, 255, 255, 0.08);
    }
    
    [data-testid="stSidebar"] * {
        color: #b0b0b0 !important;
    }
    
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: #e0e0e0 !important;
    }
    
    .stButton button {
        background: linear-gradient(135deg, #1a1a1a 0%, #0f0f0f 100%);
        color: white !important;
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 12px 28px;
        font-weight: 600;
        font-size: 14px;
        letter-spacing: 0.5px;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);
    }
    
    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 25px rgba(255, 255, 255, 0.1);
        border-color: rgba(255, 255, 255, 0.2);
    }

    .stButton button:focus, .stButton button:active {
        border-color: rgba(255, 255, 255, 0.3) !important;
        box-shadow: 0 0 0 2px rgba(255, 255, 255, 0.1) !important;
        background: linear-gradient(135deg, #1a1a1a 0%, #0f0f0f 100%) !important;
        color: white !important;
        outline: none !important;
    }
    
    /* Comprehensive Input Styling */
    .stTextInput > div[data-baseweb="input"],
    .stTextArea > div[data-baseweb="textarea"],
    .stNumberInput > div[data-baseweb="input"],
    .stDateInput > div[data-baseweb="input"] {
        border-color: rgba(255, 255, 255, 0.1) !important;
        border-radius: 12px !important;
        color: #FAFAFA !important;
    }
    
    .stTextInput > div[data-baseweb="input"]:focus-within,
    .stTextArea > div[data-baseweb="textarea"]:focus-within,
    .stNumberInput > div[data-baseweb="input"]:focus-within,
    .stDateInput > div[data-baseweb="input"]:focus-within {
        border-color: rgba(255, 255, 255, 0.5) !important;
        box-shadow: 0 0 10px rgba(255, 255, 255, 0.1) !important;
        outline: none !important;
    }
    
    /* GLOBAL TEXTAREA OVERRIDE - KILL RED AT ALL COSTS */
    textarea:focus, 
    textarea:active, 
    .stTextArea textarea:focus,
    div[data-baseweb="textarea"] {
        border-color: rgba(255, 255, 255, 0.5) !important;
        box-shadow: none !important;
        outline: none !important;
    }

    /* Specific override for standard Text Areas (Notes, etc) container */
    .stApp .stTextArea > div[data-baseweb="textarea"] {
        border-color: rgba(255, 255, 255, 0.2) !important;
        border-radius: 12px !important;
        background: rgba(20, 20, 20, 0.8) !important;
    }

    .stApp .stTextArea > div[data-baseweb="textarea"]:focus-within {
        border-color: rgba(255, 255, 255, 0.8) !important; /* Bright White */
        box-shadow: 0 0 10px rgba(255, 255, 255, 0.2) !important;
        outline: none !important;
    }
    
    .stApp .stTextArea textarea {
        background: transparent !important;
        color: #FAFAFA !important;
        caret-color: white !important;
    }

    /* Advanced Chat Input (High Specificity Override) */
    .stApp [data-testid="stChatInput"] {
        background: transparent !important;
        border: none !important;
        padding: 0 !important;
        position: sticky !important;
        bottom: 0 !important;
        z-index: 100 !important;
        display: flex !important;
        align-items: center !important;
        min-height: 70px !important;
    }
    
    /* The main container wrapper - WHITE THEME */
    .stApp [data-testid="stChatInput"] > div {
        background: rgba(10, 10, 10, 0.95) !important; 
        border: 2px solid rgba(255, 255, 255, 0.2) !important; /* White border */
        border-radius: 12px !important;
        padding: 0 !important;
        display: flex !important;
        align-items: center !important;
        min-height: 56px !important;
        width: 100% !important;
        box-shadow: none !important;
    }
    
    /* Glow on focus - WHITE */
    .stApp [data-testid="stChatInput"]:focus-within > div {
        border-color: rgba(255, 255, 255, 0.9) !important;
        box-shadow: 0 0 15px rgba(255, 255, 255, 0.2) !important;
        background: rgba(15, 15, 15, 1) !important;
    }    
    
    /* Kill intermediate containers explicitly */
    .stApp [data-testid="stChatInput"] [data-baseweb="base-input"],
    .stApp [data-testid="stChatInput"] [data-baseweb="input"] {
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
    }

    /* The text area itself - Kill all red (CHAT SPECIFIC) */
    .stApp [data-testid="stChatInput"] textarea,
    .stApp [data-testid="stChatInput"] textarea:invalid,
    .stApp [data-testid="stChatInput"] textarea:required {
        background: transparent !important;
        color: #FAFAFA !important;
        border: none !important;
        border-color: transparent !important;
        border-radius: 12px !important;
        padding: 18px 20px !important;
        font-size: 15px !important;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1) !important;
        outline: none !important;
        box-shadow: none !important;
        resize: none !important;
        flex: 1 !important;
        min-height: 56px !important;
        max-height: 56px !important;
        line-height: 1.5 !important;
        caret-color: white !important; /* White caret */
        display: flex !important;
        align-items: center !important;
    }

    /* Kill red on invalid focus */
    .stApp [data-testid="stChatInput"] textarea:focus:invalid,
    .stApp [data-testid="stChatInput"] textarea:focus:required {
        box-shadow: none !important;
        border-color: transparent !important;
        outline: none !important;
    }
    
    /* Explicitly kill focus ring on textarea */
    .stApp [data-testid="stChatInput"] textarea:focus {
        box-shadow: none !important;
        border: none !important;
        outline: none !important;
        border-color: transparent !important;
    }
    
    /* Specific override for Download Button in Export Section */
    [data-testid="stDownloadButton"] button {
        background: linear-gradient(135deg, #1a1a1a 0%, #0f0f0f 100%) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        color: white !important;
        transition: all 0.3s ease !important;
    }
    
    [data-testid="stDownloadButton"] button:hover {
        border-color: rgba(255, 255, 255, 0.3) !important;
        box-shadow: 0 4px 12px rgba(255, 255, 255, 0.1) !important;
        background: black !important;
        color: white !important;
    }
    
    [data-testid="stDownloadButton"] button:focus,
    [data-testid="stDownloadButton"] button:active {
        border-color: rgba(255, 255, 255, 0.5) !important;
        box-shadow: 0 0 0 2px rgba(255, 255, 255, 0.1) !important;
        outline: none !important;
        color: white !important;
    }
    
    /* NUCLEAR RED KILLER - TARGETING EVERYTHING */
    input, textarea, select, [data-baseweb="select"], [role="listbox"], [data-baseweb="input"] {
        caret-color: white !important;
        accent-color: white !important;
    }

    /* Force all borders to be transparent or white/grey */
    .stApp div[data-baseweb="input"],
    .stApp div[data-baseweb="select"],
    .stApp div[data-baseweb="base-input"] {
        border-color: rgba(255,255,255,0.2) !important;
        box-shadow: none !important;
    }

    /* Force Focus States to White */
    .stApp div[data-baseweb="input"]:focus-within,
    .stApp div[data-baseweb="select"]:focus-within,
    .stApp div[data-baseweb="base-input"]:focus-within {
        border-color: rgba(255,255,255,0.8) !important;
        box-shadow: 0 0 10px rgba(255,255,255,0.1) !important;
    }
    
    /* FIX MULTISELECT WHITE BACKGROUNDS - Streamlit Specific */
    .stMultiSelect [data-baseweb="select"] {
        background-color: rgb(38, 39, 48) !important;
    }
    
    .stMultiSelect [data-baseweb="select"] > div {
        background-color: rgb(38, 39, 48) !important;
    }
    
    .stMultiSelect [data-baseweb="tag"] {
        background-color: rgb(49, 51, 63) !important;
        color: white !important;
    }
    
    .stMultiSelect span[data-baseweb="tag"] {
        background-color: rgb(49, 51, 63) !important;
        color: white !important;
    }
    
    /* Dropdown menu */
    .stMultiSelect div[data-baseweb="popover"] {
        background-color: rgb(38, 39, 48) !important;
    }
    
    .stMultiSelect ul[role="listbox"] {
        background-color: rgb(38, 39, 48) !important;
    }
    
    .stMultiSelect ul[role="listbox"] li {
        background-color: rgb(38, 39, 48) !important;
        color: white !important;
    }
    
    .stMultiSelect ul[role="listbox"] li:hover {
        background-color: rgb(49, 51, 63) !important;
    }


    /* BROWSER VALIDATION KILLER (The final boss) */
    input:invalid,
    input:out-of-range,
    input:required {
        box-shadow: none !important;
        outline: none !important;
        border-color: rgba(255, 255, 255, 0.2) !important;
    }
    
    input:invalid:focus,
    input:out-of-range:focus,
    input:required:focus {
        border-color: rgba(255, 255, 255, 0.8) !important;
        box-shadow: 0 0 10px rgba(255,255,255,0.1) !important;
    }

    /* Specific Red override for error states */
    .stApp [data-testid="stNumberInput"] div[data-baseweb="input"],
    .stApp [data-testid="stDateInput"] div[data-baseweb="input"] {
        border-color: rgba(255,255,255,0.2) !important;
    }
    
    /* Input contents */
    .stNumberInput input, .stDateInput input {
        color: white !important;
        -webkit-text-fill-color: white !important;
    }
    
    [data-testid="stChatInput"] textarea::placeholder {
        color: #606060 !important;
    }
    
    [data-testid="stChatInput"] button,
    [data-testid="stChatInputSubmitButton"] {
        background: transparent !important;
        border: none !important;
        padding: 0 !important;
        margin: 0 12px 0 0 !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        border-radius: 8px !important;
        min-height: 40px !important;
        height: 40px !important;
        width: 40px !important;
        flex-shrink: 0 !important;
        align-self: center !important;
        color: white !important;
    }
    
    [data-testid="stChatInput"] > div > div:last-child {
        display: flex !important;
        align-items: center !important;
        height: 100% !important;
    }
    
    [data-testid="stChatInput"] button:hover {
        background: rgba(99, 102, 241, 0.1) !important;
        color: rgba(99, 102, 241, 1) !important;
    }
    
    [data-testid="stChatInput"] button svg {
        color: inherit !important;
    }

    input, textarea {
        caret-color: white !important;
    }
    
    /* Remove default focus rings */
    *:focus {
        outline: none !important;
        box-shadow: none !important;
    }
    
    .stSelectbox > div > div {
        background: rgba(20, 20, 20, 0.8) !important;
        border: 2px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 12px !important;
    }

    .stSelectbox > div > div:focus-within {
        border-color: rgba(255, 255, 255, 0.5) !important;
        box-shadow: 0 0 0 2px rgba(255, 255, 255, 0.1) !important;
    }
    
    .stRadio > div {
        background: rgba(20, 20, 20, 0.4) !important;
        border-radius: 12px !important;
        padding: 10px !important;
    }
    
    .stCheckbox {
        color: #b0b0b0 !important;
    }
    
    .stSlider > div > div {
        color: #e0e0e0 !important;
    }
    
    .stFileUploader > div {
        background: rgba(20, 20, 20, 0.6) !important;
        border: 2px dashed rgba(255, 255, 255, 0.2) !important;
        border-radius: 12px !important;
    }

    .stFileUploader > div:focus-within {
        border-color: rgba(255, 255, 255, 0.5) !important;
        box-shadow: 0 0 0 2px rgba(255, 255, 255, 0.1) !important;
    }
    
    .stSuccess {
        background: linear-gradient(135deg, rgba(40, 167, 69, 0.15) 0%, rgba(34, 139, 58, 0.15) 100%) !important;
        border: 1px solid rgba(40, 167, 69, 0.3) !important;
        border-radius: 8px !important;
    }
    
    .stWarning {
        background: linear-gradient(135deg, rgba(255, 193, 7, 0.15) 0%, rgba(255, 152, 0, 0.15) 100%) !important;
        border: 1px solid rgba(255, 193, 7, 0.3) !important;
        border-radius: 8px !important;
    }
    
    .stError {
        background: linear-gradient(135deg, rgba(220, 53, 69, 0.15) 0%, rgba(176, 42, 55, 0.15) 100%) !important;
        border: 1px solid rgba(220, 53, 69, 0.3) !important;
        border-radius: 8px !important;
    }
    
    .stInfo {
        background: linear-gradient(135deg, rgba(23, 162, 184, 0.15) 0%, rgba(19, 132, 150, 0.15) 100%) !important;
        border: 1px solid rgba(23, 162, 184, 0.3) !important;
        border-radius: 8px !important;
    }
    
    [data-testid="stMetricValue"] {
        color: #e0e0e0 !important;
        font-size: 28px !important;
        font-weight: 600 !important;
    }
    
    [data-testid="stMetricLabel"] {
        color: #b0b0b0 !important;
    }
    
    hr {
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.1), transparent);
        margin: 30px 0;
    }
    
    code {
        background: rgba(255, 255, 255, 0.05) !important;
        color: #cccccc !important;
        padding: 3px 8px;
        border-radius: 6px;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .stSpinner > div {
        border-top-color: #ffffff !important;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'normalizer' not in st.session_state:
    st.session_state.normalizer = DataNormalizer()
if 'forecast_engine' not in st.session_state:
    st.session_state.forecast_engine = ForecastEngine()
if 'llm_helper' not in st.session_state:
    st.session_state.llm_helper = LLMHelper()
if 'assumptions' not in st.session_state:
    st.session_state.assumptions = []
if 'events' not in st.session_state:
    st.session_state.events = []
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'current_metric' not in st.session_state:
    st.session_state.current_metric = None
if 'scenarios' not in st.session_state:
    st.session_state.scenarios = None
if 'change_log' not in st.session_state:
    st.session_state.change_log = []
if 'last_forecast_run' not in st.session_state:
    st.session_state.last_forecast_run = None
if 'ai_provider' not in st.session_state:
    st.session_state.ai_provider = 'groq'
if 'ai_model' not in st.session_state:
    st.session_state.ai_model = 'llama3:latest'
if 'groq_api_key' not in st.session_state:
    st.session_state.groq_api_key = None


def main():
    st.title("Assumption-Driven Forecasting AI")
    st.markdown("*Your AI Assistant, analyzing for you.*")
    
    # Sidebar navigation
    with st.sidebar:
        st.header("Navigation")
        page = st.radio(
            "Go to",
            ["Data Upload", "Assumptions", "Forecast", "Chat Assistant", "Export"],
            label_visibility="collapsed"
        )
        
        # NAVIGATION TRACKING: Reset configuration state when switching pages
        if 'current_page' not in st.session_state:
            st.session_state.current_page = page
        
        if st.session_state.current_page != page:
            st.session_state.current_page = page
            # Reset period confirmation so the "Gatekeeper" appears again
            st.session_state.period_column_confirmed = False
        
        
        st.divider()
        
        # Data Privacy & AI Settings
        st.divider()
        st.subheader("AI Settings")
        
        # 1. Provider Selection
        ai_provider = st.radio(
            "AI Provider",
            ["Ollama (Local)", "Groq (Cloud API)"],
            index=1 if st.session_state.ai_provider == 'groq' else 0,
            label_visibility="collapsed"
        )
        st.session_state.ai_provider = 'ollama' if "Ollama" in ai_provider else 'groq'
        
        # 2. Model Selection & Configuration
        if st.session_state.ai_provider == 'ollama':
            # Ollama Configuration
            st.session_state.ai_model = st.selectbox(
                "Model",
                ["llama3:latest", "deepseek-r1:7b"],
                index=0 if st.session_state.ai_model == 'llama3:latest' else 1,
                help="Llama 3: Fast, general purpose. DeepSeek R1: Strong reasoning."
            )
            
            # Connection Status
            if st.button("Check Local Connection", use_container_width=True):
                status = st.session_state.llm_helper.check_ollama_status()
                if status['status'] == 'connected':
                    st.success("Ollama Connected")
                else:
                    st.error("Ollama Not Running")
                    st.caption("Run: `ollama serve`")
                    
        else:
            # Cloud API Configuration (Groq or OpenAI)
            cloud_provider = st.radio("Cloud Provider", ["Groq", "OpenAI"], horizontal=True)
            
            if cloud_provider == "Groq":
                st.session_state.ai_model = "openai/gpt-oss-20b"
                st.info(f"Model: {st.session_state.ai_model}")
                
                # API Key Input
                api_key = st.text_input(
                    "Groq API Key",
                    type="password",
                    value=st.session_state.groq_api_key if st.session_state.groq_api_key else "",
                    help="Enter your Groq API key (not saved to disk)"
                )
                
                if api_key:
                    st.session_state.groq_api_key = api_key
                    st.success("Groq API Key Set")
                else:
                    st.warning("API Key Required")
            else:  # OpenAI
                st.session_state.ai_model = "gpt-4o-mini"
                st.info(f"Model: GPT-4o Mini")
                
                # API Key Input
                api_key = st.text_input(
                    "OpenAI API Key",
                    type="password",
                    value=st.session_state.get('openai_api_key', ''),
                    help="Enter your OpenAI API key from platform.openai.com"
                )
                
                if api_key:
                    st.session_state.openai_api_key = api_key
                    st.success("OpenAI API Key Set")
                else:
                    st.warning("API Key Required")
    
    
    # Route to selected page
    if page == "Data Upload":
        data_upload_page()
    elif page == "Assumptions":
        assumptions_page()
    elif page == "Forecast":
        forecast_page()
    elif page == "Chat Assistant":
        chat_page()
    elif page == "Export":
        export_page()


def data_upload_page():
    """Data upload page - now uses the multi-step wizard"""
    from wizard_ui import data_upload_wizard
    data_upload_wizard()



def assumptions_page():
    st.header("Assumptions & Events")
    
    if not hasattr(st.session_state, 'master_df') or st.session_state.master_df.empty:
        st.warning("Please upload data first")
        return
    
    # GATEKEEPER LOGIC:
    # If period is not confirmed for this session/visit, BLOCK usage and FORCE configuration.
    if not st.session_state.get('period_column_confirmed', False):
        st.info("**Data Configuration**")
        st.markdown("Please confirm which column represents your **Time Period** (Years, Months, Dates) before proceeding.")
        
        candidates = st.session_state.master_df.columns.tolist()
        
        # Try to guess logical default
        default_idx = 0
        search_terms = ['period', 'year', 'fy', 'date', 'time'] # prioritizing 'period' if we renamed it already
        
        for i, c in enumerate(candidates):
            if any(x in str(c).lower() for x in search_terms):
                default_idx = i
                break
                
        selected_period_col = st.selectbox(
            "Select Period Column", 
            candidates,
            index=default_idx,
            key="gatekeeper_period_selector"
        )
        
        if st.button("Confirm & Continue", type="primary"):
            df = st.session_state.master_df.copy()
            
            # Non-Destructive Rename Logic (Same as before)
            if 'period' in df.columns and selected_period_col != 'period':
                backup_name = 'period_backup'
                counter = 1
                while backup_name in df.columns:
                    backup_name = f"period_backup_{counter}"
                    counter += 1
                df = df.rename(columns={'period': backup_name})
            
            df = df.rename(columns={selected_period_col: 'period'})
            
            # Re-process period types
            try:
                if pd.api.types.is_numeric_dtype(df['period']):
                     if df['period'].min() > 1900 and df['period'].max() < 2100:
                         pass 
            except:
                 pass
            
            st.session_state.master_df = df.sort_values('period')
            
            # CRITICAL: Mark as confirmed to unlock the page
            st.session_state.period_column_confirmed = True
            
            # CRITICAL: Force update of metrics list since columns changed
            if hasattr(st.session_state, 'normalizer'):
                # Update normalizer's copy of the dataframe
                st.session_state.normalizer.master_df = st.session_state.master_df
            
            # CRITICAL: Update Forecast Engine reference immediately
            if hasattr(st.session_state, 'forecast_engine'):
                st.session_state.forecast_engine.set_base_data(st.session_state.master_df)

            # CRITICAL: Reset current metric if it's no longer valid (e.g. was renamed)
            if st.session_state.current_metric and st.session_state.current_metric not in st.session_state.master_df.columns:
                st.session_state.current_metric = None
                
            st.rerun()
            
        st.divider()
        st.caption("This setting is locked until you leave this page.")
        return 
        # STOP HERE. Do not show the rest of the page.

    # -------------------------------------------------------------------------
    # REST OF PAGE (Only shown if confirmed)
    # -------------------------------------------------------------------------

    metrics = st.session_state.normalizer.get_available_metrics()
    if not metrics:
        st.warning("No metrics available")
        return
    
    tab1, tab2 = st.tabs(["Growth Assumptions", "Events"])
    
    with tab1:
        st.subheader("Growth Assumptions")
        
        # Add new assumption
        with st.form("new_assumption"):
            col1, col2 = st.columns(2)
            
            with col1:
                metric = st.selectbox("Metric", metrics)
                assumption_name = st.text_input("Name", "Base Growth Rate")
                
                # v2: Range-based assumption option
                use_range = st.checkbox("Use Range (min-max for scenarios)", 
                                       help="Specify explicit range instead of confidence multipliers")
                
                if use_range:
                    st.caption("Define assumption range:")
                    min_val = st.slider("Minimum (%)", -50, 100, 3, 1) / 100
                    base_val = st.slider("Base Case (%)", -50, 100, 10, 1) / 100
                    max_val = st.slider("Maximum (%)", -50, 100, 17, 1) / 100
                    growth_rate = base_val
                else:
                    growth_rate = st.slider("Annual Growth Rate (%)", -50, 100, 5, 1) / 100
                    min_val = None
                    max_val = None
            
            with col2:
                confidence = st.selectbox("Confidence", ["low", "medium", "high"])
                
                # Layer selection - USE INTEGERS
                layer_options = {
                    "Structural Events (0)": 0,
                    "Growth Baseline (1)": 1,
                    "Market Dynamics (2)": 2,
                    "External Shocks (3)": 3
                }
                layer_name = st.selectbox("Processing Layer", list(layer_options.keys()),
                                        index=1,
                                        help="Determines processing order")
                layer = layer_options[layer_name]
                
                # Dependency selection
                available_events = ["None"] + [f"{e.name} ({e.id})" for e in st.session_state.events]
                depends_on_str = st.selectbox("Depends On Event", available_events, 
                                             help="Make this assumption conditional on an event")
                
                apply_if = "always"
                if depends_on_str != "None":
                    apply_if = st.selectbox("Apply When", 
                                           ["if_occurs", "if_not_occurs"],
                                           format_func=lambda x: "After event occurs" if x == "if_occurs" else "Before event occurs")
                
            notes = st.text_area("Notes", "")
            
            if st.form_submit_button("Add Assumption"):
                # Parse dependency
                depends_on_id = None
                if depends_on_str != "None":
                    depends_on_id = depends_on_str.split("(")[-1].rstrip(")")
                
                new_assumption = Assumption(
                    id=str(uuid.uuid4())[:8],
                    type=AssumptionType.GROWTH,
                    name=assumption_name,
                    metric=metric,
                    value=growth_rate,
                    confidence=confidence,
                    source='analyst',
                    notes=notes,
                    depends_on=depends_on_id,
                    apply_if=apply_if if depends_on_id else "always",
                    # v2 fields
                    use_range=use_range,
                    min_value=min_val,
                    max_value=max_val,
                    layer=layer
                )
                
                # Validate before adding
                validation = st.session_state.forecast_engine.validate_assumption(new_assumption)
                
                if validation['errors']:
                    for error in validation['errors']:
                        st.error(f"Error: {error}")
                    return  # Don't add if there are errors
                
                if validation['warnings']:
                    for warning in validation['warnings']:
                        st.warning(f"Warning: {warning}")
                
                st.session_state.assumptions.append(new_assumption)
                st.session_state.forecast_engine.add_assumption(new_assumption)
                
                # Log change
                st.session_state.change_log.append({
                    'timestamp': datetime.now(),
                    'type': 'assumption_added',
                    'description': f"Added {assumption_name} for {metric} ({growth_rate:.1%})",
                    'confidence': confidence
                })
                
                if use_range:
                    st.success(f"Added range assumption: {assumption_name} ({min_val:.1%} to {max_val:.1%})")
                else:
                    st.success(f"Added assumption: {assumption_name}")
        
        # Display existing assumptions
        if st.session_state.assumptions:
            st.divider()
            st.subheader("Current Assumptions")
            
            for i, assumption in enumerate(st.session_state.assumptions):
                # Source tags
                source_icons = {
                    'analyst': '[User]',  # Human input
                    'llm': '[AI]',       # AI suggested
                    'data': '[Data]'       # Data-derived
                }
                source_icon = source_icons.get(assumption.source, '[?]')
                
                # Confidence badge
                conf_colors = {
                    'high': '[High]',
                    'medium': '[Med]',
                    'low': '[Low]'
                }
                conf_badge = conf_colors.get(assumption.confidence, '[?]')
                
                with st.expander(f"{source_icon} {assumption.name} ({assumption.metric}) {conf_badge}", expanded=False):
                    col1, col2, col3 = st.columns([2, 2, 1])
                    
                    with col1:
                        st.write(f"**Type:** {assumption.type.value}")
                        st.write(f"**Value:** {assumption.value:.1%}")
                        
                        # Show dependency (NEW)
                        if assumption.depends_on:
                            # Find event name
                            event_name = "Unknown"
                            for event in st.session_state.events:
                                if event.id == assumption.depends_on:
                                    event_name = event.name
                                    break
                            
                            apply_text = "after" if assumption.apply_if == "if_occurs" else "before"
                            st.info(f"Conditional: Applies {apply_text} '{event_name}'")
                    
                    with col2:
                        st.write(f"**Confidence:** {assumption.confidence}")
                        
                        # Visual source tag
                        if assumption.source == 'analyst':
                            st.info("**Analyst Input** (Manual Entry)")
                        elif assumption.source == 'llm':
                            st.warning("**AI Suggested** (Review Recommended)")
                        elif assumption.source == 'data':
                            st.success("**Data-Derived** (From Historical Patterns)")
                    
                    with col3:
                        if st.button("Delete", key=f"del_assump_{i}"):
                            st.session_state.forecast_engine.remove_assumption(assumption.id)
                            st.session_state.assumptions.pop(i)
                            st.rerun()
                    
                    if assumption.notes:
                        st.caption(f"*{assumption.notes}*")
    
    with tab2:
        st.subheader("Discrete Events")
        
        # Add new event
        with st.form("new_event"):
            col1, col2 = st.columns(2)
            
            with col1:
                event_metric = st.selectbox("Metric", metrics, key="event_metric")
                event_name = st.text_input("Event Name", "Product Launch")
                event_type = st.selectbox("Type", [e.value for e in EventType])
            
            with col2:
                # Determine Period Type logic
                # Default to smart slider (generic mode) unless we explicitly detect datetime data
                is_date_model = False
                
                if hasattr(st.session_state, 'master_df') and hasattr(st.session_state.master_df, 'empty') and not st.session_state.master_df.empty:
                    # Check if 'period' column exists and is datetime
                    if 'period' in st.session_state.master_df.columns:
                        if pd.api.types.is_datetime64_any_dtype(st.session_state.master_df['period']):
                            is_date_model = True
                    else:
                        # Fallback: Check if index is datetime
                        if isinstance(st.session_state.master_df.index, pd.DatetimeIndex):
                            is_date_model = True
                
                if is_date_model:
                    # Use calendar date picker for actual datetime data
                    event_date_input = st.date_input("Date")
                    event_date_str = event_date_input.strftime('%Y-%m-%d')
                else:
                    # Generic / Smart Mode: Use a slider with smart labels
                    # Generate extended period labels (History + 60 future steps)
                    smart_labels = []
                    hist_len = 0
                    
                    if hasattr(st.session_state, 'master_df') and not st.session_state.master_df.empty:
                        # Get historical labels
                        if 'period' in st.session_state.master_df.columns:
                            hist_labels = st.session_state.master_df['period'].astype(str).tolist()
                        else:
                            hist_labels = [str(i) for i in range(1, len(st.session_state.master_df) + 1)]
                        
                        hist_len = len(hist_labels)
                        
                        # Extrapolate next 60 periods
                        future_labels = st.session_state.forecast_engine.extrapolate_periods(hist_labels, 60)
                        smart_labels = hist_labels + future_labels
                    else:
                        # Default fallback
                        smart_labels = [str(i) for i in range(1, 70)]
                    
                    # Slider selects an INDEX (1-based)
                    # We map this index to the smart label for display
                    max_p = len(smart_labels)
                    
                    def label_formatter(idx):
                        # adjust for 0-based list vs 1-based slider
                        if 1 <= idx <= len(smart_labels):
                            return smart_labels[idx-1]
                        return str(idx)
                    
                    # Use select_slider which supports format_func for custom labels
                    # We pass INDICES as options to ensure uniqueness (Jan, Jan...)
                    # and use format_func to show the label
                    indices = list(range(1, max_p + 1))
                    
                    event_date_idx = st.select_slider(
                        "Period (Time Step)", 
                        options=indices,
                        value=1,
                        format_func=label_formatter
                    )
                    
                    # CRITICAL FIX: Store the ACTUAL period label, not the index
                    # When data has years (2024, 2025...), we need to store "2025", not "2"
                    event_date_str = label_formatter(event_date_idx)
                    
                    # Show the selected label clearly
                    st.caption(f"Selected: **{event_date_str}**")

                impact_pct = st.slider("Impact (%)", -50, 100, 15, 5)
                decay_periods = st.slider("Decay Duration (Periods, 0=Forever)", 0, 60, 0)

            
            event_notes = st.text_area("Notes", "", key="event_notes")
            
            if st.form_submit_button("Add Event"):
                new_event = Event(
                    id=str(uuid.uuid4())[:8],
                    event_type=EventType(event_type),
                    name=event_name,
                    metric=event_metric,
                    date=event_date_str,
                    impact_multiplier=1 + (impact_pct / 100),
                    decay_periods=decay_periods,
                    notes=event_notes
                )
                
                # Validate event
                # Validate event
                last_date = None
                if hasattr(st.session_state, 'master_df') and not st.session_state.master_df.empty:
                    if is_date_model and 'period' in st.session_state.master_df.columns:
                        last_date = st.session_state.master_df['period'].max()
                    else:
                        # Fallback for generic/smart mode: Use ROW COUNT (Index)
                        # This works for both integers (1..N) and strings (Q1..Q4)
                        last_date = len(st.session_state.master_df)
                
                forecast_end = None
                if last_date is not None and pd.notna(last_date):
                    # Default validate up to 10 periods out (max slider)
                    if is_date_model:
                        forecast_end = last_date + pd.DateOffset(years=10)
                    else:
                        # Safe integer addition
                        forecast_end = int(last_date) + 10
                
                validation = st.session_state.forecast_engine.validate_event(
                    new_event, 
                    forecast_start=last_date,
                    forecast_end=forecast_end
                )
                
                if validation['errors']:
                    for error in validation['errors']:
                        st.error(f"Error: {error}")
                    return
                
                # Skip warning display - they're often misleading for data-agnostic mode
                # if validation['warnings']:
                #     for warning in validation['warnings']:
                #         st.warning(f"Warning: {warning}")
                
                st.session_state.events.append(new_event)
                st.session_state.forecast_engine.add_event(new_event)
                st.success(f"Added event: {event_name}")
        
        # Display existing events
        if st.session_state.events:
            st.divider()
            st.subheader("Planned Events")
            
            for i, event in enumerate(st.session_state.events):
                with st.expander(f"{event.name} - {event.date}", expanded=False):
                    col1, col2, col3 = st.columns([2, 2, 1])
                    
                    with col1:
                        st.write(f"**Type:** {event.event_type.value}")
                        st.write(f"**Metric:** {event.metric}")
                    
                    with col2:
                        st.write(f"**Impact:** {(event.impact_multiplier-1)*100:+.1f}%")
                        st.write(f"**Decay:** {event.decay_periods} periods")
                    
                    with col3:
                        if st.button("Delete", key=f"del_event_{i}"):
                            st.session_state.forecast_engine.remove_event(event.id)
                            st.session_state.events.pop(i)
                            st.rerun()
    



def forecast_page():
    st.header("Forecast & Scenarios")
    
    if not hasattr(st.session_state, 'master_df'):
        st.warning("Please upload data first")
        return
    
    metrics = st.session_state.normalizer.get_available_metrics()
    if not metrics:
        st.warning("No metrics available")
        return
        
    # CRITICAL: Ensure Engine always has up-to-date data (e.g. after manual period override)
    st.session_state.forecast_engine.set_base_data(st.session_state.master_df)
    
    # Metric selection
    selected_metric = st.selectbox("Select Metric to Forecast", metrics)
    st.session_state.current_metric = selected_metric
    
    # Data quality check
    st.subheader("Data Quality Check")
    quality = st.session_state.forecast_engine.validate_data_quality(selected_metric)
    
    if quality['errors']:
        st.error("**Data Quality Errors:**")
        for error in quality['errors']:
            st.error(f"Error: {error}")
        st.warning("**Cannot proceed with forecast due to data quality issues**")
        return
    
    if quality['warnings']:
        with st.expander("Data Quality Warnings", expanded=True):
            for warning in quality['warnings']:
                st.warning(f"Warning: {warning}")
    
    # Quality score badge
    quality_colors = {
        'good': 'Good',
        'fair': 'Fair',
        'poor': 'Poor'
    }
    
    st.info(f"**Data Quality:** {quality['quality_score'].upper()}")
    
    # What Changed? Timeline
    if st.session_state.change_log:
        with st.expander("What Changed?", expanded=False):
            st.markdown("**Recent changes to assumptions and events:**")
            # Show last 5 changes
            for change in reversed(st.session_state.change_log[-5:]):
                timestamp = change['timestamp'].strftime('%H:%M:%S')
                st.caption(f"**{timestamp}** - {change['description']}")
    
    # Forecast parameters
    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        forecast_periods = st.slider("Forecast Horizon (Periods)", 1, 60, 5)
    with col2:
        if st.button("Generate Forecast", type="primary"):
            # CRITICAL FIX: Ensure engine has the latest events from session state
            # This prevents bugs where engine is re-initialized but events list persists
            if 'events' in st.session_state and st.session_state.events:
                st.session_state.forecast_engine.events = list(st.session_state.events)
                
            with st.spinner("Generating scenarios..."):
                scenarios = st.session_state.forecast_engine.generate_scenarios(
                    selected_metric, forecast_periods
                )
                st.session_state.scenarios = scenarios
                st.session_state.forecast_periods_used = forecast_periods  # Store period count with scenarios
                st.session_state.last_forecast_run = datetime.now()
                
                # Generate contribution breakdown
                st.session_state.contribution_breakdown = st.session_state.forecast_engine.get_contribution_breakdown(
                    selected_metric, scenarios['base']
                )
                
                # Log change
                st.session_state.change_log.append({
                    'timestamp': datetime.now(),
                    'type': 'forecast_generated',
                    'description': f"Generated {forecast_periods}-year forecast for {selected_metric}",
                    'scenarios': ['base', 'upside', 'downside']
                })
                
                st.success("Forecast generated")
    
    # Display forecast
    if st.session_state.scenarios:
        scenarios = st.session_state.scenarios
        # Use the period count that was used to generate the forecast, not current slider value
        forecast_periods_actual = st.session_state.get('forecast_periods_used', len(scenarios['base']))
        
        # Scenario comparison chart
        st.divider()
        st.subheader("Scenario Comparison")
        
        fig = go.Figure()
        
        for scenario_name, df in scenarios.items():
            if selected_metric not in df.columns:
                st.info(f"Forecast not generated for '{selected_metric}' yet. Please click 'Generate Forecast'.")
                return
                
            fig.add_trace(go.Scatter(
                x=df['period'],
                y=df[selected_metric],
                mode='lines+markers',
                name=scenario_name.title(),
                line=dict(width=3 if scenario_name == 'base' else 2)
            ))
        
        fig.update_layout(
            title=f"{selected_metric} Forecast - All Scenarios",
            xaxis_title="Period",
            yaxis_title=selected_metric,
            hovermode='x unified',
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Contribution Breakdown (NEW)
        st.divider()
        st.subheader("Driver Contribution Analysis")
        st.markdown("*What's actually driving the forecast?*")
        
        if hasattr(st.session_state, 'contribution_breakdown') and not st.session_state.contribution_breakdown.empty:
            contrib_df = st.session_state.contribution_breakdown
            
            # Show contribution table
            st.dataframe(contrib_df, use_container_width=True)
            
            # Contribution breakdown chart
            if len(contrib_df) > 0:
                # Find contribution columns (exclude period, total_value, and _pct columns)
                contrib_cols = [col for col in contrib_df.columns 
                              if col not in ['period', 'total_value'] and not col.endswith('_pct')]
                
                if contrib_cols:
                    fig_contrib = go.Figure()
                    
                    # Helper map for friendly names
                    col_name_map = {'base_contribution': 'Baseline Growth'}
                    
                    # Map event IDs to Names
                    if hasattr(st.session_state, 'events'):
                        for event in st.session_state.events:
                            col_name_map[f"event_{event.id}"] = event.name

                    for col in contrib_cols:
                        if col in contrib_df.columns:
                            # Use friendly name if available, else clean up the column key
                            friendly_name = col_name_map.get(col, col.replace('_', ' ').title())
                            
                            fig_contrib.add_trace(go.Bar(
                                name=friendly_name,
                                x=contrib_df['period'],
                                y=contrib_df[col]
                            ))
                    
                    fig_contrib.update_layout(
                        title="Forecast Drivers by Period",
                        xaxis_title="Period",
                        yaxis_title="Contribution",
                        barmode='relative', # 'relative' allows mixed pos/neg bars (stacking)
                        height=400
                    )
                    
                    st.plotly_chart(fig_contrib, use_container_width=True)
        
        # Scenario data tables
        st.divider()
        st.subheader("Scenario Data")
        
        tab1, tab2, tab3 = st.tabs(["Base Case", "Upside", "Downside"])
        
        with tab1:
            st.dataframe(scenarios['base'], use_container_width=True)
        
        with tab2:
            st.dataframe(scenarios['upside'], use_container_width=True)
        
        with tab3:
            st.dataframe(scenarios['downside'], use_container_width=True)
        
        # Summary metrics
        st.divider()
        st.subheader("Summary Metrics")
        
        # Calculate TWO sets of metrics for clarity:
        # 1. Historical → Forecast End (includes event shocks from historical baseline)
        # 2. Forecast Period Only (pure forecast trajectory)
        
        col1, col2, col3 = st.columns(3)
        
        # Get values
        base_forecast_start = scenarios['base'][selected_metric].iloc[0]
        base_forecast_end = scenarios['base'][selected_metric].iloc[-1]
        
        upside_forecast_end = scenarios['upside'][selected_metric].iloc[-1]
        downside_forecast_end = scenarios['downside'][selected_metric].iloc[-1]
        
        # Try to get last historical value
        historical_end = None
        if hasattr(st.session_state, 'master_df') and not st.session_state.master_df.empty:
            if selected_metric in st.session_state.master_df.columns:
                historical_end = st.session_state.master_df[selected_metric].iloc[-1]
                # Ensure historical_end is not NaN
                if pd.isna(historical_end):
                    historical_end = None
        
        # Determine label
        rate_label = "CAGR"
        if not pd.api.types.is_datetime64_any_dtype(scenarios['base']['period']):
            rate_label = "Avg. Growth"
        
        # METRIC 1: Forecast Period Only (what users expect to see)
        if base_forecast_start != 0 and pd.notna(base_forecast_start) and pd.notna(base_forecast_end):
            forecast_base_cagr = ((base_forecast_end / base_forecast_start) ** (1 / forecast_periods_actual) - 1) * 100
            forecast_upside_cagr = ((upside_forecast_end / base_forecast_start) ** (1 / forecast_periods_actual) - 1) * 100
            forecast_downside_cagr = ((downside_forecast_end / base_forecast_start) ** (1 / forecast_periods_actual) - 1) * 100
        else:
            forecast_base_cagr = 0.0
            forecast_upside_cagr = 0.0
            forecast_downside_cagr = 0.0
        
        # METRIC 2: Historical → Forecast (if available)
        has_historical = historical_end is not None and historical_end != 0 and pd.notna(historical_end)
        if has_historical and pd.notna(base_forecast_end):
            # Total periods from last historical to forecast end
            total_periods = forecast_periods_actual
            historical_base_cagr = ((base_forecast_end / historical_end) ** (1 / total_periods) - 1) * 100
            historical_upside_cagr = ((upside_forecast_end / historical_end) ** (1 / total_periods) - 1) * 100
            historical_downside_cagr = ((downside_forecast_end / historical_end) ** (1 / total_periods) - 1) * 100
        
        # Display primary metrics (Forecast Period Only)
        st.markdown("#### Forecast Period Growth")
        st.caption(f"Growth rate within the {forecast_periods_actual}-period forecast horizon")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(f"Base Case {rate_label}", f"{forecast_base_cagr:.1f}%")
        
        with col2:
            st.metric(f"Upside {rate_label}", f"{forecast_upside_cagr:.1f}%", 
                     delta=f"{forecast_upside_cagr - forecast_base_cagr:+.1f}%")
        
        with col3:
            st.metric(f"Downside {rate_label}", f"{forecast_downside_cagr:.1f}%",
                     delta=f"{forecast_downside_cagr - forecast_base_cagr:+.1f}%")
        
        # Display secondary metrics if historical data available
        if has_historical:
            st.markdown("#### Historical → Forecast End")
            
            # Check if there's a significant difference (indicates major event impact)
            diff = abs(historical_base_cagr - forecast_base_cagr)
            if diff > 2.0:  # More than 2% difference
                st.caption(f"Growth from last historical point ({int(historical_end) if pd.notna(historical_end) else 'N/A':,}) to forecast end - **includes event shocks**")
            else:
                st.caption(f"Growth from last historical point ({int(historical_end) if pd.notna(historical_end) else 'N/A':,}) to forecast end")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(f"Base Case {rate_label}", f"{historical_base_cagr:.1f}%")
            
            with col2:
                st.metric(f"Upside {rate_label}", f"{historical_upside_cagr:.1f}%", 
                         delta=f"{historical_upside_cagr - historical_base_cagr:+.1f}%")
            
            with col3:
                st.metric(f"Downside {rate_label}", f"{historical_downside_cagr:.1f}%",
                         delta=f"{historical_downside_cagr - historical_base_cagr:+.1f}%")
            
            # Explanation if metrics diverge significantly
            if diff > 2.0:
                with st.expander("Why do these numbers differ?"):
                    st.markdown(f"""
                    **Forecast Period Growth ({forecast_base_cagr:.1f}%)** measures growth within your forecast horizon only.
                    
                    **Historical → Forecast End ({historical_base_cagr:.1f}%)** measures growth from your last actual data point through the forecast end, 
                    which includes the impact of any discrete events (product launches, price changes, etc.) applied early in the forecast period.
                    
                    If you have events with large negative impacts in early forecast periods, this can create a temporary dip that lowers 
                    the overall historical-to-forecast CAGR, even though the forecast trajectory itself shows healthy growth.
                    
                    **Example:** A -50% product launch in year 1 creates a sharp drop that affects the overall CAGR, but recovery happens over subsequent years.
                    """)
        
        # Scenario Driver Attribution (NEW)
        st.divider()
        st.subheader("What Drives Scenario Spread?")
        
        attribution = st.session_state.forecast_engine.get_scenario_attribution()
        
        if attribution and (attribution['upside'] or attribution['downside']):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Upside Drivers**")
                if attribution['upside']:
                    for driver, impact in sorted(attribution['upside'].items(), key=lambda x: abs(x[1]), reverse=True):
                        if impact != 0:
                            st.write(f"- {driver}: {impact:+.1f}%")
                else:
                    st.caption("No upside drivers (all high confidence)")
            
            with col2:
                st.markdown("**Downside Drivers**")
                if attribution['downside']:
                    for driver, impact in sorted(attribution['downside'].items(), key=lambda x: abs(x[1]), reverse=True):
                        if impact != 0:
                            st.write(f"- {driver}: {impact:.1f}%")
                else:
                    st.caption("No downside drivers (all high confidence)")
        else:
            st.info("Add medium or low confidence assumptions to see scenario drivers")


def chat_page():
    st.header("AI Chat Assistant")
    
    st.markdown("""
    Ask questions about your forecast, assumptions, and scenarios.
    
    **Example questions:**
    - Why does revenue spike in 2028?
    - What happens if the launch is delayed by one year?
    - Which assumption contributes most to growth?
    - Summarize the forecast drivers for a slide
    """)
    
    # Display Chat History (Chronological)
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant":
                # Show model used if available, otherwise default
                model_name = message.get("model_display", "")
                timestamp = message.get("timestamp", "")
                if model_name:
                    st.caption(f"Model: {model_name} | {timestamp}")
                else:
                    st.caption(timestamp)

    # Chat Logic
    if user_question := st.chat_input("Ask your question..."):
        # Add user message
        st.session_state.chat_history.append({
            "role": "user",
            "content": user_question,
            "timestamp": datetime.now().strftime("%H:%M:%S")
        })
        st.rerun()

    # Handle Assistant Response (if last message is user)
    if st.session_state.chat_history and st.session_state.chat_history[-1]["role"] == "user":
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    # Update LLM Helper with current settings
                    # Get the appropriate API key based on provider
                    if st.session_state.ai_provider == "openai":
                        api_key = st.session_state.get('openai_api_key', '')
                    else:  # groq or ollama
                        api_key = st.session_state.groq_api_key
                    
                    st.session_state.llm_helper = LLMHelper(
                        provider=st.session_state.ai_provider,
                        model=st.session_state.ai_model,
                        api_key=api_key
                    )
                    
                    # Helper to profile dataframe
                    def get_df_profile(df):
                        profile = {
                            "rows": len(df),
                            "columns": list(df.columns),
                            "dtypes": {k: str(v) for k, v in df.dtypes.items()},
                            "head": df.head(3).to_dict(orient='records')
                        }
                        # Add date range if applicable
                        for col in df.columns:
                            if pd.api.types.is_datetime64_any_dtype(df[col]):
                                profile["date_range"] = {
                                    "column": col,
                                    "min": str(df[col].min()),
                                    "max": str(df[col].max())
                                }
                                break # Just take the first date column found
                        return profile

                    # Prepare context with full data visibility
                    context = {
                        "date": datetime.now().strftime("%Y-%m-%d"),
                        "current_metric": st.session_state.current_metric,
                        "assumptions": [a.to_dict() for a in st.session_state.assumptions],
                        "events": [e.to_dict() for e in st.session_state.events],
                        "scenarios": None,
                        "combined_data_summary": None,
                        "uploaded_sheets_summary": {}
                    }
                    
                    # 1. Add Summary of Raw Uploaded Sheets (Direct uploads)
                    if hasattr(st.session_state, 'normalizer') and st.session_state.normalizer.data_frames:
                        for name, df in st.session_state.normalizer.data_frames.items():
                            context["uploaded_sheets_summary"][name] = get_df_profile(df)

                    # 2. Add Combined Data Context (if finalized)
                    if st.session_state.combined_df is not None:
                        df = st.session_state.combined_df
                        context["combined_data_summary"] = get_df_profile(df)
                        context["combined_data_summary"]["description"] = "Final merged dataset used for forecasting."
                    
                    # 3. Add Forecast Context
                    if st.session_state.scenarios and st.session_state.current_metric:
                        context["scenarios"] = {
                            name: df[st.session_state.current_metric].tolist()
                            for name, df in st.session_state.scenarios.items()
                        }
                    
                    # Get LLM response with conversation history for context
                    response = st.session_state.llm_helper.answer_what_if(
                        st.session_state.chat_history[-1]["content"], 
                        context,
                        conversation_history=st.session_state.chat_history[:-1]
                    )
                    
                    st.markdown(response)
                    
                    # Get display details
                    model_name = st.session_state.llm_helper.get_model_display_name()
                    timestamp = datetime.now().strftime("%H:%M:%S")
                    st.caption(f"Model: {model_name} | {timestamp}")
                    
                    # Add to history
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": response,
                        "timestamp": timestamp,
                        "model_display": model_name
                    })
                    
                except Exception as e:
                    st.error(f"Error generating response: {str(e)}")
                    if st.session_state.ai_provider == 'groq' and not st.session_state.groq_api_key:
                        st.info("Check your Groq API Key in the sidebar.")



def export_page():
    st.header("Export to Excel")
    
    if not st.session_state.assumptions and not st.session_state.scenarios:
        st.warning("No forecast data to export. Please generate a forecast first.")
        return
    
    st.markdown("""
    Export your complete forecast model to Excel:
    - **READ ME** sheet with model documentation
    - Assumptions sheet (color-coded by source)
    - Events sheet  
    - Forecast scenarios
    - Driver contribution breakdown
    """)
    
    if st.button("Generate Excel File", type="primary"):
        with st.spinner("Building Excel file..."):
            try:
                # Create Excel output
                excel_gen = ExcelOutputGenerator()
                
                # Collect metrics
                metrics = [st.session_state.current_metric] if st.session_state.current_metric else []
                
                # Add README sheet first
                if st.session_state.assumptions or st.session_state.events:
                    excel_gen.create_readme_sheet(
                        st.session_state.assumptions,
                        st.session_state.events,
                        metrics
                    )
                
                # Add assumptions
                if st.session_state.assumptions:
                    excel_gen.create_assumptions_sheet(st.session_state.assumptions)
                
                # Add events
                if st.session_state.events:
                    excel_gen.create_events_sheet(st.session_state.events)
                
                # Add forecast
                if st.session_state.scenarios and st.session_state.current_metric:
                    excel_gen.create_forecast_sheet(
                        st.session_state.current_metric,
                        st.session_state.scenarios
                    )
                    
                    # Add contribution breakdown
                    if hasattr(st.session_state, 'contribution_breakdown') and not st.session_state.contribution_breakdown.empty:
                        excel_gen.create_contribution_sheet(
                            st.session_state.current_metric,
                            st.session_state.contribution_breakdown
                        )
                
                # Save file
                output_path = Path("/tmp/forecast_output.xlsx")
                excel_gen.save(output_path)
                
                # Offer download
                with open(output_path, "rb") as f:
                    st.download_button(
                        label="Download Excel File",
                        data=f,
                        file_name=f"forecast_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
                
                st.success("Excel file ready for download")
                
                # Show what's included
                with st.expander("What's included in this export"):
                    sheets = ["READ ME (model documentation)"]
                    if st.session_state.assumptions:
                        sheets.append("Assumptions (all growth rates)")
                    if st.session_state.events:
                        sheets.append("Events (discrete shocks)")
                    if st.session_state.scenarios:
                        sheets.append(f"Forecast_{st.session_state.current_metric}")
                    if hasattr(st.session_state, 'contribution_breakdown'):
                        sheets.append(f"Contribution_{st.session_state.current_metric}")
                    
                    for sheet in sheets:
                        st.write(f"- {sheet}")
                
            except Exception as e:
                st.error(f"Error generating Excel: {str(e)}")
    
    # Summary
    st.divider()
    st.subheader("Export Summary")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Assumptions", len(st.session_state.assumptions))
    
    with col2:
        st.metric("Events", len(st.session_state.events))
    
    with col3:
        scenarios_count = 3 if st.session_state.scenarios else 0
        st.metric("Scenarios", scenarios_count)


if __name__ == "__main__":
    main()
