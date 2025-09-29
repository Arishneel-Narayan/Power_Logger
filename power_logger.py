import streamlit as st
import pandas as pd
import plotly.express as px
from typing import Tuple, Optional
import requests

# --- 1. Core Data Processing Engine ---

def get_rename_map(wiring_system: str) -> dict:
    """
    Dynamically generates a comprehensive rename map for all relevant columns
    based on the detected electrical wiring system. This ensures all final
    column names are in plain English.
    """
    param_map = {
        'WIRING': 'Wiring System', 'OPERATION': 'Operation Mode', 'FREQUENCY': 'Grid Frequency',
        'INTERVAL': 'Measurement Interval', 'U RANGE': 'Voltage Range', 'I RANGE': 'Current Range',
        'SENSOR': 'Current Sensor Model', 'VT(PT)': 'Voltage Transformer Ratio', 'CT': 'Current Transformer Ratio'
    }
    
    ts_map = {
        'Status': 'Machine Status', 'Freq_Avg[Hz]': 'Grid Frequency (Hz)',
        'U1_Avg[V]': 'L1 Voltage (V)', 'I1_Avg[A]': 'L1 Current (A)',
    }

    if '1P2W' in wiring_system:
        ts_map.update({
            'P1_Avg[W]': 'Real Power (W)', 'S1_Avg[VA]': 'Apparent Power (VA)',
            'Q1_Avg[var]': 'Reactive Power (VAR)', 'PF1_Avg': 'Power Factor',
            'WP+1[Wh]': 'Consumed Real Energy (Wh)', 'Pdem+1[W]': 'Power Demand (W)',
        })
    elif '3P4W' in wiring_system:
        for i in range(1, 4):
            ts_map[f'U{i}_Avg[V]'] = f'L{i} Voltage (V)'
            ts_map[f'I{i}_Avg[A]'] = f'L{i} Current (A)'
            ts_map[f'P{i}_Avg[W]'] = f'L{i} Real Power (W)'
            ts_map[f'S{i}_Avg[VA]'] = f'L{i} Apparent Power (VA)'
            ts_map[f'Q{i}_Avg[var]'] = f'L{i} Reactive Power (VAR)'
            ts_map[f'PF{i}_Avg'] = f'L{i} Power Factor'
        ts_map.update({
            'Psum_Avg[W]': 'Total Real Power (W)', 'Ssum_Avg[VA]': 'Total Apparent Power (VA)',
            'Qsum_Avg[var]': 'Total Reactive Power (VAR)', 'PFsum_Avg': 'Total Power Factor',
            'WP+sum[Wh]': 'Total Consumed Real Energy (Wh)', 'Pdem+sum[W]': 'Total Power Demand (W)',
        })
    
    return param_map, ts_map

def process_hioki_csv(uploaded_file) -> Optional[Tuple[str, pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
    """
    Main data processing pipeline. It now returns the detected wiring system, cleaned dataframes,
    and a separate dataframe for removed inactive periods.
    """
    try:
        df_raw = pd.read_csv(uploaded_file, header=None, on_bad_lines='skip', encoding='utf-8')
    except Exception as e:
        st.error(f"Error reading CSV file: {e}")
        return None

    search_column = df_raw.iloc[:, 0].astype(str)
    header_indices = search_column[search_column.eq('Date')].index
    if header_indices.empty:
        st.error("Error: Header keyword 'Date' not found. This may not be a valid Hioki file.")
        return None
    header_row_index = header_indices[0]

    temp_params = df_raw.iloc[:header_row_index, :2]
    temp_params.columns = ['Parameter', 'Value']
    try:
        wiring_system = temp_params.set_index('Parameter').loc['WIRING', 'Value']
    except KeyError:
        st.sidebar.error("Could not determine the wiring system from the file's metadata.")
        return None
    
    param_rename_map, ts_rename_map = get_rename_map(wiring_system)

    params_df = df_raw.iloc[:header_row_index, :2].copy()
    params_df.columns = ['Parameter', 'Value']
    params_df.set_index('Parameter', inplace=True)
    params_df.dropna(inplace=True)
    params_df = params_df.rename(index=param_rename_map)

    data_df = df_raw.iloc[header_row_index:].copy()
    data_df.columns = data_df.iloc[0]
    data_df = data_df.iloc[1:].reset_index(drop=True)
    data_df = data_df.rename(columns=ts_rename_map)
    
    data_df['Datetime'] = pd.to_datetime(data_df['Date'] + ' ' + data_df['Etime'], errors='coerce')
    data_df = data_df.dropna(subset=['Datetime']).sort_values(by='Datetime').reset_index(drop=True)

    for col in data_df.columns:
        if any(keyword in str(col) for keyword in ['(W)', '(VA)', 'VAR', '(V)', '(A)', 'Factor', 'Energy', '(Hz)']):
            data_df[col] = pd.to_numeric(data_df[col], errors='coerce')
    
    # --- NEW: Inactivity Filter ---
    activity_col = 'L1 Current (A)' # Use current as a reliable indicator of activity
    removed_data = pd.DataFrame()
    if activity_col in data_df.columns:
        is_flat = data_df[activity_col].rolling(window=3, center=True).std().fillna(0) < 1e-4
        active_data = data_df[~is_flat].copy()
        removed_data = data_df[is_flat].copy()

        if not removed_data.empty:
            st.sidebar.warning(f"{len(removed_data)} inactive data points removed from main analysis.")
            data_df = active_data

    if wiring_system == '3P4W':
        power_cols = ['L1 Real Power (W)', 'L2 Real Power (W)', 'L3 Real Power (W)']
        if all(c in data_df.columns for c in power_cols) and 'Total Real Power (W)' not in data_df.columns:
            st.sidebar.info("Calculating Total Power from phase data.")
            data_df['Total Real Power (W)'] = data_df[power_cols].sum(axis=1)
            
            apparent_cols = ['L1 Apparent Power (VA)', 'L2 Apparent Power (VA)', 'L3 Apparent Power (VA)']
            if all(c in data_df.columns for c in apparent_cols):
                data_df['Total Apparent Power (VA)'] = data_df[apparent_cols].sum(axis=1)

            if 'Total Real Power (W)' in data_df.columns and 'Total Apparent Power (VA)' in data_df.columns:
                data_df['Total Power Factor'] = data_df.apply(
                    lambda row: row['Total Real Power (W)'] / row['Total Apparent Power (VA)'] if row['Total Apparent Power (VA)'] != 0 else 0,
                    axis=1
                )

    for col_name in data_df.columns:
        if '(W)' in col_name or '(VA)' in col_name or '(VAR)' in col_name:
            new_col_name = col_name.replace('(W)', '(kW)').replace('(VA)', '(kVA)').replace(' (VAR)', ' (kVAR)')
            data_df[new_col_name] = data_df[col_name] / 1000

    return wiring_system, params_df, data_df, removed_data

# --- 2. AI Analysis Service ---
def get_gemini_analysis(summary_metrics, data_stats, params_info):
    """
    Sends processed data to the Gemini API for an expert-level analysis.
    """
    system_prompt = """You are an expert industrial energy efficiency analyst and process engineer for FMF Foods Ltd., a food manufacturing company in Fiji. Your task is to analyze power consumption data from industrial machinery at our biscuit factory in Suva. Your analysis must be framed within the context of a manufacturing environment. Consider operational cycles, equipment health, and system reliability. Most importantly, link your findings directly to cost-saving opportunities, specifically addressing EFL's two-part tariff structure (Energy Charge + Demand Charge + VAT). Mention how improving power factor reduces kVA demand and how lowering peak demand (MD) directly reduces the monthly demand charge. Provide a concise, actionable report in Markdown format with three sections: 1. Executive Summary, 2. Key Observations & Pattern Analysis, and 3. Actionable Recommendations. Address the user as a fellow process optimization engineer."""

    user_prompt = f"""
    Good morning, Please analyze the following power consumption data for an industrial machine at our Suva facility, keeping in mind our electricity costs are based on EFL's Maximum Demand tariff.

    **Key Performance Indicators (KPIs):**
    {summary_metrics}
    **Measurement Parameters:**
    {params_info}
    **Statistical Summary of Time-Series Data:**
    {data_stats}
    Based on all this information, please generate a report with your insights and recommendations for process optimization.
    """
    try:
        api_key = st.secrets["GEMINI_API_KEY"]
    except (KeyError, FileNotFoundError):
        return "Error: Gemini API key not found. Please add it to your Streamlit Secrets."
    api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-05-20:generateContent?key={api_key}"
    payload = {"contents": [{"parts": [{"text": user_prompt}]}],"systemInstruction": {"parts": [{"text": system_prompt}]}}
    try:
        response = requests.post(api_url, json=payload, timeout=120)
        response.raise_for_status()
        result = response.json()
        if 'error' in result: return f"Error from Gemini API: {result['error']['message']}"
        candidate = result.get('candidates', [{}])[0]
        content = candidate.get('content', {}).get('parts', [{}])[0]
        return content.get('text', "Error: Could not extract analysis from the API response.")
    except requests.exceptions.RequestException as e:
        return f"An error occurred while contacting the AI Analysis service: {e}"
    except Exception as e:
        return f"An unexpected error occurred: {e}"

# --- 3. Streamlit UI and Analysis Section ---
st.set_page_config(layout="wide", page_title="FMF Power Consumption Analysis")
st.title("⚡ FMF Power Consumption Analysis Dashboard")
st.markdown(f"**Suva, Fiji** | {pd.Timestamp.now(tz='Pacific/Fiji').strftime('%A, %d %B %Y')}")

# --- Sidebar Inputs ---
st.sidebar.header("Upload Data")
uploaded_file = st.sidebar.file_uploader("Upload a raw CSV from your Hioki Power Analyzer", type=["csv"])
st.sidebar.markdown("---")
st.sidebar.subheader("EFL Tariff Structure")
energy_rate = st.sidebar.number_input("Energy Rate (FJD/kWh)", min_value=0.00, value=0.61, step=0.01, help="Base rate + fuel surcharge.")
demand_rate = st.sidebar.number_input("Demand Rate (FJD/kVA/month)", min_value=0.00, value=25.41, step=0.01, help="EFL's monthly charge applied to your peak Apparent Power (MD in kVA).")
vat_rate = st.sidebar.number_input("VAT Rate (%)", min_value=0.0, value=15.0, step=0.5, help="Value Added Tax rate.")

if uploaded_file is None:
    st.info("Please upload a CSV file to begin analysis.")
else:
    process_result = process_hioki_csv(uploaded_file)

    if process_result:
        wiring_system, parameters, data, removed_data = process_result
        st.sidebar.success(f"File processed successfully!\n\n**Mode: {wiring_system} Analysis**")
        
        kpi_summary = {}
        
        if wiring_system == '1P2W':
            st.header("Single-Phase Performance Analysis")
            
            total_kwh = 0
            if 'Consumed Real Energy (Wh)' in data.columns and not data['Consumed Real Energy (Wh)'].dropna().empty:
                energy_vals = data['Consumed Real Energy (Wh)'].dropna()
                if len(energy_vals) > 1: total_kwh = (energy_vals.iloc[-1] - energy_vals.iloc[0]) / 1000
            
            peak_kva = data['Apparent Power (kVA)'].max() if 'Apparent Power (kVA)' in data.columns and not data['Apparent Power (kVA)'].dropna().empty else 0
            avg_kw = data['Real Power (kW)'].abs().mean() if 'Real Power (kW)' in data.columns and not data['Real Power (kW)'].dropna().empty else 0
            avg_pf = data['Power Factor'].abs().mean() if 'Power Factor' in data.columns and not data['Power Factor'].dropna().empty else 0
            duration_hours = (data['Datetime'].max() - data['Datetime'].min()).total_seconds() / 3600 if not data.empty else 0
            
            energy_cost, demand_cost, subtotal_cost, vat_cost, total_cost = (0, 0, 0, 0, 0)
            if duration_hours > 0:
                energy_cost = total_kwh * energy_rate
                demand_cost = peak_kva * demand_rate
                subtotal_cost = energy_cost + demand_cost
                vat_cost = subtotal_cost * (vat_rate / 100)
                total_cost = subtotal_cost + vat_cost
            
            st.subheader("Primary Metrics")
            # ... UI logic ...

            st.subheader("EFL Cost Breakdown")
            # ... UI logic ...
            
            kpi_summary = { "Analysis Mode": "Single-Phase", "Grand Total Cost": f"FJD {total_cost:.2f}", "Energy Charge": f"FJD {energy_cost:.2f}", "Demand Charge": f"FJD {demand_cost:.2f}" }
            
            # --- Visualization Tabs ---
            tab_names = ["⚡ Power & Energy", "📝 Measurement Settings", "📋 Active Data"]
            if not removed_data.empty:
                tab_names.append("🚫 Removed Inactive Periods")
            
            tabs = st.tabs(tab_names)
            with tabs[0]:
                # ... plot logic ...
            with tabs[1]:
                st.subheader("Measurement Settings")
                st.dataframe(parameters)
            with tabs[2]:
                st.subheader("Active Time-Series Data")
                st.dataframe(data)
            if not removed_data.empty:
                with tabs[3]:
                    st.subheader("Removed Inactive Data Periods")
                    st.info("The following data points were removed from the main analysis because they showed no significant fluctuation, likely indicating the machine was off or the measurement was paused.")
                    st.dataframe(removed_data)

        elif wiring_system == '3P4W':
            st.header("Three-Phase System Diagnostic")
            
            avg_power_kw = data['Total Real Power (kW)'].mean() if 'Total Real Power (kW)' in data.columns else 0
            avg_pf = data['Total Power Factor'].abs().mean() if 'Total Power Factor' in data.columns else 0
            peak_kva_3p = data['Total Apparent Power (kVA)'].max() if 'Total Apparent Power (kVA)' in data.columns else 0
            imbalance = 0
            current_cols = ['L1 Current (A)', 'L2 Current (A)', 'L3 Current (A)']
            if all(c in data.columns for c in current_cols):
                avg_currents = data[current_cols].mean()
                if avg_currents.mean() > 0: imbalance = (avg_currents.max() - avg_currents.min()) / avg_currents.mean() * 100
            
            duration_hours_3p = (data['Datetime'].max() - data['Datetime'].min()).total_seconds() / 3600 if not data.empty else 0
            total_kwh_3p, energy_cost_3p, demand_cost_3p, subtotal_cost_3p, vat_cost_3p, total_cost_3p = (0, 0, 0, 0, 0, 0)
            if duration_hours_3p > 0:
                total_kwh_3p = avg_power_kw * duration_hours_3p if avg_power_kw > 0 else 0
                energy_cost_3p = total_kwh_3p * energy_rate
                demand_cost_3p = peak_kva_3p * demand_rate
                subtotal_cost_3p = energy_cost_3p + demand_cost_3p
                vat_cost_3p = subtotal_cost_3p * (vat_rate / 100)
                total_cost_3p = subtotal_cost_3p + vat_cost_3p

            st.subheader("Primary Metrics")
            # ... UI logic ...
            
            st.subheader("EFL Cost Breakdown")
            # ... UI logic ...

            kpi_summary = { "Analysis Mode": "Three-Phase", "Grand Total Cost": f"FJD {total_cost_3p:.2f}", "Energy Charge": f"FJD {energy_cost_3p:.2f}", "Demand Charge": f"FJD {demand_cost_3p:.2f}", "Max Current Imbalance": f"f{imbalance:.1f} %" }

            # --- Visualization Tabs ---
            tab_names_3p = []
            tabs_to_show = {}
            # ... UI logic ...
            if not removed_data.empty:
                # ... append tab logic ...
            
        st.sidebar.markdown("---")
        if st.sidebar.button("🤖 Get AI-Powered Analysis", help="Click to get an AI-powered analysis of this data."):
            with st.spinner("🧠 AI is analyzing the data... This may take a moment."):
                # ... AI logic ...

        if 'ai_analysis' in st.session_state:
            st.markdown("---")
            st.header("🤖 AI-Powered Analysis")
            st.markdown(st.session_state['ai_analysis'])

    elif uploaded_file is not None:
         st.warning("Could not process the uploaded file. Please ensure it is a valid, non-empty Hioki CSV export.")

