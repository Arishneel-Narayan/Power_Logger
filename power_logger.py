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

def process_hioki_csv(uploaded_file) -> Optional[Tuple[str, pd.DataFrame, pd.DataFrame]]:
    """
    Main data processing pipeline. It now returns the detected wiring system
    along with the two cleaned DataFrames.
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
    
    for col_name in data_df.columns:
        if '(W)' in col_name or '(VA)' in col_name or '(VAR)' in col_name:
            new_col_name = col_name.replace('(W)', '(kW)').replace('(VA)', '(kVA)').replace(' (VAR)', ' (kVAR)')
            data_df[new_col_name] = data_df[col_name] / 1000

    return wiring_system, params_df, data_df


# --- 2. Streamlit UI and Analysis Section ---
st.set_page_config(layout="wide", page_title="FMF Power Consumption Analysis")
st.title("⚡ FMF Power Consumption Analysis Dashboard")
st.markdown(f"**Suva, Fiji** | {pd.Timestamp.now(tz='Pacific/Fiji').strftime('%A, %d %B %Y')}")

st.sidebar.header("Upload Data")
uploaded_file = st.sidebar.file_uploader("Upload a raw CSV from your Hioki Power Analyzer", type=["csv"])

if uploaded_file is None:
    st.info("Please upload a CSV file to begin analysis.")
else:
    process_result = process_hioki_csv(uploaded_file)

    if process_result:
        wiring_system, parameters, data = process_result
        st.sidebar.success(f"File processed successfully!\n\n**Mode: {wiring_system} Analysis**")
        
        # --- ========================================================= ---
        # ---               SINGLE-PHASE (1P2W) ANALYSIS                ---
        # --- ========================================================= ---
        if wiring_system == '1P2W':
            st.header("Single-Phase Performance Analysis")
            
            # --- Robust KPI Calculations ---
            kwh_str, peak_kw_str, avg_kw_str, avg_pf_str = "N/A", "N/A", "N/A", "N/A"
            
            if 'Consumed Real Energy (Wh)' in data.columns and not data['Consumed Real Energy (Wh)'].dropna().empty:
                energy_vals = data['Consumed Real Energy (Wh)'].dropna()
                if len(energy_vals) > 1:
                    kwh = (energy_vals.iloc[-1] - energy_vals.iloc[0]) / 1000
                    kwh_str = f"{kwh:.2f} kWh"
            
            if 'Power Demand (kW)' in data.columns and not data['Power Demand (kW)'].dropna().empty:
                peak_kw = data['Power Demand (kW)'].max()
                peak_kw_str = f"{peak_kw:.2f} kW"

            if 'Real Power (kW)' in data.columns and not data['Real Power (kW)'].dropna().empty:
                avg_kw_str = f"{data['Real Power (kW)'].abs().mean():.2f} kW"

            if 'Power Factor' in data.columns and not data['Power Factor'].dropna().empty:
                avg_pf_str = f"{data['Power Factor'].abs().mean():.3f}"
            
            st.metric("Average Power Draw", avg_kw_str)
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Consumed Energy", kwh_str)
            col2.metric("Peak Demand", peak_kw_str)
            col3.metric("Average Power Factor", avg_pf_str)

            # --- Visualization Tabs ---
            tab1, tab2, tab3 = st.tabs(["⚡ Power & Energy", "📝 Measurement Settings", "📋 Raw Data"])
            with tab1:
                plot_cols = [col for col in ['Real Power (kW)', 'Apparent Power (kVA)', 'Reactive Power (kVAR)'] if col in data.columns]
                if plot_cols:
                    st.subheader("Power Consumption Over Time")
                    st.line_chart(data, x='Datetime', y=plot_cols)
                if 'Power Factor' in data.columns:
                    st.subheader("Power Factor Over Time")
                    st.line_chart(data, x='Datetime', y='Power Factor')
                if not plot_cols and 'Power Factor' not in data.columns:
                    st.info("No power or power factor data available to plot.")
            
            with tab2:
                st.subheader("Measurement Settings")
                st.dataframe(parameters)
            with tab3:
                st.subheader("Cleaned Time-Series Data")
                st.dataframe(data)

        # --- ========================================================= ---
        # ---              THREE-PHASE (3P4W) DIAGNOSTIC                ---
        # --- ========================================================= ---
        elif wiring_system == '3P4W':
            st.header("Three-Phase System Diagnostic")

            # --- Robust KPI Calculations ---
            avg_power_str, avg_pf_str, imbalance_str = "N/A", "N/A", "N/A"
            current_cols = ['L1 Current (A)', 'L2 Current (A)', 'L3 Current (A)']
            
            if 'Total Real Power (kW)' in data.columns:
                avg_power_str = f"{data['Total Real Power (kW)'].mean():.2f} kW"
            if 'Total Power Factor' in data.columns:
                avg_pf_str = f"{data['Total Power Factor'].abs().mean():.3f}"
            if all(c in data.columns for c in current_cols):
                avg_currents = data[current_cols].mean()
                if avg_currents.mean() > 0:
                    imbalance = (avg_currents.max() - avg_currents.min()) / avg_currents.mean() * 100
                    imbalance_str = f"{imbalance:.1f} %"

            col1, col2, col3 = st.columns(3)
            col1.metric("Average Total Power", avg_power_str)
            col2.metric("Average Total Power Factor", avg_pf_str)
            col3.metric("Max Current Imbalance", imbalance_str, help="Under 5% is good, over 10% may indicate a problem.")

            # --- Visualization Tabs ---
            tabs_to_show = {}
            if all(c in data.columns for c in current_cols): tabs_to_show["📊 Load Balance"] = current_cols
            if all(c in data.columns for c in ['L1 Voltage (V)', 'L2 Voltage (V)', 'L3 Voltage (V)']): tabs_to_show["🩺 Voltage Health"] = ['L1 Voltage (V)', 'L2 Voltage (V)', 'L3 Voltage (V)']
            if all(c in data.columns for c in ['L1 Power Factor', 'L2 Power Factor', 'L3 Power Factor']): tabs_to_show["⚖️ Power Factor"] = ['L1 Power Factor', 'L2 Power Factor', 'L3 Power Factor']
            
            if tabs_to_show:
                analysis_tabs = st.tabs(list(tabs_to_show.keys()) + ["📝 Settings", "📋 Raw Data"])
                tab_idx = 0
                if "📊 Load Balance" in tabs_to_show:
                    with analysis_tabs[tab_idx]:
                        st.subheader("Current Load Balance Across Phases")
                        st.line_chart(data, x='Datetime', y=tabs_to_show["📊 Load Balance"], color=["#FF0000", "#0000FF", "#00FF00"])
                    tab_idx +=1
                if "🩺 Voltage Health" in tabs_to_show:
                    with analysis_tabs[tab_idx]:
                        st.subheader("Voltage Stability Across Phases")
                        st.line_chart(data, x='Datetime', y=tabs_to_show["🩺 Voltage Health"], color=["#FF0000", "#0000FF", "#00FF00"])
                    tab_idx +=1
                if "⚖️ Power Factor" in tabs_to_show:
                     with analysis_tabs[tab_idx]:
                        st.subheader("Power Factor Per Phase")
                        st.line_chart(data, x='Datetime', y=tabs_to_show["⚖️ Power Factor"], color=["#FF0000", "#0000FF", "#00FF00"])
                     tab_idx +=1
                with analysis_tabs[tab_idx]:
                    st.subheader("Measurement Settings")
                    st.dataframe(parameters)
                with analysis_tabs[tab_idx+1]:
                    st.subheader("Cleaned Time-Series Data")
                    st.dataframe(data)
            else:
                st.warning("No per-phase data (Current, Voltage, or Power Factor) was found to generate diagnostic charts.")
                tab1, tab2 = st.tabs(["📝 Settings", "📋 Raw Data"])
                with tab1: st.dataframe(parameters)
                with tab2: st.dataframe(data)

    elif uploaded_file is not None:
         st.warning("Could not process the uploaded file. Please ensure it is a valid, non-empty Hioki CSV export.")

