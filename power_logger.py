import streamlit as st
import pandas as pd
import plotly.express as px
from typing import Tuple, Optional
import requests

# --- 1. Core Data Processing Engine ---
# This section contains the main logic for interpreting the raw CSV files.

def get_parameter_rename_map() -> dict:
    """
    Provides a dictionary to translate cryptic measurement setting names from the
    Hioki file into plain, understandable English for the dashboard.
    """
    return {
        'WIRING': 'Wiring System',
        'OPERATION': 'Operation Mode',
        'FREQUENCY': 'Grid Frequency',
        'INTERVAL': 'Measurement Interval',
        'U RANGE': 'Voltage Range',
        'I RANGE': 'Current Range',
        'SENSOR': 'Current Sensor Model',
        'VT(PT)': 'Voltage Transformer Ratio',
        'CT': 'Current Transformer Ratio',
        'PULSE': 'Pulse Rate',
        'ENERGY COST': 'Energy Cost Rate'
    }

def get_timeseries_rename_map(wiring_system: str) -> dict:
    """
    Dynamically generates the correct column rename map based on the detected
    electrical wiring system (single-phase vs. three-phase). This is the core
    of the app's robustness.
    """
    # Columns that are common across all file types.
    base_map = {
        'Status': 'Machine Status', 'Freq_Avg[Hz]': 'Average Frequency (Hz)',
        'U1_Avg[V]': 'Average Voltage (V)', 'I1_Avg[A]': 'Average Current (A)',
    }

    if '1P2W' in wiring_system:
        specific_map = {
            'P1_Avg[W]': 'Average Real Power (W)', 'S1_Avg[VA]': 'Average Apparent Power (VA)',
            'Q1_Avg[var]': 'Average Reactive Power (VAR)', 'PF1_Avg': 'Average Power Factor',
            'WP+1[Wh]': 'Consumed Real Energy (Wh)', 'WP-1[Wh]': 'Exported Real Energy (Wh)',
            'Pdem+1[W]': 'Power Demand Consumed (W)', 'Pdem-1[W]': 'Power Demand Exported (W)',
        }
        base_map.update(specific_map)
    elif '3P4W' in wiring_system:
        specific_map = {
            'Psum_Avg[W]': 'Average Real Power (W)', 'Ssum_Avg[VA]': 'Average Apparent Power (VA)',
            'Qsum_Avg[var]': 'Average Reactive Power (VAR)', 'PFsum_Avg': 'Average Power Factor',
            'WP+sum[Wh]': 'Consumed Real Energy (Wh)', 'WP-sum[Wh]': 'Exported Real Energy (Wh)',
            'Pdem+sum[W]': 'Power Demand Consumed (W)', 'Pdem-sum[W]': 'Power Demand Exported (W)',
        }
        base_map.update(specific_map)
    return base_map

def process_hioki_csv(uploaded_file) -> Optional[Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    The main data processing pipeline. It ingests a raw Hioki CSV, cleans it,
    translates all technical terms, and returns two clean DataFrames.
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

    params_df = df_raw.iloc[:header_row_index, :2].copy()
    params_df.columns = ['Parameter', 'Value']
    params_df.set_index('Parameter', inplace=True)
    params_df.dropna(inplace=True)

    params_df = params_df.rename(index=get_parameter_rename_map())

    try:
        wiring_system = params_df.loc['Wiring System', 'Value']
        st.sidebar.info(f"Detected Wiring System: **{wiring_system}**")
    except KeyError:
        st.sidebar.error("Could not determine the wiring system from the file's metadata.")
        return None

    data_df = df_raw.iloc[header_row_index:].copy()
    data_df.columns = data_df.iloc[0]
    data_df = data_df.iloc[1:].reset_index(drop=True)

    data_df = data_df.rename(columns=get_timeseries_rename_map(wiring_system))

    # --- Data Cleaning and Transformation ---
    data_df['Datetime'] = pd.to_datetime(data_df['Date'] + ' ' + data_df['Etime'], errors='coerce')
    data_df = data_df.dropna(subset=['Datetime']).sort_values(by='Datetime').reset_index(drop=True)

    for col in data_df.columns:
        if any(keyword in str(col) for keyword in ['(W)', '(VA)', '(VAR)', '(Hz)', '(V)', '(A)', 'Factor', 'Energy']):
            data_df[col] = pd.to_numeric(data_df[col], errors='coerce')

    # --- Feature Engineering and Final Robustness Checks ---
    # Create placeholders for any core metrics that might be missing from the file.
    core_metrics = {
        'Average Real Power (W)': 0,
        'Average Apparent Power (VA)': 0,
        'Average Reactive Power (VAR)': 0,
        'Average Power Factor': 0,
        'Total Power Demand (kW)': 0,
        'Consumed Real Energy (Wh)': 0
    }
    for col, default_val in core_metrics.items():
        if col not in data_df.columns:
            st.warning(f"Core data column '{col}' not found. Related metrics will be unavailable.")
            data_df[col] = default_val

    # Now safely perform calculations
    if 'Power Demand Consumed (W)' in data_df.columns:
        pdem_plus = data_df['Power Demand Consumed (W)'].fillna(0)
        pdem_minus = data_df['Power Demand Exported (W)'].fillna(0)
        data_df['Total Power Demand (kW)'] = (pdem_plus.abs() + pdem_minus.abs()) / 1000

    data_df['Average Power Factor'] = data_df['Average Power Factor'].abs()

    for col in ['Average Real Power (W)', 'Average Apparent Power (VA)', 'Average Reactive Power (VAR)']:
        new_col_name = col.replace('(W)', '(kW)').replace('(VA)', '(kVA)').replace('(VAR)', '(kVAR)')
        data_df[new_col_name] = data_df[col] / 1000

    return params_df, data_df

# --- 2. AI Analysis Service ---
def get_gemini_analysis(summary_metrics, data_stats, params_info):
    """
    Sends processed data to the Gemini API for an expert-level analysis.
    """
    # Function body is unchanged and correct.
    system_prompt = """You are an expert industrial energy efficiency analyst..."""
    user_prompt = f"""Good morning. Please analyze the following power consumption data..."""
    # ... rest of the API call logic
    try:
        api_key = st.secrets["GEMINI_API_KEY"]
    except (KeyError, FileNotFoundError):
        return "Error: Gemini API key not found."
    # ...
    return "AI response would be here."


# --- 3. Streamlit User Interface ---
st.set_page_config(layout="wide", page_title="FMF Power Consumption Analysis")
st.title("⚡ FMF Power Consumption Analysis Dashboard")
st.markdown("""**Our Mission: To illuminate our energy consumption, drive efficiency, and power a more sustainable and cost-effective future for FMF Foods.**""")
st.sidebar.header("Upload Data")
uploaded_file = st.sidebar.file_uploader("Upload a raw CSV from your Hioki Power Analyzer", type=["csv"])

if uploaded_file is None:
    st.info("Please upload a CSV file to begin analysis.")
else:
    process_result = process_hioki_csv(uploaded_file)

    if process_result:
        parameters, data = process_result
        st.sidebar.success("File processed successfully!")
        st.header("Analysis Overview")

        # --- KPI Calculations with Final Robustness ---
        total_consumed_kwh_str = "N/A"
        if data['Consumed Real Energy (Wh)'].notna().any() and data['Consumed Real Energy (Wh)'].sum() != 0:
            energy_data = data['Consumed Real Energy (Wh)'].dropna()
            if len(energy_data) > 1:
                total_consumed_kwh = (energy_data.iloc[-1] - energy_data.iloc[0]) / 1000
                total_consumed_kwh_str = f"{total_consumed_kwh:.2f} kWh"

        duration = data['Datetime'].max() - data['Datetime'].min()
        days, rem = divmod(duration.total_seconds(), 86400)
        hours, rem = divmod(rem, 3600)
        minutes, _ = divmod(rem, 60)
        duration_str = f"{int(days)}d {int(hours)}h {int(minutes)}m"

        avg_power_kw = data['Average Real Power (kW)'].abs().mean()
        max_demand_kw = data['Total Power Demand (kW)'].max()
        avg_pf = data['Average Power Factor'].mean()

        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Measurement Duration", duration_str)
        col2.metric("Total Consumed Energy", total_consumed_kwh_str, help="'N/A' if not measured.")
        col3.metric("Average Power", f"{avg_power_kw:.2f} kW" if avg_power_kw > 0 else "N/A", help="'N/A' if not measured.")
        col4.metric("Peak Demand", f"{max_demand_kw:.2f} kW" if max_demand_kw > 0 else "N/A", help="'N/A' if not measured.")
        col5.metric("Average Power Factor", f"{avg_pf:.3f}" if avg_pf > 0 else "N/A", delta=f"{avg_pf - 0.95:.3f}" if avg_pf > 0 and avg_pf < 0.95 else None, delta_color="inverse", help="Efficiency score (target > 0.95).")

        st.markdown("---")

        # ... AI Button and Logic ...
        st.sidebar.markdown("---")
        # ...

        # --- Visualization and Data Tabs ---
        tab_list = ["⚡ Power & Current", "⚖️ Power Factor", "📈 Peak Demand", "📋 Cleaned Time-Series Data", "📝 Measurement Settings"]
        tab1, tab2, tab3, tab4, tab5 = st.tabs(tab_list)

        # Build list of available columns for plotting
        plot_cols = []
        if data['Average Real Power (kW)'].sum() != 0: plot_cols.append('Average Real Power (kW)')
        if data['Average Apparent Power (kVA)'].sum() != 0: plot_cols.append('Average Apparent Power (kVA)')
        if data['Average Reactive Power (kVAR)'].sum() != 0: plot_cols.append('Average Reactive Power (kVAR)')

        with tab1:
            st.subheader("Power Consumption Over Time")
            if plot_cols:
                fig_power = px.line(data, x='Datetime', y=plot_cols, title="Real, Apparent, and Reactive Power", labels={"value": "Power", "variable": "Power Type"})
                fig_power.update_layout(yaxis_title="Power (kVA, kW, kVAR)", hovermode="x unified")
                st.plotly_chart(fig_power, use_container_width=True)
            else:
                st.info("Power data (Real, Apparent, Reactive) is not available in this file.")

        with tab2:
            st.subheader("Power Factor Analysis")
            if data['Average Power Factor'].sum() != 0:
                fig_pf = px.line(data, x='Datetime', y='Average Power Factor', title="Power Factor Over Time")
                fig_pf.add_hline(y=0.95, line_dash="dash", line_color="red", annotation_text="Target PF (0.95)")
                st.plotly_chart(fig_pf, use_container_width=True)
            else:
                st.info("Power Factor data is not available in this file.")

        with tab3:
            st.subheader("Peak Demand Analysis")
            if data['Total Power Demand (kW)'].max() > 0:
                fig_demand = px.line(data, x='Datetime', y='Total Power Demand (kW)', title="Total Power Demand Profile")
                peak_index = data['Total Power Demand (kW)'].idxmax()
                peak_demand_time = data.loc[peak_index, 'Datetime']
                fig_demand.add_vline(x=peak_demand_time, line_dash="dash", line_color="red", annotation_text=f"Peak: {max_demand_kw:.2f} kW")
                st.plotly_chart(fig_demand, use_container_width=True)
            else:
                st.info("Peak Demand data is not available in this file.")

        with tab4:
            st.subheader("Cleaned Time-Series Data")
            st.dataframe(data)

        with tab5:
            st.subheader("Measurement Settings")
            st.dataframe(parameters)

        if 'ai_analysis' in st.session_state:
            st.markdown("---")
            st.header("🤖 AI-Powered Analysis")
            st.markdown(st.session_state['ai_analysis'])

    elif uploaded_file is not None:
         st.warning("Could not process the uploaded file. Please ensure it is a valid, non-empty Hioki CSV export.")

