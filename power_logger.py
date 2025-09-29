import streamlit as st
import pandas as pd
import plotly.express as px
from typing import Tuple, Optional
import requests

# --- 1. Core Data Processing Engine ---
# This section contains the main logic for interpreting the raw CSV files.

<<<<<<< HEAD
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
=======
def get_rename_map(wiring_system: str) -> dict:
    """
    Returns the appropriate column rename map based on the wiring system.
    """
    # Generic columns that are common or will be the target names
    base_map = {
        'Status': 'Machine Status', 'Freq_Avg[Hz]': 'Average Frequency (Hz)',
        'U1_Avg[V]': 'Average Voltage (V)', 'Ufnd1_Avg[V]': 'Fundamental Voltage (V)',
        'Udeg1_Avg[deg]': 'Voltage Phase Angle (deg)', 'I1_Avg[A]': 'Average Current (A)',
        'Ifnd1_Avg[A]': 'Fundamental Current (A)', 'Ideg1_Avg[deg]': 'Current Phase Angle (deg)',
        'Ecost1': 'Estimated Cost', 'Pulse': 'Pulse Count'
    }

    # Columns specific to single-phase (1P2W) systems
    if '1P2W' in wiring_system:
        single_phase_map = {
            'P1_Avg[W]': 'Average Real Power (W)', 'S1_Avg[VA]': 'Average Apparent Power (VA)',
            'Q1_Avg[var]': 'Average Reactive Power (VAR)', 'PF1_Avg': 'Average Power Factor',
            'WP+1[Wh]': 'Consumed Real Energy (Wh)', 'WP-1[Wh]': 'Exported Real Energy (Wh)',
            'WQLAG1[varh]': 'Lagging Reactive Energy (VARh)', 'WQLEAD1[varh]': 'Leading Reactive Energy (VARh)',
            'WP+dem1[Wh]': 'Consumed Energy (Demand Period)', 'WP-dem1[Wh]': 'Exported Energy (Demand Period)',
            'WQLAGdem1[varh]': 'Lagging Reactive Energy (Demand Period)', 'WQLEADdem1[varh]': 'Leading Reactive Energy (Demand Period)',
            'Pdem+1[W]': 'Power Demand Consumed (W)', 'Pdem-1[W]': 'Power Demand Exported (W)',
            'QdemLAG1[var]': 'Lagging Reactive Power (Demand)', 'QdemLEAD1[var]': 'Leading Reactive Power (Demand)',
            'PFdem1': 'Power Factor (Demand)'
        }
        base_map.update(single_phase_map)

    # Columns specific to three-phase (3P4W) systems, mapping the 'sum' columns
    elif '3P4W' in wiring_system:
        three_phase_map = {
            'Psum_Avg[W]': 'Average Real Power (W)', 'Ssum_Avg[VA]': 'Average Apparent Power (VA)',
            'Qsum_Avg[var]': 'Average Reactive Power (VAR)', 'PFsum_Avg': 'Average Power Factor',
            'WP+sum[Wh]': 'Consumed Real Energy (Wh)', 'WP-sum[Wh]': 'Exported Real Energy (Wh)',
            'WQLAGsum[varh]': 'Lagging Reactive Energy (VARh)', 'WQLEADsum[varh]': 'Leading Reactive Energy (VARh)',
            'WP+dem_sum[Wh]': 'Consumed Energy (Demand Period)', 'WP-dem_sum[Wh]': 'Exported Energy (Demand Period)',
            'WQLAGdem_sum[varh]': 'Lagging Reactive Energy (Demand Period)', 'WQLEADdem_sum[varh]': 'Leading Reactive Energy (Demand Period)',
            'Pdem+sum[W]': 'Power Demand Consumed (W)', 'Pdem-sum[W]': 'Power Demand Exported (W)',
            'QdemLAGsum[var]': 'Lagging Reactive Power (Demand)', 'QdemLEADsum[var]': 'Leading Reactive Power (Demand)',
            'PFdem_sum': 'Power Factor (Demand)'
        }
        base_map.update(three_phase_map)
        
    return base_map
>>>>>>> 8108238eccba4bb7f0ba7f412c9e03ca9b32ef10

def get_timeseries_rename_map(wiring_system: str) -> dict:
    """
<<<<<<< HEAD
    Dynamically generates the correct column rename map based on the detected
    electrical wiring system (single-phase vs. three-phase). This is the core
    of the app's robustness.

    Args:
        wiring_system: The system identifier string (e.g., '1P2W', '3P4W').

    Returns:
        A dictionary for renaming the time-series data columns.
    """
    # Columns that are common across all file types.
    base_map = {
        'Status': 'Machine Status',
        'Freq_Avg[Hz]': 'Average Frequency (Hz)',
        'U1_Avg[V]': 'Average Voltage (V)',
        'I1_Avg[A]': 'Average Current (A)',
    }

    # If it's a single-phase system, use the '...1...' columns.
    if '1P2W' in wiring_system:
        specific_map = {
            'P1_Avg[W]': 'Average Real Power (W)',
            'S1_Avg[VA]': 'Average Apparent Power (VA)',
            'Q1_Avg[var]': 'Average Reactive Power (VAR)',
            'PF1_Avg': 'Average Power Factor',
            'WP+1[Wh]': 'Consumed Real Energy (Wh)',
            'WP-1[Wh]': 'Exported Real Energy (Wh)',
            'Pdem+1[W]': 'Power Demand Consumed (W)',
            'Pdem-1[W]': 'Power Demand Exported (W)',
        }
        base_map.update(specific_map)
    # If it's a three-phase system, use the summary ('...sum...') columns.
    elif '3P4W' in wiring_system:
        specific_map = {
            'Psum_Avg[W]': 'Average Real Power (W)',
            'Ssum_Avg[VA]': 'Average Apparent Power (VA)',
            'Qsum_Avg[var]': 'Average Reactive Power (VAR)',
            'PFsum_Avg': 'Average Power Factor',
            'WP+sum[Wh]': 'Consumed Real Energy (Wh)',
            'WP-sum[Wh]': 'Exported Real Energy (Wh)',
            'Pdem+sum[W]': 'Power Demand Consumed (W)',
            'Pdem-sum[W]': 'Power Demand Exported (W)',
        }
        base_map.update(specific_map)
    return base_map

def process_hioki_csv(uploaded_file) -> Optional[Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    The main data processing pipeline. It ingests a raw Hioki CSV, cleans it,
    translates all technical terms, and returns two clean DataFrames.

    Args:
        uploaded_file: The file object from Streamlit's file uploader.

    Returns:
        A tuple containing (parameters_df, data_df) or None if processing fails.
=======
    Loads, cleans, and processes a Hioki power analyzer CSV file,
    adapting to both single-phase and three-phase wiring systems.
>>>>>>> 8108238eccba4bb7f0ba7f412c9e03ca9b32ef10
    """
    try:
        df_raw = pd.read_csv(uploaded_file, header=None, on_bad_lines='skip', encoding='utf-8')
    except Exception as e:
        st.error(f"Error reading CSV file: {e}")
        return None

    # Find where the actual data table starts by looking for the 'Date' keyword.
    search_column = df_raw.iloc[:, 0].astype(str)
    header_indices = search_column[search_column.eq('Date')].index
    if header_indices.empty:
        st.error("Error: Header keyword 'Date' not found. This may not be a valid Hioki file.")
        return None
    header_row_index = header_indices[0]

    # Split the file into two parts: the settings (above 'Date') and the data (from 'Date' down).
    params_df = df_raw.iloc[:header_row_index, :2].copy()
    params_df.columns = ['Parameter', 'Value']
    params_df.set_index('Parameter', inplace=True)
    params_df.dropna(inplace=True)

<<<<<<< HEAD
    # Translate the parameter names into plain English.
    params_df = params_df.rename(index=get_parameter_rename_map())

    # Detect the wiring system from the newly translated parameters.
    try:
        wiring_system = params_df.loc['Wiring System', 'Value']
        st.sidebar.info(f"Detected Wiring System: **{wiring_system}**")
    except KeyError:
        st.sidebar.error("Could not determine the wiring system from the file's metadata.")
        return None

    # Process the main data table.
=======
    # --- KEY CHANGE: Detect wiring system ---
    try:
        wiring_system = params_df.loc['WIRING', 'Value']
        st.sidebar.info(f"Detected Wiring System: **{wiring_system}**")
    except KeyError:
        st.sidebar.error("Could not determine wiring system from file.")
        return None

>>>>>>> 8108238eccba4bb7f0ba7f412c9e03ca9b32ef10
    data_df = df_raw.iloc[header_row_index:].copy()
    data_df.columns = data_df.iloc[0]
    data_df = data_df.iloc[1:].reset_index(drop=True)

<<<<<<< HEAD
    # Rename the data columns based on the detected wiring system.
    data_df = data_df.rename(columns=get_timeseries_rename_map(wiring_system))
=======
    # Data Cleaning
    if len(data_df.columns) > 1:
        cols_to_check = data_df.columns.drop('Date', errors='ignore')
        data_df.replace(r'^\s*$', pd.NA, regex=True, inplace=True)
        data_df.dropna(subset=cols_to_check, how='all', inplace=True)
>>>>>>> 8108238eccba4bb7f0ba7f412c9e03ca9b32ef10

    # --- Data Cleaning and Transformation ---
    # Create a proper Datetime column for time-series plotting.
    data_df['Datetime'] = pd.to_datetime(data_df['Date'] + ' ' + data_df['Etime'], errors='coerce')
    data_df = data_df.dropna(subset=['Datetime']).sort_values(by='Datetime').reset_index(drop=True)
<<<<<<< HEAD

    # Convert all relevant columns to numbers for calculations.
    for col in data_df.columns:
        if '(W)' in col or '(VA)' in col or '(VAR)' in col or '(Hz)' in col or '(V)' in col or '(A)' in col or 'Factor' in col or 'Energy' in col:
            data_df[col] = pd.to_numeric(data_df[col], errors='coerce')

    # --- Feature Engineering and Robust Calculations ---
    # Handle potentially missing power demand data gracefully.
    if 'Power Demand Consumed (W)' in data_df.columns and 'Power Demand Exported (W)' in data_df.columns:
        pdem_plus = data_df['Power Demand Consumed (W)'].fillna(0)
        pdem_minus = data_df['Power Demand Exported (W)'].fillna(0)
        data_df['Total Power Demand (kW)'] = (pdem_plus.abs() + pdem_minus.abs()) / 1000
    else:
        st.warning("Power Demand data not found in this file. Peak Demand analysis will be unavailable.")
        data_df['Total Power Demand (kW)'] = 0  # Create a placeholder column.

    # Correct Power Factor for potential wiring reversal and convert power to kilo-units.
=======
    
    numeric_cols = data_df.columns.drop(['Date', 'Etime', 'Status', 'Datetime'], errors='ignore')
    for col in numeric_cols:
        data_df[col] = pd.to_numeric(data_df[col], errors='coerce')
        
    # --- DYNAMIC RENAMING ---
    rename_map = get_rename_map(wiring_system)
    data_df = data_df.rename(columns=rename_map)
    
    # Post-processing Calculations (these now work universally)
>>>>>>> 8108238eccba4bb7f0ba7f412c9e03ca9b32ef10
    if 'Average Power Factor' in data_df.columns:
        data_df['Average Power Factor'] = data_df['Average Power Factor'].abs()

    for col in ['Average Real Power (W)', 'Average Apparent Power (VA)', 'Average Reactive Power (VAR)']:
        if col in data_df.columns:
            new_col_name = col.replace('(W)', '(kW)').replace('(VA)', '(kVA)').replace('(VAR)', '(kVAR)')
            data_df[new_col_name] = data_df[col] / 1000

    return params_df, data_df

# --- 2. AI Analysis Service ---
def get_gemini_analysis(summary_metrics, data_stats, params_info):
    """
    Sends processed data to the Gemini API for an expert-level analysis.
    """
    # This prompt primes the AI to act as an energy efficiency expert for FMF.
    system_prompt = """You are an expert industrial energy efficiency analyst and process engineer for FMF Foods Ltd., a food manufacturing company in Fiji. Your task is to analyze the provided power consumption data, statistics, and trend graphs from an industrial machine. Provide a concise, actionable report in Markdown format. The report should have three sections: 1. Executive Summary, 2. Key Observations & Pattern Analysis, and 3. Actionable Recommendations for Cost Reduction. Be specific and base your analysis strictly on the data provided. Address the user as a process optimization engineer."""

    user_prompt = f"""
    Good morning. Please analyze the following power consumption data for an industrial machine at our facility in Suva. Today is Tuesday, 30th September 2025.

    **Key Performance Indicators (KPIs):**
    {summary_metrics}

    **Measurement Parameters:**
    {params_info}

    **Statistical Summary of Time-Series Data:**
    {data_stats}

    Based on this information, please generate a report with your insights and recommendations for our biscuit manufacturing line.
    """
    try:
        api_key = st.secrets["GEMINI_API_KEY"]
    except (KeyError, FileNotFoundError):
        return "Error: Gemini API key not found. Please add it to your Streamlit Secrets."

    api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}"
<<<<<<< HEAD
    payload = {"contents": [{"parts": [{"text": user_prompt}]}],"systemInstruction": {"parts": [{"text": system_prompt}]}}
=======
    
    payload = {
        "contents": [{"parts": [{"text": user_prompt}]}],
        "systemInstruction": {"parts": [{"text": system_prompt}]}
    }
>>>>>>> 8108238eccba4bb7f0ba7f412c9e03ca9b32ef10

    try:
        response = requests.post(api_url, json=payload, timeout=120)
        response.raise_for_status()
        result = response.json()
        candidate = result.get('candidates', [{}])[0]
        content = candidate.get('content', {}).get('parts', [{}])[0]
        return content.get('text', "Error: Could not extract analysis from the API response.")
    except requests.exceptions.RequestException as e:
        return f"An error occurred while contacting the AI Analysis service: {e}"
    except Exception as e:
        return f"An unexpected error occurred during AI analysis: {e}"


# --- 3. Streamlit User Interface ---
st.set_page_config(layout="wide", page_title="FMF Power Consumption Analysis")

st.title("⚡ FMF Power Consumption Analysis Dashboard")
st.markdown("""
**Our Mission: To illuminate our energy consumption, drive efficiency, and power a more sustainable and cost-effective future for FMF Foods.**
""")

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

        # --- KPI Calculations ---
        total_consumed_kwh = 0
        energy_data = data['Consumed Real Energy (Wh)'].dropna()
        if len(energy_data) > 1:
            total_consumed_kwh = (energy_data.iloc[-1] - energy_data.iloc[0]) / 1000

        duration = data['Datetime'].max() - data['Datetime'].min()
        days, rem = divmod(duration.total_seconds(), 86400)
        hours, rem = divmod(rem, 3600)
        minutes, _ = divmod(rem, 60)
        duration_str = f"{int(days)}d {int(hours)}h {int(minutes)}m"

        avg_power_kw = data['Average Real Power (kW)'].abs().mean()
        max_demand_kw = data['Total Power Demand (kW)'].max()
        avg_pf = data['Average Power Factor'].mean()

        # --- Display KPIs ---
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Measurement Duration", duration_str)
        col2.metric("Total Consumed Energy", f"{total_consumed_kwh:.2f} kWh")
        col3.metric("Average Power", f"{avg_power_kw:.2f} kW")
<<<<<<< HEAD
        col4.metric("Peak Demand", f"{max_demand_kw:.2f} kW" if max_demand_kw > 0 else "N/A", help="The highest power draw, which can affect utility bills. 'N/A' if not measured.")
        col5.metric("Average Power Factor", f"{avg_pf:.3f}", delta=f"{avg_pf - 0.95:.3f}" if avg_pf < 0.95 else None, delta_color="inverse", help="Efficiency score (target > 0.95). A negative delta is bad.")

=======
        col4.metric("Peak Demand", f"{max_demand_kw:.2f} kW")
        col5.metric("Average Power Factor", f"{avg_pf:.3f}", delta=f"{avg_pf - 0.95:.3f}" if avg_pf < 0.95 else None, delta_color="inverse", help="Target is > 0.95. A negative delta is bad.")
        
>>>>>>> 8108238eccba4bb7f0ba7f412c9e03ca9b32ef10
        st.markdown("---")

        # --- AI Analysis Section ---
        st.sidebar.markdown("---")
        if st.sidebar.button("🤖 Get AI-Powered Insights", help="Analyzes the current data to provide actionable recommendations."):
            with st.spinner("🧠 The AI is analyzing your data... This may take a moment."):
                summary_metrics_text = f"- Measurement Duration: {duration_str}\n- Total Consumed Energy: {total_consumed_kwh:.2f} kWh\n- Average Power: {avg_power_kw:.2f} kW\n- Peak Demand: {max_demand_kw:.2f} kW\n- Average Power Factor: {avg_pf:.3f}"
                stats_cols = ['Average Real Power (kW)', 'Average Apparent Power (kVA)', 'Average Reactive Power (kVAR)', 'Average Power Factor', 'Average Current (A)', 'Total Power Demand (kW)']
                existing_stats_cols = [col for col in stats_cols if col in data.columns]
                data_stats_text = data[existing_stats_cols].describe().to_string()
                params_info_text = parameters.to_string()

                ai_response = get_gemini_analysis(summary_metrics_text, data_stats_text, params_info_text)
                st.session_state['ai_analysis'] = ai_response

        # --- Visualization and Data Tabs ---
        tab_list = ["⚡ Power & Current", "⚖️ Power Factor", "📈 Peak Demand", "📋 Cleaned Time-Series Data", "📝 Measurement Settings"]
        tab1, tab2, tab3, tab4, tab5 = st.tabs(tab_list)

        with tab1:
            st.subheader("Power Consumption Over Time")
            fig_power = px.line(data, x='Datetime', y=['Average Real Power (kW)', 'Average Apparent Power (kVA)', 'Average Reactive Power (kVAR)'],
                                  title="Real, Apparent, and Reactive Power", labels={"value": "Power", "variable": "Power Type"})
            fig_power.update_layout(yaxis_title="Power (kVA, kW, kVAR)", hovermode="x unified")
            st.plotly_chart(fig_power, use_container_width=True)

        with tab2:
            st.subheader("Power Factor Analysis")
            st.markdown("This chart shows the corrected Power Factor. A value of 1.0 is perfect efficiency, while a value consistently below 0.95 can lead to higher utility charges.")
            fig_pf = px.line(data, x='Datetime', y='Average Power Factor', title="Power Factor Over Time")
            fig_pf.add_hline(y=0.95, line_dash="dash", line_color="red", annotation_text="Target PF (0.95)")
            st.plotly_chart(fig_pf, use_container_width=True)

        with tab3:
            st.subheader("Peak Demand Analysis")
            if data['Total Power Demand (kW)'].max() > 0:
                st.markdown("This chart shows the total power demand. The highest peak on this graph can determine utility demand charges, making it a key target for cost reduction.")
                fig_demand = px.line(data, x='Datetime', y='Total Power Demand (kW)', title="Total Power Demand Profile")
                peak_index = data['Total Power Demand (kW)'].idxmax()
                peak_demand_time = data.loc[peak_index, 'Datetime']
                fig_demand.add_vline(x=peak_demand_time, line_dash="dash", line_color="red", annotation_text=f"Peak: {max_demand_kw:.2f} kW")
                st.plotly_chart(fig_demand, use_container_width=True)
            else:
                st.info("Peak demand data is not available in this file, so this chart cannot be displayed.")

        with tab4:
            st.subheader("Cleaned Time-Series Data")
            st.dataframe(data)

        with tab5:
            st.subheader("Measurement Settings")
            st.dataframe(parameters)

<<<<<<< HEAD
        # --- Display AI Analysis at the bottom ---
=======
        with tab6:
            st.subheader("Variable Explanations")
            st.markdown("""
| Hioki Variable Name | Plain English Name | Significance & Insight | How It Helps Reduce Consumption |
| :--- | :--- | :--- | :--- |
| **P1_Avg[W]** or **Psum_Avg[W]** | Average Real Power | The 'useful' power performing actual work (e.g., mixing dough). This is the component you want to use effectively. | **Quantify Waste:** Compare power draw during idle vs. active states to identify energy wasted by equipment not being shut down. |
| **S1_Avg[VA]** or **Ssum_Avg[VA]**| Average Apparent Power | The total power your system must be able to handle, including both useful (Real) and wasted (Reactive) power. | **System Capacity:** Reducing this (by improving Power Factor) can free up electrical capacity, potentially avoiding costly transformer upgrades. |
| **Q1_Avg[var]** or **Qsum_Avg[var]** | Average Reactive Power | The 'wasted' power required solely to create magnetic fields for motors to operate. It does no useful work. | **Pinpoint Inefficiency:** This is the primary target for Power Factor Correction. High reactive power indicates significant potential savings. |
| **PF1_Avg** or **PFsum_Avg**| Average Power Factor | A direct score of electrical efficiency (Real Power / Apparent Power). 1.0 is perfect; < 0.95 is inefficient. | **Justify Investment:** Use this KPI to justify installing capacitor banks for Power Factor Correction, which directly eliminates utility penalties. |
| **WP+1[Wh]** or **WP+sum[Wh]** | Consumed Real Energy | The cumulative total of energy used over time. This is the primary metric your electricity bill is based on. | **Track Savings:** This is the ultimate measure of success. Track this value before and after process changes to validate energy savings. |
| **Pdem+1[W]** or **Pdem+sum[W]**| Power Demand (Consumed) | The average power usage over a short interval (e.g., 15 mins). Your utility bill's 'Demand Charge' is based on the highest peak. | **Reduce Peak Charges:** Identify when peaks occur and implement strategies like staggering machine start-ups to lower this value, directly cutting costs. |
| **I1_Avg[A]** | Average Current | The flow of electricity. For a given task, higher current can indicate mechanical stress, friction, or overload. | **Predictive Maintenance:** Monitor current trends. A gradual increase at a constant speed often signals failing bearings or other issues that increase load. |
| **WQLAG1[varh]** or **WQLAGsum[varh]** | Lagging Reactive Energy | The cumulative total of 'wasted' reactive power from motors. | **Size the Solution:** Use this value to accurately size the capacitor banks needed for a Power Factor Correction project. |
| **U1_Avg[V]** | Average Voltage | The electrical potential supplied. Should be stable. | **Diagnose Supply Issues:** Significant voltage drops or instability can harm equipment and affect performance. This helps identify grid supply problems. |
| **Freq_Avg[Hz]** | Average Frequency | The speed of the AC supply from the grid. Should be very stable (e.g., 50Hz or 60Hz). | **Verify Grid Quality:** Confirms the stability of the power being supplied to your factory. |
            """)

        # --- Display AI Analysis Section (Moved to the bottom) ---
>>>>>>> 8108238eccba4bb7f0ba7f412c9e03ca9b32ef10
        if 'ai_analysis' in st.session_state:
            st.markdown("---")
            st.header("🤖 AI-Powered Analysis")
            st.markdown(st.session_state['ai_analysis'])

    elif uploaded_file is not None:
         st.warning("Could not process the uploaded file. Please ensure it is a valid, non-empty Hioki CSV export.")
