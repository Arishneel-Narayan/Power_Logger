import streamlit as st
import pandas as pd
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from typing import Tuple, Optional, Dict
import requests
import io

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
    
    base_names = {
        'U': 'Voltage', 'I': 'Current', 'P': 'Real Power', 'S': 'Apparent Power',
        'Q': 'Reactive Power', 'PF': 'Power Factor', 'Freq': 'Grid Frequency',
        'Ufnd': 'Fundamental Voltage', 'Ifnd': 'Fundamental Current',
        'Udeg': 'Voltage Angle', 'Ideg': 'Current Angle', 'WP+': 'Consumed Real Energy',
        'Pdem+': 'Power Demand', 'WQLAG': 'Lagging Reactive Energy', 'WQLEAD': 'Leading Reactive Energy'
    }
    
    suffixes = {'_Avg': 'Avg', '_max': 'Max', '_min': 'Min'}
    units = {
        'V': '(V)', 'A': '(A)', 'W': '(W)', 'VA': '(VA)', 'var': '(VAR)',
        'Hz': '(Hz)', 'deg': '(deg)', 'Wh': '(Wh)', 'varh': '(kVARh)'
    }

    ts_map = {'Status': 'Machine Status'} # Keep 'Status' for the filter step

    phases = []
    if '1P2W' in wiring_system:
        phases = [('1', '')]
    elif '3P4W' in wiring_system:
        phases = [('1', 'L1 '), ('2', 'L2 '), ('3', 'L3 '), ('sum', 'Total ')]
    # Handle Hioki's P_Avg[W] as Total Avg Real Power for 3P4W
    elif '3P3W' in wiring_system:
        phases = [('1', 'L1 '), ('3', 'L3 '), ('sum', 'Total ')]


    for tech_prefix, eng_prefix in base_names.items():
        for phase_suffix, phase_prefix in phases:
            for suffix_key, suffix_val in suffixes.items():
                for unit_key, unit_val in units.items():
                    # Format: U1_Avg[V] -> L1 Avg Voltage (V)
                    hioki_name = f"{tech_prefix}{phase_suffix}{suffix_key}[{unit_key}]"
                    eng_name = f"{phase_prefix}{suffix_val} {eng_prefix} {unit_val}"
                    ts_map[hioki_name] = eng_name
            # Handle special cases without suffixes (e.g., PF1_Avg)
            if 'PF' in tech_prefix:
                ts_map[f"{tech_prefix}{phase_suffix}_Avg"] = f"{phase_prefix}Power Factor"

    # Specific overrides and additions
    ts_map.update({
        'P_Avg[W]': 'Total Avg Real Power (W)',
        'S_Avg[VA]': 'Total Avg Apparent Power (VA)',
        'Q_Avg[var]': 'Total Avg Reactive Power (VAR)',
        'PF_Avg': 'Total Power Factor',
        'P_max[W]': 'Total Max Real Power (W)',
        'S_max[VA]': 'Total Max Apparent Power (VA)',
        'Pdem+1[W]': 'Power Demand (W)', 'Pdem+sum[W]': 'Total Power Demand (W)',
        'WP+1[Wh]': 'Consumed Real Energy (Wh)', 'WP+sum[Wh]': 'Total Consumed Real Energy (Wh)',
        'WQLAG1[varh]': 'Lagging Reactive Energy (kVARh)', 'WQLAGsum[varh]': 'Total Lagging Reactive Energy (kVARh)'
    })

    return param_map, ts_map

@st.cache_data
def load_hioki_data(uploaded_file):
    """
    Main data processing pipeline. Correctly parses dates, handles negative values,
    and returns the full, unfiltered dataset.
    This function is cached for performance.
    """
    try:
        content = uploaded_file.getvalue().decode('utf-8')
        lines = content.splitlines()
    except Exception as e:
        st.error(f"Error reading or decoding file: {e}")
        return None

    header_row_index = -1
    for i, line in enumerate(lines):
        if line.strip().startswith('Date,Etime,'):
            header_row_index = i
            break
            
    if header_row_index == -1:
        st.error("Error: Could not find the header row ('Date,Etime,...') in the file.")
        return None

    metadata_lines = lines[:header_row_index]
    data_lines = lines[header_row_index:]

    try:
        params_df_raw = pd.read_csv(io.StringIO("\n".join(metadata_lines)), header=None, on_bad_lines='skip', usecols=[0, 1])
        params_df_raw.columns = ['Parameter', 'Value']
        wiring_system = params_df_raw.set_index('Parameter').loc['WIRING', 'Value']
    except (KeyError, IndexError):
        st.sidebar.error("Could not determine wiring system from metadata.")
        return None
    except Exception as e:
        st.error(f"Error parsing file metadata: {e}")
        return None
        
    param_rename_map, ts_rename_map = get_rename_map(wiring_system)

    params_df_raw.set_index('Parameter', inplace=True)
    params_df_raw.dropna(inplace=True)
    params_df = params_df_raw.rename(index=param_rename_map)

    try:
        data_df = pd.read_csv(io.StringIO("\n".join(data_lines)))
    except Exception as e:
        st.error(f"Error parsing the main data table: {e}")
        return None
        
    data_df = data_df.rename(columns=ts_rename_map)
    
    # --- DATETIME PARSING FIX ---
    # Correctly parse the 'Date' column which contains the full timestamp.
    # Add dayfirst=True to robustly handle DD/MM/YYYY formats if they appear.
    data_df['Datetime'] = pd.to_datetime(data_df['Date'], errors='coerce', dayfirst=True)
    
    if data_df['Datetime'].isnull().all():
        st.error("Error: Could not parse any timestamps from the 'Date' column. The data cannot be processed.")
        return None
        
    data_df.dropna(subset=['Datetime'], inplace=True)
    # --- END FIX ---

    identifier_cols_to_check = ['Datetime', 'Date', 'Etime', 'Machine Status']
    existing_identifiers = [col for col in identifier_cols_to_check if col in data_df.columns]
    measurement_cols = data_df.columns.drop(existing_identifiers, errors='ignore')
    data_df.dropna(subset=measurement_cols, how='all', inplace=True)
    
    if data_df.empty:
        st.error("No valid data rows with measurements were found.")
        return None
    
    data_df = data_df.sort_values(by='Datetime').reset_index(drop=True)

    for col in data_df.columns:
        if any(keyword in str(col) for keyword in ['(W)', '(VA)', 'VAR', '(V)', '(A)', 'Factor', 'Energy', '(Hz)', '(kVARh)']):
            data_df[col] = pd.to_numeric(data_df[col], errors='coerce')

    for col in data_df.columns:
        if 'Power Factor' in col or 'Power' in col:
            data_df[col] = data_df[col].abs()
    
    # Calculate missing 'Total' columns if they weren't in the file
    if wiring_system == '3P4W':
        power_cols = ['L1 Avg Real Power (W)', 'L2 Avg Real Power (W)', 'L3 Avg Real Power (W)']
        if all(c in data_df.columns for c in power_cols) and 'Total Avg Real Power (W)' not in data_df.columns:
            data_df['Total Avg Real Power (W)'] = data_df[power_cols].sum(axis=1)
            
            apparent_cols = ['L1 Avg Apparent Power (VA)', 'L2 Avg Apparent Power (VA)', 'L3 Avg Apparent Power (VA)']
            if all(c in data_df.columns for c in apparent_cols) and 'Total Avg Apparent Power (VA)' not in data_df.columns:
                data_df['Total Avg Apparent Power (VA)'] = data_df[apparent_cols].sum(axis=1)
            
            if 'Total Avg Real Power (W)' in data_df.columns and 'Total Avg Apparent Power (VA)' in data_df.columns and 'Total Power Factor' not in data_df.columns:
                data_df['Total Power Factor'] = data_df.apply(
                    lambda row: row['Total Avg Real Power (W)'] / row['Total Avg Apparent Power (VA)'] if row['Total Avg Apparent Power (VA)'] > 0 else 0,
                    axis=1
                )

    # Convert all power/energy units to k-units for easier reading
    for col_name in list(data_df.columns): # Use list() to allow modification during iteration
        if '(W)' in col_name or '(VA)' in col_name or '(VAR)' in col_name:
            new_col_name = col_name.replace('(W)', '(kW)').replace('(VA)', '(kVA)').replace('(VAR)', '(kVAR)')
            if new_col_name not in data_df.columns: # Avoid overwriting
                data_df[new_col_name] = data_df[col_name] / 1000

    return wiring_system, params_df, data_df

# --- 2. AI Service ---

def generate_trend_summary(data: pd.DataFrame, wiring_system: str) -> str:
    """Creates a narrative summary of the key power fluctuations."""
    if data.empty:
        return "No data available to analyze trends."

    key_metric = ""
    if wiring_system == '3P4W' and 'Total Avg Real Power (kW)' in data.columns:
        key_metric = 'Total Avg Real Power (kW)'
    elif wiring_system == '1P2W' and 'Avg Real Power (kW)' in data.columns:
        key_metric = 'Avg Real Power (kW)'

    if not key_metric or data[key_metric].dropna().empty:
        return "Key power metric not available for trend analysis."

    metric_series = data[key_metric]
    
    # Use requested date format
    date_format = '%a %d %b %Y, %H:%M:%S'
    start_time = data['Datetime'].iloc[0].strftime(date_format)
    end_time = data['Datetime'].iloc[-1].strftime(date_format)

    initial_power = metric_series.iloc[0]
    final_power = metric_series.iloc[-1]
    
    peak_power = metric_series.max()
    peak_time = data.loc[metric_series.idxmax(), 'Datetime'].strftime(date_format)
    
    min_power = metric_series.min()
    min_time = data.loc[metric_series.idxmin(), 'Datetime'].strftime(date_format)

    summary = (
        f"The analyzed period from {start_time} to {end_time} shows a significant fluctuation in power consumption. "
        f"It began at {initial_power:.2f} kW. "
        f"The primary operational peak reached {peak_power:.2f} kW at {peak_time}, indicating a major load event. "
        f"Conversely, the load dropped to a minimum of {min_power:.2f} kW at {min_time}, likely representing an idle or low-power state. "
        f"The period concluded at a level of {final_power:.2f} kW."
    )
    return summary

def generate_statistical_summary(data: pd.DataFrame, wiring_system: str) -> str:
    """Generates a structured statistical summary of the most important columns."""
    summary_lines = []
    
    # Define columns of interest
    cols_of_interest = []
    if wiring_system == '3P4W':
        cols_of_interest = [
            'Total Avg Real Power (kW)', 'Total Avg Apparent Power (kVA)', 'Total Power Factor',
            'L1 Avg Current (A)', 'L2 Avg Current (A)', 'L3 Avg Current (A)',
            'L1 Avg Voltage (V)', 'L2 Avg Voltage (V)', 'L3 Avg Voltage (V)'
        ]
    elif wiring_system == '1P2W':
        cols_of_interest = [
            'Avg Real Power (kW)', 'Avg Apparent Power (kVA)', 'Power Factor',
            'Avg Current (A)', 'Avg Voltage (V)'
        ]

    for col in cols_of_interest:
        if col in data.columns and not data[col].empty:
            try:
                stats = data[col].describe()
                line = (
                    f"**{col}:** "
                    f"Mean={stats['mean']:.2f}, "
                    f"Min={stats['min']:.2f}, "
                    f"Max={stats['max']:.2f}, "
                    f"StdDev={stats['std']:.2f}"
                )
                summary_lines.append(f"- {line}")
            except Exception:
                # Skip if stats fail for any reason
                pass
                
    if not summary_lines:
        return "No detailed statistics could be generated for key metrics."
        
    return "\n".join(summary_lines)

def get_gemini_analysis(summary_metrics, detailed_stats, trend_summary, params_info, additional_context=""):
    """Contacts the Gemini API for an expert analysis."""
    system_prompt = """You are an expert industrial energy efficiency analyst and process engineer for FMF Foods Ltd., a food manufacturing company in Fiji. Your task is to analyze power consumption data from industrial machinery at our biscuit factory in Suva. Your analysis must be framed within the context of a manufacturing environment.
    Consider the following core principles:
    - **Operational Cycles:** You MUST use the 'Summary of Trends & Fluctuations' to understand the operational sequence (e.g., start-up, peak load, idle time) and correlate it with the detailed statistics.
    - **Equipment Health:** Interpret electrical data as indicators of mechanical health. Use the 'Detailed Statistical Summary' to understand the magnitude and volatility of loads.
    - **Cost Reduction:** Link your findings directly to cost-saving opportunities by focusing on reducing peak demand (MD) and improving power factor.
    - **Quantitative Significance:** When analyzing percentage-based metrics (like current imbalance), you MUST refer to the absolute values in the 'Detailed Statistical Summary' to determine the real-world impact.
    Provide a concise, actionable report in Markdown format with three sections: 1. Executive Summary, 2. Key Observations & Pattern Analysis, and 3. Actionable Recommendations. Address the user as a fellow process optimization engineer."""
    
    user_prompt = f"""
    Good morning, Please analyze the following power consumption data for an industrial machine at our Suva facility.
    **Key Performance Indicators:**\n{summary_metrics}
    **Summary of Trends & Fluctuations:**\n{trend_summary}
    **Detailed Statistical Summary (Key Metrics):**\n{detailed_stats}
    **Measurement Parameters:**\n{params_info}
    """
    if additional_context:
        user_prompt += f"**Additional Engineer's Context:**\n{additional_context}"
    user_prompt += "\nBased on all this information, please generate a report with your insights and recommendations for process optimization."
    
    try:
        api_key = st.secrets["GEMINI_API_KEY"]
    except (KeyError, FileNotFoundError):
        return "Error: Gemini API key not found. Please add it to your Streamlit Secrets."
    
    # Use the newer gemini-1.5-flash-latest model
    api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent?key={api_key}"
    
    payload = {
        "contents": [{"parts": [{"text": user_prompt}]}],
        "systemInstruction": {"parts": [{"text": system_prompt}]}
    }
    
    try:
        response = requests.post(api_url, json=payload, timeout=120)
        response.raise_for_status() # Will raise an exception for 4XX/5XX errors
        
        result = response.json()
        
        if 'error' in result:
            return f"Error from Gemini API: {result['error']['message']}"
            
        candidate = result.get('candidates', [{}])[0]
        
        # Check for safety ratings or finish reason
        if candidate.get('finishReason') not in ['STOP', 'MAX_TOKENS']:
             return f"Error: API call finished unexpectedly. Reason: {candidate.get('finishReason', 'Unknown')}"

        if not candidate.get('content', {}).get('parts', []):
            return "Error: API returned an empty response. This may be due to safety filters."
            
        content = candidate['content']['parts'][0]
        return content.get('text', "Error: Could not extract analysis from the API response.")
        
    except requests.exceptions.HTTPError as http_err:
        return f"HTTP error occurred: {http_err} - {response.text}"
    except requests.exceptions.RequestException as req_err:
        return f"A network error occurred: {req_err}"
    except Exception as e:
        return f"An unexpected error occurred while contacting the AI: {e}"

# --- 3. Helper Function for Excel Download ---

def to_excel_bytes(df: pd.DataFrame) -> bytes:
    """Converts a DataFrame to an in-memory Excel file (bytes)."""
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Processed_Data')
    return output.getvalue()

# --- 4. Streamlit UI and Analysis Section ---
st.set_page_config(layout="wide", page_title="FMF Power Consumption Analysis")
st.title("⚡ FMF Power Consumption Analysis Dashboard")

# Use requested date format for the title
current_time_fiji = pd.Timestamp.now(tz='Pacific/Fiji').strftime('%a %d %b %Y')
st.markdown(f"**Suva, Fiji** | {current_time_fiji}")

st.sidebar.header("Upload Data")
uploaded_file = st.sidebar.file_uploader("Upload a raw CSV from your Hioki Power Analyzer", type=["csv"])

if uploaded_file is None:
    st.info("Please upload a CSV file to begin analysis.")
else:
    # Use the cached function to load data
    process_result = load_hioki_data(uploaded_file)

    if process_result:
        wiring_system, parameters, data_raw = process_result # Renamed to data_raw
        
        # --- NEW LOGIC: SEPARATE RAW FROM CLEAN ---
        
        # 1. Filter the raw data to create the clean analysis DataFrame
        data_full = pd.DataFrame() # Initialize
        
        if 'Machine Status' in data_raw.columns:
            # Coerce to numeric, turning 'ERR' codes into NaN. Handles '0', '0.0', '00000000'
            data_raw['Status_Numeric'] = pd.to_numeric(data_raw['Machine Status'], errors='coerce')
            # Filter for rows where status is exactly 0
            data_full = data_raw[data_raw['Status_Numeric'] == 0].copy()
            
            rows_dropped = len(data_raw) - len(data_full)
            if rows_dropped > 0:
                st.sidebar.info(f"Filtered out {rows_dropped} rows with non-zero status codes for analysis.")
            if data_full.empty:
                st.error("No data with Status=0 was found. Cannot perform analysis.")
                st.stop()
        else:
            st.sidebar.warning("Could not find 'Machine Status' column. Analyzing all data.")
            data_full = data_raw.copy()
        
        # --- END NEW LOGIC ---
            
        st.sidebar.success(f"File processed successfully!\n\n**Mode: {wiring_system} Analysis**")
        
        if data_full.empty:
            st.error("File was processed, but no valid data was found. Please check the file contents.")
            st.stop() # Stop execution if no data
            
        data = data_full.copy() # Make a copy for filtering

        if not data.empty:
            st.sidebar.markdown("---")
            st.sidebar.subheader("Filter Data by Time")
            min_ts, max_ts = data_full['Datetime'].min(), data_full['Datetime'].max()
            
            # Use requested date format for the slider
            slider_format = "DD/MM/YY - HH:mm"
            
            start_time, end_time = st.sidebar.slider(
                "Select a time range for analysis:",
                min_value=min_ts.to_pydatetime(), max_value=max_ts.to_pydatetime(),
                value=(min_ts.to_pydatetime(), max_ts.to_pydatetime()),
                format=slider_format
            )
            # Filter the main 'data' DataFrame (which is a copy of data_full)
            data = data_full[(data_full['Datetime'] >= start_time) & (data_full['Datetime'] <= end_time)].copy()
            
            if data.empty:
                st.warning("No data found in the selected time range. Try expanding the filter.")
                st.stop() # Stop execution if filter results in no data
        
        kpi_summary = {}
        
        if wiring_system == '1P2W':
            st.header("Single-Phase Performance Analysis")
            
            total_kwh = 0
            energy_col = 'Consumed Real Energy (Wh)'
            if energy_col in data.columns and not data[energy_col].dropna().empty:
                energy_vals = data[energy_col].dropna()
                if len(energy_vals) > 1: total_kwh = (energy_vals.iloc[-1] - energy_vals.iloc[0]) / 1000
            
            peak_kva = data['Avg Apparent Power (kVA)'].max() if 'Avg Apparent Power (kVA)' in data.columns else 0
            avg_kw = data['Avg Real Power (kW)'].abs().mean() if 'Avg Real Power (kW)' in data.columns else 0
            avg_pf = data['Power Factor'].abs().mean() if 'Power Factor' in data.columns else 0
            
            st.subheader("Performance Metrics")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Consumed Energy", f"{total_kwh:.2f} kWh" if total_kwh > 0 else "N/A")
            col2.metric("Peak Demand (MD)", f"{peak_kva:.2f} kVA" if peak_kva > 0 else "N/A")
            col3.metric("Average Power Draw", f"{avg_kw:.2f} kW" if avg_kw > 0 else "N/A")
            col4.metric("Average Power Factor", f"{avg_pf:.3f}" if avg_pf > 0 else "N/A")

            kpi_summary = { 
                "Analysis Mode": "Single-Phase", "Total Consumed Energy": f"{total_kwh:.2f} kWh",
                "Peak Demand (MD)": f"{peak_kva:.2f} kVA", "Average Power Draw": f"{avg_kw:.2f} kW",
                "Average Power Factor": f"{avg_pf:.3f}"
            }
            
            tab_names = ["⚡ Power & Energy", "📝 Measurement Settings", "📋 Full Data Table"]
            
            tabs = st.tabs(tab_names)
            with tabs[0]:
                plot_cols = [col for col in ['Avg Real Power (kW)', 'Avg Apparent Power (kVA)', 'Avg Reactive Power (kVAR)'] if col in data.columns]
                if plot_cols:
                    st.subheader("Power Consumption Over Time")
                    st.info("This graph shows the Real (useful work), Apparent (total supplied), and Reactive (wasted) power. Look for high Apparent or Reactive power relative to Real power, which indicates electrical inefficiency.")
                    fig_power = px.line(data, x='Datetime', y=plot_cols)
                    st.plotly_chart(fig_power, use_container_width=True)
                    with st.expander("Show Key Power Statistics"):
                        stats_data = {
                            "Max Real Power": f"{data['Avg Real Power (kW)'].max():.2f} kW" if 'Avg Real Power (kW)' in data.columns else "N/A",
                            "Max Apparent Power": f"{data['Avg Apparent Power (kVA)'].max():.2f} kVA" if 'Avg Apparent Power (kVA)' in data.columns else "N/A"
                        }
                        st.json(stats_data)
                if 'Power Factor' in data.columns:
                    st.subheader("Power Factor Over Time")
                    st.info("Power Factor is an efficiency score (Real Power / Apparent Power). Values below 0.95 (the red line) can lead to utility penalties and indicate wasted energy.")
                    fig_pf = px.line(data, x='Datetime', y='Power Factor')
                    fig_pf.add_hline(y=0.95, line_dash="dash", line_color="red")
                    st.plotly_chart(fig_pf, use_container_width=True)
                    with st.expander("Show Power Factor Statistics"):
                        stats_pf = {
                            "Minimum Power Factor": f"{data['Power Factor'].min():.3f}",
                            "Average Power Factor": f"{data['Power Factor'].mean():.3f}"
                        }
                        st.json(stats_pf)

            with tabs[1]:
                st.subheader("Measurement Settings")
                st.dataframe(parameters)
            with tabs[2]:
                st.subheader("Full Raw Data Table (All Status Codes)")
                # Show the original data_raw table
                st.dataframe(data_raw)
                
                # Download button downloads the CLEAN (Status=0) data_full
                excel_data = to_excel_bytes(data_full)
                st.download_button(
                    label="📥 Download CLEAN (Status=0) Data as Excel",
                    data=excel_data,
                    file_name=f"{uploaded_file.name.split('.')[0]}_processed_clean.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    help="This downloads only the data rows with Status=0, which are used for all graphs and analysis."
                )

        elif wiring_system == '3P4W':
            st.header("Three-Phase System Diagnostic")
            
            # KPIs are calculated on the time-filtered 'data' (which is Status=0)
            avg_power_kw = data['Total Avg Real Power (kW)'].mean() if 'Total Avg Real Power (kW)' in data.columns else 0
            avg_pf = data['Total Power Factor'].mean() if 'Total Power Factor' in data.columns else 0
            
            # Find the true peak kVA (Max Demand)
            peak_kva_3p = 0
            if 'Total Max Apparent Power (kVA)' in data.columns:
                 peak_kva_3p = data['Total Max Apparent Power (kVA)'].max()
            elif 'Total Avg Apparent Power (kVA)' in data.columns:
                 peak_kva_3p = data['Total Avg Apparent Power (kVA)'].max()

            imbalance = 0
            current_cols_avg = ['L1 Avg Current (A)', 'L2 Avg Current (A)', 'L3 Avg Current (A)']
            if all(c in data.columns for c in current_cols_avg):
                avg_currents = data[current_cols_avg].mean()
                if avg_currents.mean() > 0: imbalance = (avg_currents.max() - avg_currents.min()) / avg_currents.mean() * 100
            
            st.subheader("Performance Metrics")
            col1, col2, col3, col4 = st.columns(4)
            
            # --- KPI FORMATTING FIX ---
            col1.metric("Avg. Total Power", f"{avg_power_kw:.2f} kW" if avg_power_kw > 0 else "N/A")
            col2.metric("Peak Demand (MD)", f"{peak_kva_3p:.2f} kVA" if peak_kva_3p > 0 else "N/A")
            col3.metric("Avg. Total PF", f"{avg_pf:.3f}" if avg_pf > 0 else "N/A")
            # --- END FIX ---
            
            col4.metric("Max Current Imbalance", f"{imbalance:.1f} %" if imbalance > 0 else "N/A", help="Under 5% is good.")

            kpi_summary = { 
                "Analysis Mode": "Three-Phase", "Average Total Power": f"{avg_power_kw:.2f} kW",
                "Peak Demand (MD)": f"{peak_kva_3p:.2f} kVA", "Average Total Power Factor": f"{avg_pf:.3f}",
                "Max Current Imbalance": f"{imbalance:.1f} %"
            }
            
            tab_names_3p = ["📅 Daily Breakdown", "📊 Current & Load Balance", "🩺 Voltage Health", "⚡ Power Analysis", "⚖️ Power Factor", "📝 Settings", "📋 Full Data Table"]
            tabs = st.tabs(tab_names_3p)
            
            with tabs[0]:
                st.subheader("24-Hour Operational Snapshot")
                st.info("Select a specific day to generate a detailed 24-hour subplot of all key electrical parameters. This is essential for comparing shift performance or analyzing specific production runs.")
                
                # Use the clean, (Status=0) data_full for day selection
                unique_days = data_full['Datetime'].dt.date.unique()
                
                # Use requested date format
                day_format_func = lambda d: d.strftime('%a %d %b %Y')
                
                selected_day = st.selectbox("Select a day for detailed analysis:", options=unique_days, format_func=day_format_func)
                
                if selected_day:
                    # Filter the clean data_full for the selected day
                    daily_data = data_full[data_full['Datetime'].dt.date == selected_day]
                    
                    if daily_data.empty:
                        st.warning("No data found for the selected day.")
                    else:
                        fig = make_subplots(rows=4, cols=1, shared_xaxes=True, subplot_titles=("Voltage Envelope (V)", "Current Envelope (A)", "Real Power (kW)", "Power Factor"))

                        # --- PLOTTING FIX ---

                        # 1. Voltage -> Row 1
                        for i in range(1, 4):
                            for stat in ['Min', 'Avg', 'Max']:
                                col = f'L{i} {stat} Voltage (V)'
                                if col in daily_data.columns:
                                    fig.add_trace(go.Scatter(x=daily_data['Datetime'], y=daily_data[col], name=col, mode='lines'), row=1, col=1)

                        # 2. Current -> Row 2
                        for i in range(1, 4):
                            for stat in ['Min', 'Avg', 'Max']:
                                col = f'L{i} {stat} Current (A)'
                                if col in daily_data.columns:
                                    fig.add_trace(go.Scatter(x=daily_data['Datetime'], y=daily_data[col], name=col, mode='lines'), row=2, col=1)

                        # 3. Real Power -> Row 3
                        for i in range(1, 4):
                            col = f'L{i} Avg Real Power (kW)'
                            if col in daily_data.columns:
                                fig.add_trace(go.Scatter(x=daily_data['Datetime'], y=daily_data[col], name=col, mode='lines'), row=3, col=1)

                        # 4. Power Factor -> Row 4
                        for i in range(1, 4):
                            col = f'L{i} Power Factor'
                            if col in daily_data.columns:
                                fig.add_trace(go.Scatter(x=daily_data['Datetime'], y=daily_data[col], name=col, mode='lines'), row=4, col=1)
                        
                        # --- END FIX ---

                        # Updated chart title format to match request
                        fig.update_layout(height=1000, title_text=f"Full Operational Breakdown for {selected_day.strftime('%a %d %b %Y')}", showlegend=True)
                        st.plotly_chart(fig, use_container_width=True)
            
            # The rest of the tabs use the time-filtered 'data' (which is Status=0)
            with tabs[1]:
                st.subheader("Current Operational Envelope per Phase")
                st.info("This chart shows the full range of current drawn by the machine on each phase, from minimum to maximum. It is crucial for identifying peak inrush currents during start-up and understanding the full load variation.")
                current_cols_all = [f'{p} {s} Current (A)' for p in ['L1', 'L2', 'L3'] for s in ['Min', 'Avg', 'Max']]
                plot_cols = [c for c in current_cols_all if c in data.columns]
                if plot_cols:
                    fig = px.line(data, x='Datetime', y=plot_cols)
                    st.plotly_chart(fig, use_container_width=True)
                    with st.expander("Show Current Statistics"):
                        st.dataframe(data[plot_cols].describe().T[['mean', 'min', 'max']].rename(columns={'mean':'Average', 'min':'Minimum', 'max':'Maximum'}))

            with tabs[2]:
                st.subheader("Voltage Operational Envelope per Phase")
                st.info("This chart displays the voltage stability across all three phases, showing the minimum, average, and maximum recorded values. It is essential for diagnosing power quality issues like voltage sags (dips) under load or surges (spikes) from the grid.")
                voltage_cols_all = [f'{p} {s} Voltage (V)' for p in ['L1', 'L2', 'L3'] for s in ['Min', 'Avg', 'Max']]
                plot_cols = [c for c in voltage_cols_all if c in data.columns]
                if plot_cols:
                    fig = px.line(data, x='Datetime', y=plot_cols)
                    st.plotly_chart(fig, use_container_width=True)
                    with st.expander("Show Voltage Statistics"):
                        st.dataframe(data[plot_cols].describe().T[['mean', 'min', 'max']].rename(columns={'mean':'Average', 'min':'Minimum', 'max':'Maximum'}))

            with tabs[3]:
                st.subheader("Power Analysis")
                st.info("These charts show the Real (useful work), Apparent (total), and Reactive (wasted) power. The top chart shows the total system power, while the bottom chart breaks down the real power by phase to identify imbalances in work being done.")
                total_power_cols = [c for c in ['Total Avg Real Power (kW)', 'Total Avg Apparent Power (kVA)', 'Total Avg Reactive Power (kVAR)'] if c in data.columns]
                if total_power_cols:
                    fig = px.line(data, x='Datetime', y=total_power_cols)
                    st.plotly_chart(fig, use_container_width=True)

                phase_power_cols = [c for c in ['L1 Avg Real Power (kW)', 'L2 Avg Real Power (kW)', 'L3 Avg Real Power (kW)'] if c in data.columns]
                if phase_power_cols:
                    fig2 = px.line(data, x='Datetime', y=phase_power_cols)
                    st.plotly_chart(fig2, use_container_width=True)

            with tabs[4]:
                st.subheader("Power Factor per Phase")
                st.info("Power factor is a measure of electrical efficiency. A value of 1.0 is perfect. Values below 0.95 often incur utility penalties. This chart helps identify if one specific phase is the cause of poor overall efficiency.")
                pf_cols = [c for c in ['L1 Power Factor', 'L2 Power Factor', 'L3 Power Factor'] if c in data.columns]
                if pf_cols:
                    fig = px.line(data, x='Datetime', y=pf_cols)
                    st.plotly_chart(fig, use_container_width=True)
                    with st.expander("Show Power Factor Statistics"):
                        st.dataframe(data[pf_cols].describe().T[['mean', 'min', 'max']].rename(columns={'mean':'Average', 'min':'Minimum', 'max':'Maximum'}))

            with tabs[5]:
                st.subheader("Measurement Settings")
                st.dataframe(parameters)
            
            with tabs[6]:
                st.subheader("Full Raw Data Table (All Status Codes)")
                # Show the original data_raw table
                st.dataframe(data_raw)
                
                # Download button downloads the CLEAN (Status=0) data_full
                excel_data = to_excel_bytes(data_full)
                st.download_button(
                    label="📥 Download CLEAN (Status=0) Data as Excel",
                    data=excel_data,
                    file_name=f"{uploaded_file.name.split('.')[0]}_processed_clean.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    help="This downloads only the data rows with Status=0, which are used for all graphs and analysis."
                )
        
        # --- AI Section (Common to both) ---
        st.sidebar.markdown("---")
        st.sidebar.subheader("Add Custom AI Context")
        additional_context = st.sidebar.text_area("Provide specific details about the machine or process (optional):", help="E.g., 'This is the main dough mixer, model XYZ.' or 'The large spike at 10:00 was a planned startup.'")

        if st.sidebar.button("🤖 Get AI-Powered Analysis"):
            # Use the time-filtered 'data' (which is Status=0) for the AI analysis
            if data.empty:
                st.sidebar.error("Cannot run analysis on an empty dataset. Widen your time filter.")
            else:
                with st.spinner("🧠 AI is analyzing the data... This may take a moment."):
                    summary_metrics_text = "\n".join([f"- {key}: {value}" for key, value in kpi_summary.items() if "N/A" not in str(value)])
                    trend_summary_text = generate_trend_summary(data, wiring_system)
                    # --- NEW STATS CALL ---
                    data_stats_text = generate_statistical_summary(data, wiring_system)
                    # --- END NEW STATS CALL ---
                    params_info_text = parameters.to_string()
                    
                    ai_response = get_gemini_analysis(summary_metrics_text, data_stats_text, trend_summary_text, params_info_text, additional_context)
                    st.session_state['ai_analysis'] = ai_response

        if 'ai_analysis' in st.session_state:
            st.markdown("---")
            st.header("🤖 AI-Powered Analysis")
            st.markdown(st.session_state['ai_analysis'])

    elif uploaded_file is not None:
        # This message will show if process_result is None (e.g., from an error in load_hioki_data)
        st.warning("Could not process the uploaded file. Please ensure it is a valid, non-empty Hioki CSV export.")
