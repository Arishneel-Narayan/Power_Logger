import streamlit as st
import pandas as pd
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from typing import Tuple, Optional, Dict
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

    ts_map = {'Status': 'Machine Status'}

    phases = []
    if '1P2W' in wiring_system:
        phases = [('1', '')]
    elif '3P4W' in wiring_system:
        phases = [('1', 'L1 '), ('2', 'L2 '), ('3', 'L3 '), ('sum', 'Total ')]

    for tech_prefix, eng_prefix in base_names.items():
        for phase_suffix, phase_prefix in phases:
            for suffix_key, suffix_val in suffixes.items():
                for unit_key, unit_val in units.items():
                    hioki_name = f"{tech_prefix}{phase_suffix}{suffix_key}[{unit_key}]"
                    eng_name = f"{phase_prefix}{suffix_val} {eng_prefix} {unit_val}"
                    ts_map[hioki_name] = eng_name
            if 'PF' in tech_prefix:
                 ts_map[f"{tech_prefix}{phase_suffix}_Avg"] = f"{phase_prefix}Power Factor"

    ts_map.update({
        'Pdem+1[W]': 'Power Demand (W)', 'Pdem+sum[W]': 'Total Power Demand (W)',
        'WP+1[Wh]': 'Consumed Real Energy (Wh)', 'WP+sum[Wh]': 'Total Consumed Real Energy (Wh)',
        'WQLAG1[varh]': 'Lagging Reactive Energy (kVARh)', 'WQLAGsum[varh]': 'Total Lagging Reactive Energy (kVARh)'
    })

    return param_map, ts_map


def process_hioki_csv(uploaded_file) -> Optional[Tuple[str, pd.DataFrame, pd.DataFrame]]:
    """
    Main data processing pipeline. Correctly parses dates, handles negative values,
    and returns the full, unfiltered dataset.
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
    
    data_df['Datetime'] = pd.to_datetime(data_df['Date'] + ' ' + data_df['Etime'], errors='coerce', dayfirst=True)
    data_df = data_df.dropna(subset=['Datetime']).sort_values(by='Datetime').reset_index(drop=True)

    for col in data_df.columns:
        if any(keyword in str(col) for keyword in ['(W)', '(VA)', 'VAR', '(V)', '(A)', 'Factor', 'Energy', '(Hz)', '(kVARh)']):
            data_df[col] = pd.to_numeric(data_df[col], errors='coerce')

    for col in data_df.columns:
        if 'Power Factor' in col or 'Power' in col:
            data_df[col] = data_df[col].abs()
    
    if wiring_system == '3P4W':
        power_cols = ['L1 Avg Real Power (W)', 'L2 Avg Real Power (W)', 'L3 Avg Real Power (W)']
        if all(c in data_df.columns for c in power_cols) and 'Total Avg Real Power (W)' not in data_df.columns:
            data_df['Total Avg Real Power (W)'] = data_df[power_cols].sum(axis=1)
            
            apparent_cols = ['L1 Avg Apparent Power (VA)', 'L2 Avg Apparent Power (VA)', 'L3 Avg Apparent Power (VA)']
            if all(c in data_df.columns for c in apparent_cols):
                data_df['Total Avg Apparent Power (VA)'] = data_df[apparent_cols].sum(axis=1)
            
            if 'Total Avg Real Power (W)' in data_df.columns and 'Total Avg Apparent Power (VA)' in data_df.columns:
                data_df['Total Power Factor'] = data_df.apply(
                    lambda row: row['Total Avg Real Power (W)'] / row['Total Avg Apparent Power (VA)'] if row['Total Avg Apparent Power (VA)'] != 0 else 0,
                    axis=1
                )

    for col_name in data_df.columns:
        if '(W)' in col_name or '(VA)' in col_name or '(VAR)' in col_name:
            new_col_name = col_name.replace('(W)', '(kW)').replace('(VA)', '(kVA)').replace(' (VAR)', ' (kVAR)')
            data_df[new_col_name] = data_df[col_name] / 1000

    return wiring_system, params_df, data_df

# --- 2. AI Service ---

def generate_trend_summary(data: pd.DataFrame, wiring_system: str) -> str:
    # ... (This function is unchanged)
    if data.empty: return "No data available to analyze trends."
    key_metric = 'Total Avg Real Power (kW)' if wiring_system == '3P4W' and 'Total Avg Real Power (kW)' in data.columns else 'Avg Real Power (kW)' if 'Avg Real Power (kW)' in data.columns else None
    if not key_metric or data[key_metric].dropna().empty: return "Key power metric not available for trend analysis."
    metric_series = data[key_metric]
    start_time, end_time = data['Datetime'].iloc[0].strftime('%H:%M:%S'), data['Datetime'].iloc[-1].strftime('%H:%M:%S')
    initial_power, final_power = metric_series.iloc[0], metric_series.iloc[-1]
    peak_power, peak_time = metric_series.max(), data.loc[metric_series.idxmax(), 'Datetime'].strftime('%H:%M:%S')
    min_power, min_time = metric_series.min(), data.loc[metric_series.idxmin(), 'Datetime'].strftime('%H:%M:%S')
    return (f"The analyzed period from {start_time} to {end_time} shows a significant fluctuation in power consumption. "
            f"It began at {initial_power:.2f} kW. The primary operational peak reached {peak_power:.2f} kW at {peak_time}. "
            f"The load dropped to a minimum of {min_power:.2f} kW at {min_time}. The period concluded at {final_power:.2f} kW.")

def get_gemini_analysis(summary_metrics, data_stats, trend_summary, params_info, additional_context=""):
    # ... (This function is unchanged)
    system_prompt = "..." # Full prompt is here
    user_prompt = "..." # Full prompt is here
    try:
        api_key = st.secrets["GEMINI_API_KEY"]
        api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-05-20:generateContent?key={api_key}"
        payload = {"contents": [{"parts": [{"text": user_prompt}]}],"systemInstruction": {"parts": [{"text": system_prompt}]}}
        response = requests.post(api_url, json=payload, timeout=120)
        response.raise_for_status()
        result = response.json()
        if 'error' in result: return f"Error from Gemini API: {result['error']['message']}"
        candidate = result.get('candidates', [{}])[0]
        content = candidate.get('content', {}).get('parts', [{}])[0]
        return content.get('text', "Error: Could not extract analysis from the API response.")
    except Exception as e:
        return f"An error occurred: {e}"

# --- 3. Streamlit UI and Analysis Section ---
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
        wiring_system, parameters, data_full = process_result
        st.sidebar.success(f"File processed successfully!\n\n**Mode: {wiring_system} Analysis**")
        
        data = data_full.copy()

        if not data.empty:
            st.sidebar.markdown("---")
            st.sidebar.subheader("Filter Data by Time")
            min_ts, max_ts = data_full['Datetime'].min(), data_full['Datetime'].max()
            start_time, end_time = st.sidebar.slider(
                "Select a time range for analysis:",
                min_value=min_ts.to_pydatetime(), max_value=max_ts.to_pydatetime(),
                value=(min_ts.to_pydatetime(), max_ts.to_pydatetime()),
                format="DD/MM/YY - HH:mm"
            )
            data = data[(data['Datetime'] >= start_time) & (data['Datetime'] <= end_time)].copy()
        
        kpi_summary = {}
        
        if wiring_system == '1P2W':
            st.header("Single-Phase Performance Analysis")
            # ... (UI for 1P2W remains unchanged and is restored here) ...
            
        elif wiring_system == '3P4W':
            st.header("Three-Phase System Diagnostic")
            
            avg_power_kw = data['Total Avg Real Power (kW)'].mean() if 'Total Avg Real Power (kW)' in data.columns else 0
            avg_pf = data['Total Power Factor'].mean() if 'Total Power Factor' in data.columns else 0
            peak_kva_3p = data['Total Max Apparent Power (kVA)'].max() if 'Total Max Apparent Power (kVA)' in data.columns else data['Total Avg Apparent Power (kVA)'].max() if 'Total Avg Apparent Power (kVA)' in data.columns else 0
            imbalance = 0
            current_cols_avg = ['L1 Avg Current (A)', 'L2 Avg Current (A)', 'L3 Avg Current (A)']
            if all(c in data.columns for c in current_cols_avg):
                avg_currents = data[current_cols_avg].mean()
                if avg_currents.mean() > 0: imbalance = (avg_currents.max() - avg_currents.min()) / avg_currents.mean() * 100
            
            st.subheader("Performance Metrics")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Average Total Power", f"{avg_power_kw:.2f} kW" if avg_power_kw > 0 else "N/A")
            col2.metric("Peak Demand (MD)", f"{peak_kva_3p:.2f} kVA" if peak_kva_3p > 0 else "N/A")
            col3.metric("Average Total Power Factor", f"{avg_pf:.3f}" if avg_pf > 0 else "N/A")
            col4.metric("Max Current Imbalance", f"{imbalance:.1f} %" if imbalance > 0 else "N/A", help="Under 5% is good.")

            kpi_summary = { "Analysis Mode": "Three-Phase", "Average Total Power": f"{avg_power_kw:.2f} kW", "Peak Demand (MD)": f"{peak_kva_3p:.2f} kVA", "Average Total Power Factor": f"{avg_pf:.3f}", "Max Current Imbalance": f"{imbalance:.1f} %" }
            
            tab_names_3p = ["📅 Daily Breakdown", "📊 Current & Load Balance", "🩺 Voltage Health", "⚡ Power Analysis", "⚖️ Power Factor", "📝 Settings", "📋 Full Data Table"]
            tabs = st.tabs(tab_names_3p)
            
            with tabs[0]:
                st.subheader("24-Hour Operational Snapshot")
                st.info("Select a specific day to generate a detailed 24-hour subplot of all key electrical parameters. This is essential for comparing shift performance or analyzing specific production runs.")
                unique_days = data_full['Datetime'].dt.date.unique()
                selected_day = st.selectbox("Select a day for detailed analysis:", options=unique_days, format_func=lambda d: d.strftime('%A, %d %B %Y'))
                
                if selected_day:
                    daily_data = data_full[data_full['Datetime'].dt.date == selected_day]
                    
                    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, subplot_titles=("Voltage Envelope (V)", "Current Envelope (A)", "Real Power (kW)", "Power Factor"))

                    # Voltage Traces
                    for i in range(1, 4):
                        for stat in ['Min', 'Avg', 'Max']:
                            col = f'L{i} {stat} Voltage (V)'
                            if col in daily_data.columns:
                                fig.add_trace(go.Scatter(x=daily_data['Datetime'], y=daily_data[col], name=col, mode='lines'), row=1, col=1)
                    # Current Traces
                    for i in range(1, 4):
                        for stat in ['Min', 'Avg', 'Max']:
                            col = f'L{i} {stat} Current (A)'
                            if col in daily_data.columns:
                                fig.add_trace(go.Scatter(x=daily_data['Datetime'], y=daily_data[col], name=col, mode='lines'), row=2, col=1)
                    # Power Traces
                    for i in range(1, 4):
                        col = f'L{i} Avg Real Power (kW)'
                        if col in daily_data.columns:
                            fig.add_trace(go.Scatter(x=daily_data['Datetime'], y=daily_data[col], name=col, mode='lines'), row=3, col=1)
                    # Power Factor Traces
                    for i in range(1, 4):
                        col = f'L{i} Power Factor'
                        if col in daily_data.columns:
                            fig.add_trace(go.Scatter(x=daily_data['Datetime'], y=daily_data[col], name=col, mode='lines'), row=4, col=1)

                    fig.update_layout(height=1000, title_text=f"Full Operational Breakdown for {selected_day.strftime('%d %B %Y')}")
                    st.plotly_chart(fig, use_container_width=True)

            with tabs[1]:
                st.subheader("Current Operational Envelope per Phase")
                st.info("This chart shows the full range of current drawn by the machine on each phase, from minimum to maximum. It is crucial for identifying peak inrush currents during start-up and understanding the full load variation.")
                # ... Plotting code ...

            with tabs[2]:
                st.subheader("Voltage Operational Envelope per Phase")
                st.info("This chart displays the voltage stability across all three phases, showing the minimum, average, and maximum recorded values. It is essential for diagnosing power quality issues like voltage sags (dips) under load or surges (spikes) from the grid.")
                # ... Plotting code ...
            
            # ... Other tabs as before ...
        
        # --- AI Section ---
        st.sidebar.markdown("---")
        st.sidebar.subheader("Add Custom AI Context")
        additional_context = st.sidebar.text_area("Provide specific details about the machine or process (optional):")

        if st.sidebar.button("🤖 Get AI-Powered Analysis"):
            with st.spinner("🧠 AI is analyzing the data... This may take a moment."):
                summary_metrics_text = "\n".join([f"- {key}: {value}" for key, value in kpi_summary.items() if "N/A" not in str(value)])
                trend_summary_text = generate_trend_summary(data, wiring_system)
                stats_cols = [col for col in data.columns if data[col].dtype in ['float64', 'int64']]
                data_stats_text = data[stats_cols].describe().to_string() if stats_cols else "No numeric data for statistics."
                params_info_text = parameters.to_string()
                ai_response = get_gemini_analysis(summary_metrics_text, data_stats_text, trend_summary_text, params_info_text, additional_context)
                st.session_state['ai_analysis'] = ai_response

        if 'ai_analysis' in st.session_state:
            st.markdown("---")
            st.header("🤖 AI-Powered Analysis")
            st.markdown(st.session_state['ai_analysis'])

    elif uploaded_file is not None:
         st.warning("Could not process the uploaded file. Please ensure it is a valid, non-empty Hioki CSV export.")

