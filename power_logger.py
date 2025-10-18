import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from typing import Tuple, Optional, List
import requests

# --- 1. Core Data Processing Engine ---

def get_rename_map(wiring_system: str) -> dict:
    """
    Dynamically generates a comprehensive rename map for all relevant columns
    based on the detected electrical wiring system.
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
    units = {'V': '(V)', 'A': '(A)', 'W': '(W)', 'VA': '(VA)', 'var': '(VAR)', 'Hz': '(Hz)', 'deg': '(deg)', 'Wh': '(Wh)', 'varh': '(kVARh)'}
    ts_map = {'Status': 'Machine Status'}
    phases = [('1', '')] if '1P2W' in wiring_system else [('1', 'L1 '), ('2', 'L2 '), ('3', 'L3 '), ('sum', 'Total ')] if '3P4W' in wiring_system else []
    for tech, eng in base_names.items():
        for phase_sfx, phase_pfx in phases:
            for sfx_k, sfx_v in suffixes.items():
                for unit_k, unit_v in units.items():
                    ts_map[f"{tech}{phase_sfx}{sfx_k}[{unit_k}]"] = f"{phase_pfx}{sfx_v} {eng} {unit_v}"
            if 'PF' in tech: ts_map[f"{tech}{phase_sfx}_Avg"] = f"{phase_pfx}Power Factor"
    ts_map.update({
        'Pdem+1[W]': 'Power Demand (W)', 'Pdem+sum[W]': 'Total Power Demand (W)', 'WP+1[Wh]': 'Consumed Real Energy (Wh)',
        'WP+sum[Wh]': 'Total Consumed Real Energy (Wh)', 'WQLAG1[varh]': 'Lagging Reactive Energy (kVARh)',
        'WQLAGsum[varh]': 'Total Lagging Reactive Energy (kVARh)'
    })
    return param_map, ts_map


def process_hioki_csv(uploaded_file) -> Optional[Tuple[str, pd.DataFrame, pd.DataFrame]]:
    """
    Main data processing pipeline. It returns the detected wiring system and the cleaned dataframes
    for parameters and time-series data.
    """
    try:
        df_raw = pd.read_csv(uploaded_file, header=None, on_bad_lines='skip', encoding='utf-8')
    except Exception as e:
        st.error(f"Error reading CSV file: {e}")
        return None
    header_idx = df_raw[df_raw.iloc[:, 0].astype(str).eq('Date')].index
    if header_idx.empty:
        st.error("Header 'Date' not found. This may not be a valid Hioki file.")
        return None
    header_row_index = header_idx[0]
    try:
        wiring_system = df_raw.iloc[:header_row_index, :2].set_index(0).loc['WIRING', 1]
    except KeyError:
        st.sidebar.error("Could not determine wiring system from metadata.")
        return None
    param_rename_map, ts_rename_map = get_rename_map(wiring_system)
    params_df = df_raw.iloc[:header_row_index, :2].copy()
    params_df.columns = ['Parameter', 'Value']
    params_df.set_index('Parameter', inplace=True)
    params_df.dropna(inplace=True)
    params_df = params_df.rename(index=param_rename_map)
    data_df = df_raw.iloc[header_row_index:].copy()
    data_df.columns = data_df.iloc[0]
    data_df = data_df.iloc[1:].reset_index(drop=True).rename(columns=ts_rename_map)
    data_df['Datetime'] = pd.to_datetime(data_df['Date'] + ' ' + data_df['Etime'], errors='coerce', dayfirst=True)
    data_df.dropna(subset=['Datetime'], inplace=True)
    data_df.sort_values(by='Datetime', inplace=True, ignore_index=True)
    for col in data_df.columns:
        if any(k in str(col) for k in ['(W)', '(VA)', 'VAR', '(V)', '(A)', 'Factor', 'Energy', '(Hz)', '(kVARh)']):
            data_df[col] = pd.to_numeric(data_df[col], errors='coerce')
    
    if wiring_system == '3P4W':
        p_cols = [f'L{i} Avg Real Power (W)' for i in range(1, 4)]
        if all(c in data_df for c in p_cols) and 'Total Avg Real Power (W)' not in data_df:
            st.sidebar.info("Calculating Total Power from phase data.")
            data_df['Total Avg Real Power (W)'] = data_df[p_cols].sum(axis=1)
            
            apparent_cols = ['L1 Avg Apparent Power (VA)', 'L2 Avg Apparent Power (VA)', 'L3 Avg Apparent Power (VA)']
            if all(c in data_df.columns for c in apparent_cols):
                data_df['Total Avg Apparent Power (VA)'] = data_df[apparent_cols].sum(axis=1)
            
            reactive_cols = ['L1 Avg Reactive Power (VAR)', 'L2 Avg Reactive Power (VAR)', 'L3 Avg Reactive Power (VAR)']
            if all(c in data_df.columns for c in reactive_cols) and 'Total Avg Reactive Power (VAR)' not in data_df.columns:
                data_df['Total Avg Reactive Power (VAR)'] = data_df[reactive_cols].sum(axis=1)

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

# --- 2. AI and Cost Calculation Services ---
def get_gemini_analysis(summary_metrics, data_stats, params_info, additional_context=""):
    """Contacts the Gemini API for an expert analysis."""
    system_prompt = "You are an expert industrial energy efficiency analyst and process engineer for FMF Foods Ltd., a food manufacturing company in Fiji. Your task is to analyze power consumption data from industrial machinery at our biscuit factory in Suva. Your analysis must be framed within the context of a manufacturing environment. Consider operational cycles, equipment health, and system reliability. Most importantly, link your findings directly to cost-saving opportunities by focusing on reducing peak demand (MD) and improving power factor. Provide a concise, actionable report in Markdown format with three sections: 1. Executive Summary, 2. Key Observations & Pattern Analysis, and 3. Actionable Recommendations. Address the user as a fellow process optimization engineer."
    user_prompt = f"Good morning, Please analyze the following power consumption data from our Suva facility.\n\n**Key Performance Indicators:**\n{summary_metrics}\n\n**Measurement Parameters:**\n{params_info}\n\n**Statistical Summary of Time-Series Data:**\n{data_stats}\n"
    if additional_context: user_prompt += f"**Additional Engineer's Context:**\n{additional_context}\n"
    user_prompt += "Based on all this information, please generate your report."
    try: api_key = st.secrets["GEMINI_API_KEY"]
    except (KeyError, FileNotFoundError): return "Error: Gemini API key not found in Streamlit Secrets."
    api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-05-20:generateContent?key={api_key}"
    payload = {"contents": [{"parts": [{"text": user_prompt}]}],"systemInstruction": {"parts": [{"text": system_prompt}]}}
    try:
        response = requests.post(api_url, json=payload, timeout=120)
        response.raise_for_status()
        result = response.json()
        if 'error' in result: return f"Error from Gemini API: {result['error']['message']}"
        return result.get('candidates', [{}])[0].get('content', {}).get('parts', [{}])[0].get('text', "Error: Could not extract analysis.")
    except requests.exceptions.RequestException as e: return f"An error occurred while contacting the AI Analysis service: {e}"
    except Exception as e: return f"An unexpected error occurred: {e}"

# --- 3. Streamlit UI and Analysis Section ---
st.set_page_config(layout="wide", page_title="FMF Power Consumption Analysis")
st.title("⚡ FMF Power Consumption Analysis Dashboard")
st.markdown(f"**Suva, Fiji** | {pd.Timestamp.now(tz='Pacific/Fiji').strftime('%A, %d %B %Y')}")

st.sidebar.header("Upload Data")
uploaded_file = st.sidebar.file_uploader("1. Upload Hioki Power Analyzer CSV", type=["csv"])
production_log_file = st.sidebar.file_uploader("2. (Optional) Upload Production Log", type=["csv", "xlsx", "xls"])

if uploaded_file is None:
    st.info("Please upload a Hioki CSV file to begin analysis.")
else:
    process_result = process_hioki_csv(uploaded_file)
    if process_result:
        wiring_system, parameters, data_full = process_result
        st.sidebar.success(f"File processed successfully!\n\n**Mode: {wiring_system} Analysis**")
        
        data = data_full.copy()
        if not data.empty:
            st.sidebar.markdown("---")
            st.sidebar.subheader("Filter Data by Time")
            dt_options = data['Datetime'].dt.to_pydatetime()
            start_time, end_time = st.sidebar.select_slider("Select a time range for analysis:", options=dt_options, value=(dt_options[0], dt_options[-1]), format_func=lambda dt: dt.strftime("%d %b, %H:%M"))
            data = data[(data['Datetime'] >= start_time) & (data['Datetime'] <= end_time)].copy()

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
            avg_pf = data['Power Factor'].mean() if 'Power Factor' in data.columns else 0
            
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
            
            tab_names = ["⚡ Power Analysis", "📝 Measurement Settings", "📋 Full Data"]
            tabs = st.tabs(tab_names)

            with tabs[0]:
                st.subheader("Power Analysis")
                if 'Avg Real Power (kW)' in data.columns:
                    fig_kw = px.line(data, x='Datetime', y='Avg Real Power (kW)', title='Real Power (kW) Over Time')
                    st.plotly_chart(fig_kw, use_container_width=True)
                
                if 'Avg Apparent Power (kVA)' in data.columns:
                    fig_kva = px.line(data, x='Datetime', y='Avg Apparent Power (kVA)', title='Apparent Power (kVA) Over Time')
                    st.plotly_chart(fig_kva, use_container_width=True)
                
                if 'Avg Reactive Power (kVAR)' in data.columns:
                    fig_kvar = px.line(data, x='Datetime', y='Avg Reactive Power (kVAR)', title='Reactive Power (kVAR) Over Time')
                    st.plotly_chart(fig_kvar, use_container_width=True)
                
                if 'Power Factor' in data.columns:
                    fig_pf = px.line(data, x='Datetime', y='Power Factor', title='Power Factor Over Time')
                    fig_pf.add_hline(y=0.95, line_dash="dash", line_color="red")
                    st.plotly_chart(fig_pf, use_container_width=True)

            with tabs[1]:
                st.subheader("Measurement Settings")
                st.dataframe(parameters)
            
            with tabs[2]:
                st.subheader("Full Time-Series Data")
                st.dataframe(data)


        elif wiring_system == '3P4W':
            st.header("Three-Phase System Diagnostic")
            
            avg_power_kw = data['Total Avg Real Power (kW)'].mean() if 'Total Avg Real Power (kW)' in data.columns else 0
            avg_pf = data['Total Power Factor'].mean() if 'Total Power Factor' in data.columns else 0
            peak_kva_3p = data['Total Avg Apparent Power (kVA)'].max() if 'Total Avg Apparent Power (kVA)' in data.columns else 0
            imbalance = 0
            current_cols = ['L1 Avg Current (A)', 'L2 Avg Current (A)', 'L3 Avg Current (A)']
            if all(c in data.columns for c in current_cols):
                avg_currents = data[current_cols].mean()
                if avg_currents.mean() > 0: imbalance = (avg_currents.max() - avg_currents.min()) / avg_currents.mean() * 100
            
            st.subheader("Performance Metrics")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Average Total Power", f"{avg_power_kw:.2f} kW" if avg_power_kw > 0 else "N/A")
            col2.metric("Peak Demand (MD)", f"{peak_kva_3p:.2f} kVA" if peak_kva_3p > 0 else "N/A")
            col3.metric("Average Total Power Factor", f"{avg_pf:.3f}" if avg_pf > 0 else "N/A")
            col4.metric("Max Current Imbalance", f"{imbalance:.1f} %" if imbalance > 0 else "N/A", help="Under 5% is good.")

            kpi_summary = { 
                "Analysis Mode": "Three-Phase", "Average Total Power": f"{avg_power_kw:.2f} kW",
                "Peak Demand (MD)": f"{peak_kva_3p:.2f} kVA", "Average Total Power Factor": f"{avg_pf:.3f}",
                "Max Current Imbalance": f"{imbalance:.1f} %"
            }
            
            tab_names_3p = []
            tabs_to_show = {}
            
            # Define which tabs to show based on available data
            tabs_to_show["⚡ Power Analysis"] = True
            if all(c in data.columns for c in current_cols): tabs_to_show["📊 Load Balance"] = True
            if all(c in data.columns for c in ['L1 Avg Voltage (V)', 'L2 Avg Voltage (V)', 'L3 Avg Voltage (V)']): tabs_to_show["🩺 Voltage Health"] = True
            if all(c in data.columns for c in ['L1 Power Factor', 'L2 Power Factor', 'L3 Power Factor']): tabs_to_show["⚖️ Power Factor"] = True
            
            tab_names_3p.extend(list(tabs_to_show.keys()))
            tab_names_3p.extend(["📝 Settings", "📋 Full Data"])
            
            tabs = st.tabs(tab_names_3p)
            current_tab = 0

            if "⚡ Power Analysis" in tabs_to_show:
                with tabs[current_tab]:
                    st.subheader("Total Power Analysis")
                    if 'Total Avg Real Power (kW)' in data.columns:
                        fig_kw = px.line(data, x='Datetime', y='Total Avg Real Power (kW)', title='Total Real Power (kW) Over Time')
                        st.plotly_chart(fig_kw, use_container_width=True)
                    if 'Total Avg Apparent Power (kVA)' in data.columns:
                        fig_kva = px.line(data, x='Datetime', y='Total Avg Apparent Power (kVA)', title='Total Apparent Power (kVA) Over Time')
                        st.plotly_chart(fig_kva, use_container_width=True)
                    if 'Total Power Factor' in data.columns:
                        fig_pf = px.line(data, x='Datetime', y='Total Power Factor', title='Total Power Factor Over Time')
                        fig_pf.add_hline(y=0.95, line_dash="dash", line_color="red")
                        st.plotly_chart(fig_pf, use_container_width=True)
                current_tab += 1

            if "📊 Load Balance" in tabs_to_show:
                with tabs[current_tab]:
                    st.subheader("Current Load Balance Across Phases")
                    fig_balance = px.line(data, x='Datetime', y=current_cols, color_discrete_map={'L1 Avg Current (A)': 'red', 'L2 Avg Current (A)': 'blue', 'L3 Avg Current (A)': 'green'})
                    st.plotly_chart(fig_balance, use_container_width=True)
                    with st.expander("Show Current Statistics"):
                        st.dataframe(data[current_cols].describe().T[['mean', 'min', 'max']].rename(columns={'mean':'Average', 'min':'Minimum', 'max':'Maximum'}))
                current_tab += 1

            if "🩺 Voltage Health" in tabs_to_show:
                with tabs[current_tab]:
                    st.subheader("Voltage Stability Across Phases")
                    voltage_cols = ['L1 Avg Voltage (V)', 'L2 Avg Voltage (V)', 'L3 Avg Voltage (V)']
                    fig_voltage = px.line(data, x='Datetime', y=voltage_cols, color_discrete_map={voltage_cols[0]: 'red', voltage_cols[1]: 'blue', voltage_cols[2]: 'green'})
                    st.plotly_chart(fig_voltage, use_container_width=True)
                    with st.expander("Show Voltage Statistics"):
                        st.dataframe(data[voltage_cols].describe().T[['mean', 'min', 'max']].rename(columns={'mean':'Average', 'min':'Minimum', 'max':'Maximum'}))
                current_tab += 1

            if "⚖️ Power Factor" in tabs_to_show:
                with tabs[current_tab]:
                    st.subheader("Power Factor Per Phase")
                    pf_cols = ['L1 Power Factor', 'L2 Power Factor', 'L3 Power Factor']
                    fig_pf_3p = px.line(data, x='Datetime', y=pf_cols, color_discrete_map={pf_cols[0]: 'red', pf_cols[1]: 'blue', pf_cols[2]: 'green'})
                    st.plotly_chart(fig_pf_3p, use_container_width=True)
                    with st.expander("Show Power Factor Statistics"):
                        st.dataframe(data[pf_cols].describe().T[['mean', 'min', 'max']].rename(columns={'mean':'Average', 'min':'Minimum', 'max':'Maximum'}))
                current_tab += 1

            with tabs[current_tab]:
                st.subheader("Measurement Settings")
                st.dataframe(parameters)

            with tabs[current_tab+1]:
                st.subheader("Full Time-Series Data")
                st.dataframe(data)

        st.sidebar.markdown("---")
        st.sidebar.subheader("Add Custom AI Context")
        additional_context = st.sidebar.text_area("Provide specific details about the machine or process (optional):")

        if st.sidebar.button("🤖 Get AI-Powered Analysis"):
            with st.spinner("🧠 AI is analyzing the data... This may take a moment."):
                summary_text = "\n".join([f"- {k}: {v}" for k, v in kpi_summary.items() if "N/A" not in str(v)])
                stats_cols = [c for c in data.columns if data[c].dtype in ['float64', 'int64']]
                stats_text = data[stats_cols].describe().to_string() if stats_cols else "No numeric data for statistics."
                params_text = parameters.to_string()
                st.session_state['ai_analysis'] = get_gemini_analysis(summary_text, stats_text, params_text, additional_context)
        if 'ai_analysis' in st.session_state:
            st.markdown("---")
            st.header("🤖 AI-Powered Analysis")
            st.markdown(st.session_state['ai_analysis'])

    elif uploaded_file is not None:
         st.warning("Could not process the uploaded file. Please ensure it is a valid, non-empty Hioki CSV export.")
