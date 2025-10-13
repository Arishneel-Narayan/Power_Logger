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

def process_hioki_csv(uploaded_file) -> Optional[Tuple[str, pd.DataFrame, pd.DataFrame, pd.DataFrame, List[str]]]:
    """Main data processing pipeline for Hioki CSV files."""
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
    data_df['Datetime'] = pd.to_datetime(data_df['Date'] + ' ' + data_df['Etime'], errors='coerce')
    data_df.dropna(subset=['Datetime'], inplace=True)
    data_df.sort_values(by='Datetime', inplace=True, ignore_index=True)
    for col in data_df.columns:
        if any(k in str(col) for k in ['(W)', '(VA)', 'VAR', '(V)', '(A)', 'Factor', 'Energy', '(Hz)', '(kVARh)']):
            data_df[col] = pd.to_numeric(data_df[col], errors='coerce')
    activity_col = next((c for c in ['L1 Avg Current (A)', 'Avg Current (A)'] if c in data_df.columns), None)
    removed_data = pd.DataFrame()
    if activity_col and not data_df[activity_col].dropna().empty:
        is_flat = data_df[activity_col].rolling(window=5, center=True).std(ddof=0).fillna(0) < 1e-4
        if is_flat.any():
            st.sidebar.warning(f"{is_flat.sum()} inactive data points removed.")
            removed_data = data_df[is_flat].copy()
            data_df = data_df[~is_flat].copy()
    if wiring_system == '3P4W':
        p_cols = [f'L{i} Avg Real Power (W)' for i in range(1, 4)]
        if all(c in data_df for c in p_cols) and 'Total Avg Real Power (W)' not in data_df:
            st.sidebar.info("Calculating Total Power from phase data.")
            data_df['Total Avg Real Power (W)'] = data_df[p_cols].abs().sum(axis=1)
            s_cols = [f'L{i} Avg Apparent Power (VA)' for i in range(1, 4)]
            if all(c in data_df for c in s_cols): data_df['Total Avg Apparent Power (VA)'] = data_df[s_cols].sum(axis=1)
            q_cols = [f'L{i} Avg Reactive Power (VAR)' for i in range(1, 4)]
            if all(c in data_df for c in q_cols): data_df['Total Avg Reactive Power (VAR)'] = data_df[q_cols].sum(axis=1)
            if 'Total Avg Real Power (W)' in data_df and 'Total Avg Apparent Power (VA)' in data_df:
                data_df['Total Power Factor'] = data_df.apply(lambda r: r['Total Avg Real Power (W)'] / r['Total Avg Apparent Power (VA)'] if r['Total Avg Apparent Power (VA)'] > 0 else 0, axis=1)
    for col in data_df.columns:
        if '(W)' in col or '(VA)' in col or '(VAR)' in col:
            new_col = col.replace('(W)', '(kW)').replace('(VA)', '(kVA)').replace('(VAR)', '(kVAR)')
            data_df[new_col] = data_df[col] / 1000
    warnings = []
    if any((data_df[c] < -0.01).any() for c in data_df.select_dtypes(include=np.number).columns if 'Real Power' in c or 'Power Factor' in c):
        warnings.append("⚠️ **Potential Polarity Issue Detected:** Negative power/PF values found. This may indicate a reversed CT clamp. KPIs are calculated using absolute values.")
    return wiring_system, params_df, data_df, removed_data, warnings

def process_production_log(uploaded_file) -> Optional[pd.DataFrame]:
    """Reads and cleans the production log file."""
    try:
        prod_df = pd.read_excel(uploaded_file) if uploaded_file.name.endswith(('.xlsx', '.xls')) else pd.read_csv(uploaded_file)
        prod_df.columns = [c.strip().replace(' ', '_') for c in prod_df.columns]
        rename_map = {'Timestamp': 'Start_Time', 'Total_Run_Hours': 'Run_Hours', 'Units_Produced_(tonnes)': 'Production_Output_tonnes'}
        prod_df.rename(columns=rename_map, inplace=True)
        prod_df['Start_Time'] = pd.to_datetime(prod_df['Start_Time'])
        prod_df['End_Time'] = prod_df.apply(lambda r: r['Start_Time'] + pd.to_timedelta(r['Run_Hours'], unit='h'), axis=1)
        return prod_df.sort_values('Start_Time').reset_index(drop=True)
    except Exception as e:
        st.error(f"Error processing production log: {e}")
        return None

def merge_data(power_df, prod_df):
    """Merges power data with production data based on timestamps."""
    merged = pd.merge_asof(power_df.sort_values('Datetime'), prod_df, left_on='Datetime', right_on='Start_Time', direction='backward')
    return merged[merged['Datetime'] <= merged['End_Time']]

def add_production_overlay(fig, prod_df):
    """Adds shaded regions to a Plotly figure to indicate production status."""
    color_map = {'Production': 'rgba(76, 175, 80, 0.2)', 'Idle': 'rgba(255, 193, 7, 0.2)', 'Startup': 'rgba(244, 67, 54, 0.2)', 'CIP': 'rgba(33, 150, 243, 0.2)'}
    for _, row in prod_df.iterrows():
        fig.add_vrect(x0=row['Start_Time'], x1=row['End_Time'], fillcolor=color_map.get(row['Production_Status'], 'rgba(158, 158, 158, 0.2)'), opacity=0.5, layer="below", line_width=0, annotation_text=row['Production_Status'], annotation_position="top left")
    return fig

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
        wiring_system, parameters, data_full, removed_data, warnings = process_result
        st.sidebar.success(f"File processed successfully!\n\n**Mode: {wiring_system} Analysis**")
        prod_data = None
        if production_log_file:
            prod_data = process_production_log(production_log_file)
            if prod_data is not None:
                st.sidebar.info("Production log loaded and merged.")
                data_full = merge_data(data_full, prod_data)
        if warnings:
            for warning in warnings: st.warning(warning)
        data = data_full.copy()
        if not data.empty:
            st.sidebar.markdown("---")
            st.sidebar.subheader("Filter Data by Time")
            dt_options = data['Datetime'].dt.to_pydatetime()
            start_time, end_time = st.sidebar.select_slider("Select a time range for analysis:", options=dt_options, value=(dt_options[0], dt_options[-1]), format_func=lambda dt: dt.strftime("%d %b, %H:%M"))
            data = data[(data['Datetime'] >= start_time) & (data['Datetime'] <= end_time)].copy()
        
        tab_list = []
        if prod_data is not None: tab_list.append("🏭 Production Efficiency")
        if wiring_system == '1P2W': st.header("Single-Phase Performance Analysis"); tab_list.extend(["⚡ Power & Energy", "📈 Key Metrics"])
        elif wiring_system == '3P4W': st.header("Three-Phase System Diagnostic"); tab_list.append("📈 Key Metrics");
        if '3P4W' in wiring_system:
            if all(c in data for c in ['L1 Avg Current (A)', 'L2 Avg Current (A)', 'L3 Avg Current (A)']): tab_list.append("📊 Load Balance")
            if all(c in data for c in ['L1 Avg Voltage (V)', 'L2 Avg Voltage (V)', 'L3 Avg Voltage (V)']): tab_list.append("🩺 Voltage Health")
            if all(c in data for c in ['L1 Power Factor', 'L2 Power Factor', 'L3 Power Factor']): tab_list.append("⚖️ Power Factor")
        tab_list.extend(["📝 Settings", "📋 Filtered Data"]);
        if not removed_data.empty: tab_list.append("🚫 Removed Data")
        tabs = st.tabs(tab_list); tab_map = {name: tab for name, tab in zip(tab_list, tabs)}
        kpi_summary = {}

        # --- TAB CONTENT ---
        if "🏭 Production Efficiency" in tab_map:
            with tab_map["🏭 Production Efficiency"]:
                prod_analysis_df = data[data['Production_Status'] == 'Production'].copy()
                if not prod_analysis_df.empty:
                    prod_analysis_df['Interval_h'] = prod_analysis_df['Datetime'].diff().dt.total_seconds().fillna(0) / 3600
                    power_col = 'Total Avg Real Power (kW)' if 'Total Avg Real Power (kW)' in prod_analysis_df else 'Avg Real Power (kW)'
                    prod_analysis_df['Energy_kWh_Interval'] = prod_analysis_df[power_col] * prod_analysis_df['Interval_h']
                    summary_df = prod_analysis_df.groupby(['Company', 'Line_ID', 'Product_SKU', 'Start_Time']).agg(Total_Energy_kWh=('Energy_kWh_Interval', 'sum'), Production_Output_tonnes=('Production_Output_tonnes', 'first')).reset_index()
                    summary_df['Production_Output_kg'] = summary_df['Production_Output_tonnes'] * 1000
                    summary_df['SEC_kWh_per_kg'] = np.divide(summary_df['Total_Energy_kWh'], summary_df['Production_Output_kg']).replace([np.inf, -np.inf], 0)
                    c1, c2 = st.columns(2)
                    with c1: st.plotly_chart(px.bar(summary_df, x='Product_SKU', y='SEC_kWh_per_kg', color='Company', title='<b>Specific Energy Consumption (SEC)</b>'), use_container_width=True)
                    with c2: st.plotly_chart(px.scatter(summary_df, x='Production_Output_kg', y='Total_Energy_kWh', color='Company', title='<b>Energy vs. Production Output</b>'), use_container_width=True)
                else: st.info("No 'Production' status events found in the selected time range.")
        
        if "⚡ Power & Energy" in tab_map:
            with tab_map["⚡ Power & Energy"]:
                plot_cols = [c for c in ['Avg Real Power (kW)', 'Avg Apparent Power (kVA)', 'Avg Reactive Power (kVAR)'] if c in data]
                if plot_cols:
                    st.subheader("Power Consumption Over Time"); fig = px.line(data, x='Datetime', y=plot_cols)
                    if prod_data is not None and st.checkbox("Overlay Production Status", key="overlay_1p"):
                        fig = add_production_overlay(fig, prod_data[(prod_data['Start_Time'] <= end_time) & (prod_data['End_Time'] >= start_time)])
                    st.plotly_chart(fig, use_container_width=True)

        if "📈 Key Metrics" in tab_map:
            with tab_map["📈 Key Metrics"]:
                st.subheader("Performance Metrics & Cost Estimation")
                # Calculations
                duration_hours, total_kwh, peak_kva, avg_kw, load_factor = 0, 0, 0, 0, 0
                if not data.empty and len(data) > 1:
                    duration_hours = (data['Datetime'].iloc[-1] - data['Datetime'].iloc[0]).total_seconds() / 3600
                    if wiring_system == '1P2W':
                        energy_col, pwr_col, ap_pwr_col, pf_col = 'Consumed Real Energy (Wh)', 'Avg Real Power (kW)', 'Avg Apparent Power (kVA)', 'Power Factor'
                    else: # 3P4W
                        energy_col, pwr_col, ap_pwr_col, pf_col = 'Total Consumed Real Energy (Wh)', 'Total Avg Real Power (kW)', 'Total Avg Apparent Power (kVA)', 'Total Power Factor'
                    
                    if energy_col in data.columns and not data[energy_col].dropna().empty:
                        vals = data[energy_col].dropna(); total_kwh = (vals.iloc[-1] - vals.iloc[0]) / 1000 if len(vals) > 1 else 0
                    peak_kva = data[ap_pwr_col].max() if ap_pwr_col in data else 0
                    avg_kw = data[pwr_col].abs().mean() if pwr_col in data else 0
                    avg_pf = data[pf_col].abs().mean() if pf_col in data else 0
                    peak_kw = data[pwr_col].abs().max() if pwr_col in data else 0
                    load_factor = avg_kw / peak_kw if peak_kw > 0 else 0
                
                # Display Metrics
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Total Consumed Energy", f"{total_kwh:.2f} kWh")
                m2.metric("Peak Demand (MD)", f"{peak_kva:.2f} kVA")
                m3.metric("Average Power Draw", f"{avg_kw:.2f} kW")
                m4.metric("Average Power Factor", f"{avg_pf:.3f}")

                with st.expander("Estimate Operating Cost"):
                    c1, c2 = st.columns(2)
                    cost_kwh = c1.number_input("Cost per kWh ($)", value=0.40, step=0.01, format="%.2f")
                    cost_kva = c2.number_input("Peak Demand Charge per kVA ($)", value=18.00, step=0.50, format="%.2f")
                    cost_energy = total_kwh * cost_kwh; cost_demand = peak_kva * cost_kva
                    st.markdown("---")
                    r1, r2, r3 = st.columns(3)
                    r1.metric("Energy Cost", f"${cost_energy:,.2f}"); r2.metric("Demand Cost", f"${cost_demand:,.2f}"); r3.metric("Total Estimated Cost", f"${cost_energy+cost_demand:,.2f}")
                
                kpi_summary = {"Total Consumed Energy": f"{total_kwh:.2f} kWh", "Peak Demand (MD)": f"{peak_kva:.2f} kVA", "Average Power Draw": f"{avg_kw:.2f} kW", "Average Power Factor": f"{avg_pf:.3f}", "Operating Duration": f"{duration_hours:.2f} hours", "Load Factor": f"{load_factor:.2f}"}

        if "📊 Load Balance" in tab_map:
            with tab_map["📊 Load Balance"]:
                st.subheader("Current Load Balance Across Phases"); current_cols = ['L1 Avg Current (A)', 'L2 Avg Current (A)', 'L3 Avg Current (A)']
                fig = px.line(data, x='Datetime', y=current_cols)
                if prod_data is not None and st.checkbox("Overlay Production Status", key="overlay_3p_current"):
                    fig = add_production_overlay(fig, prod_data[(prod_data['Start_Time'] <= end_time) & (prod_data['End_Time'] >= start_time)])
                st.plotly_chart(fig, use_container_width=True)

        if "🩺 Voltage Health" in tab_map:
            with tab_map["🩺 Voltage Health"]: st.plotly_chart(px.line(data, x='Datetime', y=['L1 Avg Voltage (V)', 'L2 Avg Voltage (V)', 'L3 Avg Voltage (V)'], title='<b>Voltage Stability Across Phases</b>'), use_container_width=True)
        if "⚖️ Power Factor" in tab_map:
            with tab_map["⚖️ Power Factor"]: st.plotly_chart(px.line(data, x='Datetime', y=[c for c in ['L1 Power Factor', 'L2 Power Factor', 'L3 Power Factor', 'Total Power Factor'] if c in data], title='<b>Power Factor Per Phase</b>'), use_container_width=True)
        if "📝 Settings" in tab_map:
            with tab_map["📝 Settings"]: st.dataframe(parameters)
        if "📋 Filtered Data" in tab_map:
            with tab_map["📋 Filtered Data"]: st.dataframe(data)
        if "🚫 Removed Data" in tab_map:
            with tab_map["🚫 Removed Data"]: st.dataframe(removed_data)
            
        st.sidebar.markdown("---"); st.sidebar.subheader("Add Custom AI Context")
        additional_context = st.sidebar.text_area("Provide specific details about the machine or process (e.g., 'This is the dough mixer, running batch #123').", key="ai_context")
        if st.sidebar.button("🤖 Get AI-Powered Analysis"):
            with st.spinner("🧠 AI is analyzing the data... This may take a moment."):
                summary_text = "\n".join([f"- {k}: {v}" for k, v in kpi_summary.items() if "N/A" not in str(v)])
                stats_cols = [c for c in data.columns if data[c].dtype in ['float64', 'int64']]
                stats_text = data[stats_cols].describe().to_string() if stats_cols else "No numeric data for statistics."
                params_text = parameters.to_string()
                st.session_state['ai_analysis'] = get_gemini_analysis(summary_text, stats_text, params_text, additional_context)
        if 'ai_analysis' in st.session_state:
            st.markdown("---"); st.header("🤖 AI-Powered Analysis"); st.markdown(st.session_state['ai_analysis'])

