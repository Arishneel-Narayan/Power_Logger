import streamlit as st
import pandas as pd
import plotly.express as px
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
        if any(keyword in str(col) for keyword in ['(W)', '(VA)', 'VAR', '(V)', '(A)', 'Factor', 'Energy', '(Hz)', '(kVARh)']):
            data_df[col] = pd.to_numeric(data_df[col], errors='coerce')
    
    activity_col = 'L1 Avg Current (A)'
    removed_data = pd.DataFrame()
    if activity_col in data_df.columns and not data_df[activity_col].dropna().empty:
        is_flat = data_df[activity_col].rolling(window=5, center=True).std(ddof=0).fillna(0) < 1e-4
        active_data = data_df[~is_flat].copy()
        removed_data = data_df[is_flat].copy()

        if not removed_data.empty:
            st.sidebar.warning(f"{len(removed_data)} inactive data points were removed from main analysis.")
            data_df = active_data

    if wiring_system == '3P4W':
        power_cols = ['L1 Avg Real Power (W)', 'L2 Avg Real Power (W)', 'L3 Avg Real Power (W)']
        if all(c in data_df.columns for c in power_cols) and 'Total Avg Real Power (W)' not in data_df.columns:
            st.sidebar.info("Calculating Total Power from phase data.")
            data_df['Total Avg Real Power (W)'] = data_df[power_cols].sum(axis=1)
            
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

    return wiring_system, params_df, data_df, removed_data

# --- 2. AI and Cost Calculation Services ---

def get_gemini_analysis(summary_metrics, data_stats, params_info, additional_context=""):
    system_prompt = """You are an expert industrial energy efficiency analyst and process engineer for FMF Foods Ltd., a food manufacturing company in Fiji. Your task is to analyze power consumption data from industrial machinery at our biscuit factory in Suva. Your analysis must be framed within the context of a manufacturing environment. Consider operational cycles, equipment health, and system reliability. Most importantly, link your findings directly to cost-saving opportunities, specifically addressing EFL's complex tariff structure (tiered energy, demand charges, excess reactive power, and VAT). Mention how improving power factor reduces kVARh and how lowering peak demand (MD) directly reduces the monthly demand charge. Provide a concise, actionable report in Markdown format with three sections: 1. Executive Summary, 2. Key Observations & Pattern Analysis, and 3. Actionable Recommendations. Address the user as a fellow process optimization engineer."""
    user_prompt = f"""
    Good morning, Please analyze the following power consumption data for an industrial machine at our Suva facility, keeping in mind our electricity costs are based on EFL's complex Maximum Demand tariff structure.

    **Key Financial & Performance Indicators:**
    {summary_metrics}
    **Measurement Parameters:**
    {params_info}
    **Statistical Summary of Time-Series Data:**
    {data_stats}
    """
    if additional_context:
        user_prompt += f"**Additional Engineer's Context:**\n{additional_context}"
    user_prompt += "\nBased on all this information, please generate a report with your insights and recommendations for process optimization."
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

def calculate_efl_bill(band: str, total_kwh: float, peak_kw: float, total_kvarh: float, duration_hours: float) -> Dict:
    """Calculates an estimated EFL bill based on the selected demand band."""
    
    VAT_RATE = 0.125
    REACTIVE_RATE = 0.4295

    energy_charge, demand_charge, reactive_charge = 0, 0, 0

    # --- Energy Charge Calculation ---
    if band == "Small Business (<75kW)":
        projected_monthly_kwh = (total_kwh / duration_hours) * 730 if duration_hours > 0 else 0
        if projected_monthly_kwh <= 14999:
            energy_charge = total_kwh * 0.4099
        else:
            # For simplicity in this short-term analysis, we apply the higher rate to all consumption
            # as it's likely representative of the monthly pattern.
            energy_charge = total_kwh * 0.4295
    elif band == "Medium (75-500kW)":
        energy_charge = total_kwh * 0.2781
        demand_charge = peak_kw * 35.33
    elif band == "Large (500-1000kW)":
        energy_charge = total_kwh * 0.3026
        demand_charge = peak_kw * 37.57
    elif band == "Extra Large (>1000kW)":
        energy_charge = total_kwh * 0.3270
        demand_charge = peak_kw * 39.24

    # --- Reactive Charge Calculation ---
    allowed_kvarh = 0.62 * total_kwh
    excess_kvarh = max(0, total_kvarh - allowed_kvarh)
    reactive_charge = excess_kvarh * REACTIVE_RATE

    # --- Final Bill Calculation ---
    subtotal = energy_charge + demand_charge + reactive_charge
    vat_amount = subtotal * VAT_RATE
    grand_total = subtotal + vat_amount

    return {
        "Energy Charge": energy_charge, "Demand Charge": demand_charge,
        "Reactive Charge": reactive_charge, "Subtotal": subtotal,
        "VAT Amount": vat_amount, "Grand Total": grand_total
    }


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
        wiring_system, parameters, data_full, removed_data = process_result
        st.sidebar.success(f"File processed successfully!\n\n**Mode: {wiring_system} Analysis**")
        
        data = data_full.copy()

        if not data.empty:
            # --- Sidebar UI ---
            st.sidebar.markdown("---")
            st.sidebar.subheader("EFL Tariff Structure")
            
            # --- Auto-detect Demand Band ---
            peak_kw_for_band_detection = 0
            if wiring_system == '1P2W' and 'Avg Real Power (kW)' in data.columns:
                peak_kw_for_band_detection = data['Avg Real Power (kW)'].max()
            elif wiring_system == '3P4W' and 'Total Avg Real Power (kW)' in data.columns:
                peak_kw_for_band_detection = data['Total Avg Real Power (kW)'].max()
            
            demand_bands = ["Small Business (<75kW)", "Medium (75-500kW)", "Large (500-1000kW)", "Extra Large (>1000kW)"]
            default_band_index = 0
            if 75 <= peak_kw_for_band_detection < 500:
                default_band_index = 1
            elif 500 <= peak_kw_for_band_detection < 1000:
                default_band_index = 2
            elif peak_kw_for_band_detection >= 1000:
                default_band_index = 3

            demand_band = st.sidebar.selectbox(
                "Select Demand Band:", options=demand_bands, index=default_band_index,
                help="Automatically suggested based on the peak kW in your data. You can override it to model different scenarios."
            )
            st.sidebar.caption(f"VAT is fixed at 12.5%")

            # --- Date/Time Filter ---
            st.sidebar.markdown("---")
            st.sidebar.subheader("Filter Data by Time")
            dt_options = data['Datetime'].dt.to_pydatetime()
            start_time, end_time = st.sidebar.select_slider(
                "Select a time range for analysis:",
                options=dt_options,
                value=(dt_options[0], dt_options[-1]),
                format_func=lambda dt: dt.strftime("%d %b, %H:%M")
            )
            data = data[(data['Datetime'] >= start_time) & (data['Datetime'] <= end_time)].copy()
        
        kpi_summary = {}
        
        if wiring_system == '1P2W':
            st.header("Single-Phase Performance Analysis")
            
            total_kwh, peak_kw, total_kvarh = 0, 0, 0
            duration_hours = (data['Datetime'].max() - data['Datetime'].min()).total_seconds() / 3600 if not data.empty else 0
            
            if 'Consumed Real Energy (Wh)' in data.columns and not data['Consumed Real Energy (Wh)'].dropna().empty:
                energy_vals = data['Consumed Real Energy (Wh)'].dropna()
                if len(energy_vals) > 1: total_kwh = (energy_vals.iloc[-1] - energy_vals.iloc[0]) / 1000
            
            if 'Avg Real Power (kW)' in data.columns: peak_kw = data['Avg Real Power (kW)'].max()
            
            if 'Lagging Reactive Energy (kVARh)' in data.columns:
                kvarh_vals = data['Lagging Reactive Energy (kVARh)'].dropna()
                if len(kvarh_vals) > 1: total_kvarh = (kvarh_vals.iloc[-1] - kvarh_vals.iloc[0])

            bill_details = calculate_efl_bill(demand_band, total_kwh, peak_kw, total_kvarh, duration_hours)

            st.subheader("EFL Bill Estimate")
            cost_help_text = f"For the {duration_hours:.1f} hour measurement period."
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Energy Charge", f"FJD {bill_details['Energy Charge']:.2f}", help=cost_help_text)
            c2.metric("Demand Charge", f"FJD {bill_details['Demand Charge']:.2f}", help="Monthly charge based on this period's peak.")
            c3.metric("Reactive Charge", f"FJD {bill_details['Reactive Charge']:.2f}", help=cost_help_text)
            c4.metric("Subtotal", f"FJD {bill_details['Subtotal']:.2f}", help=cost_help_text)
            c5.metric("Grand Total (inc. VAT)", f"FJD {bill_details['Grand Total']:.2f}", help=cost_help_text)
            
            kpi_summary = bill_details
            kpi_summary['Peak kW'] = f"{peak_kw:.2f}"
            
        elif wiring_system == '3P4W':
            st.header("Three-Phase System Diagnostic")
            
            avg_power_kw = data['Total Avg Real Power (kW)'].mean() if 'Total Avg Real Power (kW)' in data.columns else 0
            peak_kw = data['Total Avg Real Power (kW)'].max() if 'Total Avg Real Power (kW)' in data.columns else 0
            duration_hours = (data['Datetime'].max() - data['Datetime'].min()).total_seconds() / 3600 if not data.empty else 0
            total_kwh = avg_power_kw * duration_hours if duration_hours > 0 else 0
            
            avg_kvar = data['Total Avg Reactive Power (kVAR)'].mean() if 'Total Avg Reactive Power (kVAR)' in data.columns else 0
            total_kvarh = avg_kvar * duration_hours if duration_hours > 0 else 0

            bill_details = calculate_efl_bill(demand_band, total_kwh, peak_kw, total_kvarh, duration_hours)
            
            st.subheader("EFL Bill Estimate")
            cost_help_text_3p = f"For the {duration_hours:.1f} hour measurement period."
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Energy Charge", f"FJD {bill_details['Energy Charge']:.2f}", help=cost_help_text_3p)
            c2.metric("Demand Charge", f"FJD {bill_details['Demand Charge']:.2f}", help="Monthly charge based on this period's peak.")
            c3.metric("Reactive Charge", f"FJD {bill_details['Reactive Charge']:.2f}", help=cost_help_text_3p)
            c4.metric("Subtotal", f"FJD {bill_details['Subtotal']:.2f}", help=cost_help_text_3p)
            c5.metric("Grand Total (inc. VAT)", f"FJD {bill_details['Grand Total']:.2f}", help=cost_help_text_3p)

            kpi_summary = bill_details
            kpi_summary['Peak kW'] = f"{peak_kw:.2f}"
            
        # --- Common UI elements (AI, etc.) ---
        st.sidebar.markdown("---")
        st.sidebar.subheader("Add Custom AI Context")
        additional_context = st.sidebar.text_area("Provide specific details about the machine or process (optional):")

        if st.sidebar.button("🤖 Get AI-Powered Analysis"):
            with st.spinner("🧠 AI is analyzing the data... This may take a moment."):
                summary_metrics_text = "\n".join([f"- {key}: {value}" for key, value in kpi_summary.items()])
                stats_cols = [col for col in data.columns if data[col].dtype in ['float64', 'int64']]
                data_stats_text = data[stats_cols].describe().to_string() if stats_cols else "No numeric data for statistics."
                params_info_text = parameters.to_string()
                ai_response = get_gemini_analysis(summary_metrics_text, data_stats_text, params_info_text, additional_context)
                st.session_state['ai_analysis'] = ai_response

        if 'ai_analysis' in st.session_state:
            st.markdown("---")
            st.header("🤖 AI-Powered Analysis")
            st.markdown(st.session_state['ai_analysis'])

    elif uploaded_file is not None:
         st.warning("Could not process the uploaded file. Please ensure it is a valid, non-empty Hioki CSV export.")

