import streamlit as st
import pandas as pd
import plotly.express as px
from typing import Tuple, Optional
import requests
from fpdf import FPDF
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
        'Pdem+': 'Power Demand'
    }
    
    suffixes = {'_Avg': 'Avg', '_max': 'Max', '_min': 'Min'}
    units = {
        'V': '(V)', 'A': '(A)', 'W': '(W)', 'VA': '(VA)', 'var': '(VAR)',
        'Hz': '(Hz)', 'deg': '(deg)', 'Wh': '(Wh)'
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
        'WP+1[Wh]': 'Consumed Real Energy (Wh)', 'WP+sum[Wh]': 'Total Consumed Real Energy (Wh)'
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

# --- 2. AI and PDF Generation Services ---

def get_gemini_analysis(summary_metrics, data_stats, params_info, additional_context=""):
    # ... (This function is unchanged)
    system_prompt = """You are an expert industrial energy efficiency analyst and process engineer for FMF Foods Ltd., a food manufacturing company in Fiji. Your task is to analyze power consumption data from industrial machinery at our biscuit factory in Suva. Your analysis must be framed within the context of a manufacturing environment. Consider operational cycles, equipment health, and system reliability. Most importantly, link your findings directly to cost-saving opportunities, specifically addressing EFL's two-part tariff structure (Energy Charge + Demand Charge + VAT). Mention how improving power factor reduces kVA demand and how lowering peak demand (MD) directly reduces the monthly demand charge. Provide a concise, actionable report in Markdown format with three sections: 1. Executive Summary, 2. Key Observations & Pattern Analysis, and 3. Actionable Recommendations. Address the user as a fellow process optimization engineer."""
    user_prompt = f"""
    Good morning, Please analyze the following power consumption data for an industrial machine at our Suva facility, keeping in mind our electricity costs are based on EFL's Maximum Demand tariff.

    **Key Performance Indicators (KPIs):**
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

class PDF(FPDF):
    # ... (This class is unchanged)
    def header(self):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, 'FMF Power Consumption Analysis Report', 0, 1, 'C')
        self.set_font('Arial', '', 8)
        self.cell(0, 5, f"Generated on: {pd.Timestamp.now(tz='Pacific/Fiji').strftime('%Y-%m-%d %H:%M:%S')}", 0, 1, 'C')
        self.ln(5)
    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')
    def chapter_title(self, title):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, title, 0, 1, 'L')
        self.ln(4)
    def chapter_body(self, body):
        self.set_font('Arial', '', 10)
        self.multi_cell(0, 5, body)
        self.ln()
    def add_kpis(self, kpi_dict):
        self.set_font('Arial', 'B', 10)
        for key, value in kpi_dict.items():
            self.cell(60, 8, str(key), border=1)
            self.set_font('Arial', '', 10)
            self.cell(0, 8, str(value), border=1)
            self.ln()
        self.ln()
    def add_plot(self, fig, title, insight):
        self.chapter_title(title)
        try:
            img_bytes = fig.to_image(format="png", width=700, height=400, scale=2)
            img = io.BytesIO(img_bytes)
            self.image(img, x=10, w=190)
            self.ln(5)
            self.set_font('Arial', 'B', 10)
            self.cell(0, 5, "Insight:")
            self.ln()
            self.set_font('Arial', 'I', 10)
            self.multi_cell(0, 5, insight)
        except RuntimeError:
            self.set_font('Arial', 'I', 10)
            self.set_text_color(255, 0, 0)
            self.multi_cell(0, 5, "[Chart rendering failed due to a server environment issue. The data is still valid.]")
            self.set_text_color(0, 0, 0)
        self.ln(10)

def create_report_pdf(filename, kpis, ai_analysis_text, figures, context):
    pdf = PDF()
    pdf.add_page()
    pdf.chapter_title(f"Analysis for: {filename}")
    if context:
        pdf.set_font('Arial', 'B', 10)
        pdf.cell(0, 5, "Engineer's Context:")
        pdf.ln()
        pdf.set_font('Arial', 'I', 10)
        pdf.multi_cell(0, 5, context.encode('latin-1', 'replace').decode('latin-1'))
        pdf.ln(5)
    pdf.chapter_title("Key Performance & Cost Indicators")
    pdf.add_kpis(kpis)
    if ai_analysis_text:
        pdf.chapter_title("AI-Powered Analysis")
        pdf.chapter_body(ai_analysis_text.encode('latin-1', 'replace').decode('latin-1'))
    if figures:
        pdf.add_page()
        pdf.chapter_title("Graphical Analysis")
        for fig_title, fig_data in figures.items():
            pdf.add_plot(fig_data['fig'], fig_title, fig_data['insight'])
    return pdf.output().encode('latin-1')


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
        
        data = data_full.copy() # Start with the full dataset

        if not data.empty:
            # --- Universal Pre-calculation for Tariff Suggestion ---
            total_kwh_full = 0
            duration_hours_full = (data['Datetime'].max() - data['Datetime'].min()).total_seconds() / 3600 if not data.empty else 0
            
            if wiring_system == '1P2W':
                energy_col = 'Consumed Real Energy (Wh)'
                if energy_col in data.columns and not data[energy_col].dropna().empty:
                    energy_vals = data[energy_col].dropna()
                    if len(energy_vals) > 1: total_kwh_full = (energy_vals.iloc[-1] - energy_vals.iloc[0]) / 1000
            elif wiring_system == '3P4W':
                avg_power_kw_full = data['Total Avg Real Power (kW)'].mean() if 'Total Avg Real Power (kW)' in data.columns else 0
                if duration_hours_full > 0 and avg_power_kw_full > 0:
                    total_kwh_full = avg_power_kw_full * duration_hours_full

            default_tariff_index = 0
            if duration_hours_full > 1: # Only project if we have a reasonable amount of data
                avg_hourly_kwh = total_kwh_full / duration_hours_full
                projected_monthly_kwh = avg_hourly_kwh * 730 # 24 * 30.4
                if projected_monthly_kwh > 15000:
                    default_tariff_index = 1
            
            # --- Sidebar UI ---
            st.sidebar.markdown("---")
            st.sidebar.subheader("EFL Tariff Structure")
            tariff_option = st.sidebar.radio(
                "Select Monthly Consumption Tier:",
                ('Standard Use (< 15,000 kWh)', 'High Use (> 15,000 kWh)'),
                index=default_tariff_index,
                help="This is automatically suggested based on the data. You can override it to model different scenarios."
            )
            if tariff_option == 'Standard Use (< 15,000 kWh)':
                energy_rate = 0.4099
            else:
                energy_rate = 0.4295
            st.sidebar.caption(f"Applied Rate: FJD {energy_rate:.4f}/kWh (VAT Exclusive)")

            demand_rate = st.sidebar.number_input("Demand Rate (FJD/kVA/month)", min_value=0.00, value=25.41, step=0.01, help="EFL's monthly charge applied to your peak Apparent Power (MD in kVA).")
            vat_rate = st.sidebar.number_input("VAT Rate (%)", min_value=0.0, value=15.0, step=0.5, help="Value Added Tax rate.")

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
        figures_to_plot = {}
        
        if wiring_system == '1P2W':
            # ... Calculations and UI ...
        elif wiring_system == '3P4W':
            # ... Calculations and UI ...

        # --- AI and PDF Section ---
        st.sidebar.markdown("---")
        st.sidebar.subheader("Add Custom AI Context")
        additional_context = st.sidebar.text_area("Provide specific details about the machine or process (optional):")
        
        if st.sidebar.button("🤖 Get AI-Powered Analysis"):
            # ... AI logic ...

        st.sidebar.markdown("---")
        st.sidebar.subheader("Download Report")
        
        # ... PDF logic ...
        
        if 'ai_analysis' in st.session_state:
            st.markdown("---")
            st.header("🤖 AI-Powered Analysis")
            st.markdown(st.session_state['ai_analysis'])

    elif uploaded_file is not None:
         st.warning("Could not process the uploaded file. Please ensure it is a valid, non-empty Hioki CSV export.")

