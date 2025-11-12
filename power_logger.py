import streamlit as st
import pandas as pd
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from typing import Tuple, Optional, Dict
import requests
import io
from fpdf import FPDF
import plotly.io as pio

# --- 1. Core Data Processing Engine ---

# Absolute power threshold for "operational" PF calculation (1.0 kW)
POWER_THRESHOLD_KW = 1.0

def get_rename_map(wiring_system: str) -> dict:
    """
    Dynamically generates a comprehensive rename map for all relevant columns.
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
    elif '3P3W' in wiring_system:
        phases = [('1', 'L1 '), ('3', 'L3 '), ('sum', 'Total ')]

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
    Main data processing pipeline. Features robust date parsing and 
    forced recalculation of Total Power Factor to fix logger magnitude errors.
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
    
    # --- ROBUST DATE PARSING ---
    # Force YYYY-MM-DD HH:MM:SS format to prevent "Oct 10" being read as "Dec 10"
    try:
        data_df['Datetime'] = pd.to_datetime(data_df['Date'], errors='coerce', format="%Y-%m-%d %H:%M:%S")
    except ValueError:
        data_df['Datetime'] = pd.to_datetime(data_df['Date'], errors='coerce')
    
    if data_df['Datetime'].isnull().all():
        st.error("Error: Could not parse any timestamps. The data cannot be processed.")
        return None
        
    data_df.dropna(subset=['Datetime'], inplace=True)

    # Filter out unnecessary columns for clean IDs
    identifier_cols_to_check = ['Datetime', 'Date', 'Etime', 'Machine Status']
    existing_identifiers = [col for col in identifier_cols_to_check if col in data_df.columns]
    measurement_cols = data_df.columns.drop(existing_identifiers, errors='ignore')
    data_df.dropna(subset=measurement_cols, how='all', inplace=True)
    
    if data_df.empty:
        st.error("No valid data rows with measurements were found.")
        return None
    
    data_df = data_df.sort_values(by='Datetime').reset_index(drop=True)

    # Convert to numeric
    for col in data_df.columns:
        if any(keyword in str(col) for keyword in ['(W)', '(VA)', 'VAR', '(V)', '(A)', 'Factor', 'Energy', '(Hz)', '(kVARh)']):
            data_df[col] = pd.to_numeric(data_df[col], errors='coerce')

    # --- ABSOLUTE VALUE FIX ---
    # Correct for reversed CT clamps (negative PF)
    for col in data_df.columns:
        if 'Power Factor' in col or 'Power' in col:
            data_df[col] = data_df[col].abs()
    
    # --- 3P4W CALCULATIONS & TOTALS ---
    if wiring_system == '3P4W':
        power_cols = ['L1 Avg Real Power (W)', 'L2 Avg Real Power (W)', 'L3 Avg Real Power (W)']
        if all(c in data_df.columns for c in power_cols) and 'Total Avg Real Power (W)' not in data_df.columns:
            data_df['Total Avg Real Power (W)'] = data_df[power_cols].sum(axis=1)
            
        apparent_cols = ['L1 Avg Apparent Power (VA)', 'L2 Avg Apparent Power (VA)', 'L3 Avg Apparent Power (VA)']
        if all(c in data_df.columns for c in apparent_cols) and 'Total Avg Apparent Power (VA)' not in data_df.columns:
            data_df['Total Avg Apparent Power (VA)'] = data_df[apparent_cols].sum(axis=1)

    # --- CRITICAL FIX: RECALCULATE TOTAL POWER FACTOR ---
    # Ignore the logger's PF_Avg column if possible, as it often contains magnitude errors.
    # Recalculate using: Abs(Real) / Abs(Apparent)
    if 'Total Avg Real Power (W)' in data_df.columns and 'Total Avg Apparent Power (VA)' in data_df.columns:
        data_df['Total Power Factor'] = data_df.apply(
            lambda row: row['Total Avg Real Power (W)'] / row['Total Avg Apparent Power (VA)'] if row['Total Avg Apparent Power (VA)'] > 0 else 0,
            axis=1
        )
    
    # Convert all units to k-units (kW, kVA, kVAR)
    for col_name in list(data_df.columns):
        if '(W)' in col_name or '(VA)' in col_name or '(VAR)' in col_name:
            new_col_name = col_name.replace('(W)', '(kW)').replace('(VA)', '(kVA)').replace('(VAR)', '(kVAR)')
            if new_col_name not in data_df.columns:
                data_df[new_col_name] = data_df[col_name] / 1000

    return wiring_system, params_df, data_df

# --- 2. AI Service & Summaries ---

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
        f"Period: {start_time} to {end_time}. "
        f"Start Load: {initial_power:.2f} kW. End Load: {final_power:.2f} kW. "
        f"Peak Load: {peak_power:.2f} kW at {peak_time}. "
        f"Min Load: {min_power:.2f} kW."
    )
    return summary

def generate_transform_summary(file_name: str, data_raw: pd.DataFrame, data_clean: pd.DataFrame) -> str:
    """Generates a log of all data transformations."""
    summary_lines = [f"- File: `{file_name}` (Loaded {len(data_raw)} raw rows)."]
    dropped = len(data_raw) - len(data_clean)
    if dropped > 0:
        summary_lines.append(f"- Filtered {dropped} error/status rows. Analysis is based on {len(data_clean)} clean rows.")
    else:
        summary_lines.append(f"- No error/status rows found. Analysis is based on all {len(data_clean)} rows.")
    summary_lines.append("- Applied `.abs()` to all Power and PF columns to correct for potential reversed CT clamps.")
    summary_lines.append("- **Critically: Overwrote logger's 'Total Power Factor'. Recalculated as `Abs(Total Real Power) / Abs(Total Apparent Power)` to fix magnitude errors.**")
    summary_lines.append("- Converted all power units to kW, kVA, and kVAR.")
    return "\n".join(summary_lines)

def generate_detailed_analysis_text(data: pd.DataFrame, wiring_system: str) -> str:
    """Generates structured stats & trends for key columns."""
    if data.empty: return "No data available."
    summary_lines = []
    
    # Define columns of interest
    cols_of_interest = []
    power_col = ''
    
    if wiring_system == '3P4W':
        cols_of_interest = [
            'Total Avg Real Power (kW)', 'Total Avg Apparent Power (kVA)', 'Total Power Factor',
            'L1 Avg Current (A)', 'L2 Avg Current (A)', 'L3 Avg Current (A)',
            'L1 Avg Voltage (V)', 'L2 Avg Voltage (V)', 'L3 Avg Voltage (V)'
        ]
        power_col = 'Total Avg Real Power (kW)'
    elif wiring_system == '1P2W':
        cols_of_interest = [
            'Avg Real Power (kW)', 'Avg Apparent Power (kVA)', 'Power Factor',
            'Avg Current (A)', 'Avg Voltage (V)'
        ]
        power_col = 'Avg Real Power (kW)'

    for col in cols_of_interest:
        if col in data.columns and not data[col].dropna().empty:
            try:
                col_series = data[col].dropna()
                
                # Filter PF stats based on power threshold to ignore idle noise
                if 'Power Factor' in col and power_col in data.columns and not data[data[power_col] > POWER_THRESHOLD_KW].empty:
                    col_series = data[data[power_col] > POWER_THRESHOLD_KW][col].dropna()
                
                if col_series.empty: continue

                stats = col_series.describe()
                trend = col_series.iloc[-1] - col_series.iloc[0]
                line = (
                    f"**{col}:** "
                    f"Mean={stats['mean']:.2f}, "
                    f"Min={stats['min']:.2f}, "
                    f"Max={stats['max']:.2f}, "
                    f"StdDev={stats['std']:.2f}, "
                    f"**Trend={trend:+.2f}** (Start={col_series.iloc[0]:.2f}, End={col_series.iloc[-1]:.2f})"
                )
                summary_lines.append(f"- {line}")
            except Exception:
                pass # Skip column if stats fail
                
    if not summary_lines:
        return "No detailed statistics could be generated for key metrics."
        
    return "\n".join(summary_lines)

def get_gemini_analysis(summary_metrics: str, 
                        detailed_stats_and_trends: str, 
                        trend_summary: str, 
                        params_info: str, 
                        transform_log: str, 
                        additional_context: str = "") -> str:
    """Contacts the Gemini API for an expert analysis."""
    
    system_prompt = """You are an expert industrial energy efficiency analyst and process engineer for FMF Foods Ltd., a food manufacturing company in Fiji. Your task is to analyze power consumption data from industrial machinery at our biscuit factory in Suva. Your analysis must be framed within the context of a manufacturing environment.
    Your analysis MUST be based SOLELY on the 'CLEANED (Status=0) DATA'.
    
    Core Principles:
    - **Operational Cycles:** Use the 'Summary of Trends & Fluctuations' to understand the *main* power sequence.
    - **Pattern Analysis:** Use the 'Detailed Stats & Trends' section to analyze the specific behavior of *each* key metric (Voltage, Current, PF). Look for correlations, e.g., "Current on L1 increased, causing a voltage dip on L1."
    - **Equipment Health:** Interpret electrical data as indicators of mechanical health. Use the 'Detailed Stats & Trends' to understand the magnitude (Mean), volatility (StdDev), and load profiles (Trend).
    - **Cost Reduction:** Link your findings directly to cost-saving opportunities by focusing on reducing peak demand (MD) and improving power factor.
    - **Quantitative Significance:** When analyzing metrics, you MUST refer to the absolute values in the 'Detailed Stats & Trends' to determine the real-world impact.
    - **CRITICAL ENGINEERING FEEDBACK:** You MUST de-prioritize current imbalance recommendations if the *absolute* difference between phase currents (in Amps) is minor (e.g., less than 50A), even if the *percentage* seems high. A 20-30 Amp difference is not significant enough to warrant a 'tedious current audit'. Focus on Peak Demand (MD) and Power Factor as the primary cost-saving drivers.
    
    Provide a concise, actionable report in Markdown format with three sections: 1. Executive Summary, 2. Key Observations & Pattern Analysis, and 3. Actionable Recommendations. Address the user as a fellow process optimization engineer."""
    
    user_prompt = f"""
    Good morning, Please analyze the following power consumption data for an industrial machine at our Suva facility.
    
    **Data Transformation Log (CRITICAL CONTEXT):**
    {transform_log}
    
    **Key Performance Indicators (Selected Period):**
    {summary_metrics}
    
    **Summary of Trends & Fluctuations (Selected Period):**
    {trend_summary}
    
    **Detailed Stats & Trends (Selected Period):**
    {detailed_stats_and_trends}
    
    **Measurement Parameters:**
    {params_info}
    """
    if additional_context:
        user_prompt += f"\n**Additional Engineer's Context:**\n{additional_context}"
    user_prompt += "\nBased on all this information, please generate a report with your insights and recommendations for process optimization."
    
    try:
        api_key = st.secrets["GEMINI_API_KEY"]
    except (KeyError, FileNotFoundError):
        return "Error: Gemini API key not found. Please add it to your Streamlit Secrets."
    
    api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-pro:generateContent?key={api_key}"
    
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

# --- 4. PDF Report Generation Functions ---

# This helper class creates a basic FPDF structure
class PDF(FPDF):
    def header(self):
        self.set_font('Helvetica', 'B', 12)
        self.cell(0, 10, 'FMF Power Consumption Analysis Report', 0, 1, 'C')
        self.set_font('Helvetica', '', 8)
        self.cell(0, 5, f'Generated: {pd.Timestamp.now(tz="Pacific/Fiji").strftime("%a %d %b %Y, %H:%M")}', 0, 1, 'C')
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font('Helvetica', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

    def chapter_title(self, title):
        self.set_font('Helvetica', 'B', 14)
        self.cell(0, 10, title, 0, 1, 'L')
        self.ln(2)

    def chapter_body(self, body):
        self.set_font('Helvetica', '', 10)
        self.multi_cell(0, 5, body)
        self.ln()

    def add_kpi_table(self, kpi_dict):
        self.set_font('Helvetica', 'B', 10)
        self.cell(60, 8, 'Metric', 1)
        self.cell(0, 8, 'Value', 1)
        self.ln()
        self.set_font('Helvetica', '', 10)
        for key, value in kpi_dict.items():
            self.cell(60, 8, str(key), 1)
            self.cell(0, 8, str(value), 1)
            self.ln()

    def add_plotly_chart(self, fig, title):
        self.add_page(orientation='L') # Landscape for graphs
        self.chapter_title(title)
        try:
            # Export graph to in-memory image
            img_bytes = pio.to_image(fig, format="png", width=1000, height=500)
            img_stream = io.BytesIO(img_bytes)
            # A4 Landscape width is 297mm. Margins 10mm each side = 277mm.
            self.image(img_stream, w=277) 
        except Exception as e:
            self.set_text_color(255, 0, 0) # Red
            self.cell(0, 10, f"Error generating graph '{title}': {e}")
            self.set_text_color(0, 0, 0)
        self.ln(5)

# --- NEW: Refactored KPI Generation ---
def generate_kpis(data: pd.DataFrame, wiring_system: str) -> dict:
    """Calculates the main KPI dictionary from a given dataframe."""
    kpi_summary = {}
    if data.empty:
        return {"Error": "No data"}
    
    # Helper to calculate Average PF safely, using the absolute power threshold
    def calc_avg_pf(df, power_col, pf_col):
        if power_col in df.columns and pf_col in df.columns:
            # Filter for data rows where power is above the 1.0 kW threshold
            operational_data = df[df[power_col] > POWER_THRESHOLD_KW]
            if not operational_data.empty:
                operational_pf = operational_data[pf_col].dropna()
                return operational_pf.mean() if not operational_pf.empty else 0
        return 0

    if wiring_system == '1P2W':
        total_kwh = 0
        energy_col = 'Consumed Real Energy (Wh)'
        if energy_col in data.columns and not data[energy_col].dropna().empty:
            energy_vals = data[energy_col].dropna()
            if len(energy_vals) > 1: total_kwh = (energy_vals.iloc[-1] - energy_vals.iloc[0]) / 1000
        
        peak_kva = data['Avg Apparent Power (kVA)'].max() if 'Avg Apparent Power (kVA)' in data.columns else 0
        avg_kw = data['Avg Real Power (kW)'].abs().mean() if 'Avg Real Power (kW)' in data.columns else 0
        
        avg_pf = calc_avg_pf(data, 'Avg Real Power (kW)', 'Power Factor')
        
        kpi_summary = { 
            "Analysis Mode": "Single-Phase", "Total Consumed Energy": f"{total_kwh:.2f} kWh",
            "Peak Demand (MD)": f"{peak_kva:.2f} kVA", "Average Power Draw": f"{avg_kw:.2f} kW",
            "Avg. Total PF": f"{avg_pf:.3f}"
        }
        
    elif wiring_system == '3P4W':
        avg_power_kw = data['Total Avg Real Power (kW)'].mean() if 'Total Avg Real Power (kW)' in data.columns else 0
        
        avg_pf = calc_avg_pf(data, 'Total Avg Real Power (kW)', 'Total Power Factor')

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

        kpi_summary = { 
            "Analysis Mode": "Three-Phase", "Avg. Total Power": f"{avg_power_kw:.2f} kW",
            "Peak Demand (MD)": f"{peak_kva_3p:.2f} kVA", "Avg. Total PF": f"{avg_pf:.3f}",
            "Max Current Imbalance": f"{imbalance:.1f} %"
        }
    return kpi_summary

# --- NEW: Main PDF Generation Function ---
def generate_pdf_report(
    file_name: str,
    parameters: pd.DataFrame,
    wiring_system: str,
    kpi_summary_selected: dict,
    kpi_summary_full: dict,
    ai_analysis: str,
    data_full: pd.DataFrame # Use full, clean data for graphs
) -> bytes:
    """Generates the full PDF report."""
    
    # Handle potential Kaleido issue in headless environments
    try:
        pio.kaleido.scope.mathjax = None
    except AttributeError:
        pass # In case kaleido is not fully initialized

    pdf = PDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    
    # --- Page 1: Title ---
    pdf.add_page()
    pdf.set_font('Helvetica', 'B', 24)
    pdf.cell(0, 20, 'FMF Power Consumption Analysis', 0, 1, 'C')
    pdf.set_font('Helvetica', 'B', 16)
    pdf.cell(0, 15, f'File: {file_name}', 0, 1, 'C')
    pdf.ln(10)
    pdf.set_font('Helvetica', '', 12)
    start_date = data_full['Datetime'].min().strftime('%a %d %b %Y')
    end_date = data_full['Datetime'].max().strftime('%a %d %b %Y')
    pdf.cell(0, 10, f'Full Data Range: {start_date} to {end_date}', 0, 1, 'C')

    # --- Page 2: AI Analysis ---
    pdf.add_page()
    pdf.chapter_title('1. AI-Powered Analysis')
    pdf.chapter_body(ai_analysis)

    # --- Page 3: Metrics ---
    pdf.add_page()
    pdf.chapter_title('2. Key Performance Indicators (KPIs)')
    
    pdf.set_font('Helvetica', 'B', 12)
    pdf.cell(0, 10, 'Metrics for Selected Period', 0, 1, 'L')
    pdf.add_kpi_table(kpi_summary_selected)
    pdf.ln(10)
    
    pdf.set_font('Helvetica', 'B', 12)
    pdf.cell(0, 10, 'Metrics for Full Period (All Status=0 Data)', 0, 1, 'L')
    pdf.add_kpi_table(kpi_summary_full)
    
    # --- Page 4+: Graphs (from FULL data) ---
    pdf.chapter_title('3. Full Period Graphs')
    
    # Re-create graphs using the full, clean dataset (data_full)
    if wiring_system == '1P2W':
        plot_cols = [col for col in ['Avg Real Power (kW)', 'Avg Apparent Power (kVA)', 'Avg Reactive Power (kVAR)'] if col in data_full.columns]
        if plot_cols:
            fig_power = px.line(data_full, x='Datetime', y=plot_cols)
            fig_power.update_layout(xaxis_title="Date & Time", yaxis_title="Power (kW, kVA, kVAR)", xaxis_tickformat="%a %d %b\n%H:%M")
            pdf.add_plotly_chart(fig_power, "Power Consumption Over Time (Full Period)")

        if 'Power Factor' in data.columns:
            fig_pf = px.line(data_full, x='Datetime', y='Power Factor')
            fig_pf.add_hline(y=0.95, line_dash="dash", line_color="red")
            fig_pf.update_layout(xaxis_title="Date & Time", yaxis_title="Power Factor", xaxis_tickformat="%a %d %b\n%H:%M")
            pdf.add_plotly_chart(fig_pf, "Power Factor Over Time (Full Period)")
    
    elif wiring_system == '3P4W':
        # Graph 1: Total System Power
        total_power_cols = [c for c in ['Total Avg Real Power (kW)', 'Total Avg Apparent Power (kVA)', 'Total Avg Reactive Power (kVAR)'] if c in data_full.columns]
        if total_power_cols:
            fig_power_total = px.line(data_full, x='Datetime', y=total_power_cols)
            fig_power_total.update_layout(xaxis_title="Date & Time", yaxis_title="Power (kW, kVA, kVAR)", xaxis_tickformat="%a %d %b\n%H:%M", title="Total System Power (Full Period)")
            pdf.add_plotly_chart(fig_power_total, "Total System Power (Full Period)")

        # Graph 2: Current Envelope
        current_cols_all = [f'{p} {s} Current (A)' for p in ['L1', 'L2', 'L3'] for s in ['Min', 'Avg', 'Max']]
        plot_cols_current = [c for c in current_cols_all if c in data_full.columns]
        if plot_cols_current:
            fig_current = px.line(data_full, x='Datetime', y=plot_cols_current)
            fig_current.update_layout(xaxis_title="Date & Time", yaxis_title="Current (A)", xaxis_tickformat="%a %d %b\n%H:%M", title="Current Operational Envelope (Full Period)")
            pdf.add_plotly_chart(fig_current, "Current Operational Envelope (Full Period)")

        # Graph 3: Voltage Envelope
        voltage_cols_all = [f'{p} {s} Voltage (V)' for p in ['L1', 'L2', 'L3'] for s in ['Min', 'Avg', 'Max']]
        plot_cols_voltage = [c for c in voltage_cols_all if c in data_full.columns]
        if plot_cols_voltage:
            fig_voltage = px.line(data_full, x='Datetime', y=plot_cols_voltage)
            fig_voltage.update_layout(xaxis_title="Date & Time", yaxis_title="Voltage (V)", xaxis_tickformat="%a %d %b\n%H:%M", title="Voltage Operational Envelope (Full Period)")
            pdf.add_plotly_chart(fig_voltage, "Voltage Operational Envelope (Full Period)")
            
        # Graph 4: Power Factor per Phase
        pf_cols = [c for c in ['L1 Power Factor', 'L2 Power Factor', 'L3 Power Factor', 'Total Power Factor'] if c in data_full.columns]
        if pf_cols:
            fig_pf_phase = px.line(data_full, x='Datetime', y=pf_cols)
            fig_pf_phase.update_layout(xaxis_title="Date & Time", yaxis_title="Power Factor", xaxis_tickformat="%a %d %b\n%H:%M", title="Power Factor per Phase (Full Period)")
            pdf.add_plotly_chart(fig_pf_phase, "Power Factor per Phase (Full Period)")
    
    # --- Last Page: Settings ---
    pdf.add_page(orientation='P')
    pdf.chapter_title('4. Measurement Settings')
    pdf.chapter_body(parameters.to_string())

    # Return as bytes
    return pdf.output(dest='S').encode('latin-1')


# --- 5. Streamlit UI and Analysis Section ---
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
    # --- SESSION STATE FIX: Clear state on new file upload ---
    if 'current_file_name' not in st.session_state or st.session_state.current_file_name != uploaded_file.name:
        st.session_state.clear()
        st.session_state.current_file_name = uploaded_file.name
    # --- END FIX ---
    
    # Use the cached function to load data
    process_result = load_hioki_data(uploaded_file)

    if process_result:
        wiring_system, parameters, data_raw = process_result
        
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
        
        # --- KPI Generation ---
        kpi_summary = generate_kpis(data, wiring_system)
        
        if wiring_system == '1P2W':
            st.header("Single-Phase Performance Analysis")
            
            # --- KPI TYPE ERROR FIX ---
            # Get values from dict, convert to float for comparison
            total_kwh_str = kpi_summary.get("Total Consumed Energy", "N/A").split(" ")[0]
            peak_kva_str = kpi_summary.get("Peak Demand (MD)", "N/A").split(" ")[0]
            avg_kw_str = kpi_summary.get("Average Power Draw", "N/A").split(" ")[0]
            avg_pf_val = kpi_summary.get("Avg. Total PF", 0) # Already a float
            # --- END FIX ---
            
            st.subheader("Performance Metrics")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Consumed Energy", f"{float(total_kwh_str):.2f} kWh" if total_kwh_str != "N/A" else "N/A")
            col2.metric("Peak Demand (MD)", f"{float(peak_kva_str):.2f} kVA" if peak_kva_str != "N/A" else "N/A")
            col3.metric("Average Power Draw", f"{float(avg_kw_str):.2f} kW" if avg_kw_str != "N/A" else "N/A")
            col4.metric("Avg. Total PF", f"{avg_pf_val:.3f}" if avg_pf_val > 0 else "N/A")
            
            # --- RESTORED TABS ---
            tab_names = ["⚡ Power & Energy", "📝 Measurement Settings", "📋 Full Data Table"]
            tabs = st.tabs(tab_names)
            
            with tabs[0]:
                plot_cols = [col for col in ['Avg Real Power (kW)', 'Avg Apparent Power (kVA)', 'Avg Reactive Power (kVAR)'] if col in data.columns]
                if plot_cols:
                    st.subheader("Power Consumption Over Time")
                    st.info("This graph shows the Real (useful work), Apparent (total supplied), and Reactive (wasted) power. Look for high Apparent or Reactive power relative to Real power, which indicates electrical inefficiency.")
                    fig_power = px.line(data, x='Datetime', y=plot_cols)
                    
                    # --- GRAPH FORMATTING ---
                    fig_power.update_layout(
                        xaxis_title="Date & Time",
                        yaxis_title="Power (kW, kVA, kVAR)",
                        xaxis_tickformat="%a %d %b\n%H:%M" # e.g., "Mon 21 Oct 14:00"
                    )
                    fig_power.update_xaxes(showticklabels=True)
                    # --- END FORMATTING ---
                    
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
                    
                    # --- GRAPH FORMATTING ---
                    fig_pf.update_layout(
                        xaxis_title="Date & Time",
                        yaxis_title="Power Factor",
                        xaxis_tickformat="%a %d %b\n%H:%M"
                    )
                    fig_pf.update_xaxes(showticklabels=True)
                    # --- END FORMATTING ---
                    
                    st.plotly_chart(fig_pf, use_container_width=True)
                    with st.expander("Show Power Factor Statistics"):
                        stats_pf = {
                            "Minimum Power Factor": f"{data['Power Factor'].min():.3f}",
                            "Average Power Factor": f"{avg_pf_val:.3f}" # Use the corrected average
                        }
                        st.json(stats_pf)

            with tabs[1]:
                st.subheader("Measurement Settings")
                st.dataframe(parameters)
            
            with tabs[2]:
                st.subheader("Full Raw Data Table (All Status Codes)")
                st.info("This table shows the complete, unfiltered data, including any rows with error status codes. All graphs and KPIs are calculated *after* filtering out non-zero status rows.")
                st.dataframe(data_raw)
                
                # Download button downloads the CLEAN (Status=0) data_full
                st.download_button(
                    label="📥 Download Clean Data (Excel)",
                    data=to_excel_bytes(data_full),
                    file_name=f"{uploaded_file.name.split('.')[0]}_processed_clean.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    help="This downloads only the data rows with Status=0, which are used for all graphs and analysis."
                )

        elif wiring_system == '3P4W':
            st.header("Three-Phase System Diagnostic")
            
            # --- KPI TYPE ERROR FIX ---
            avg_power_kw_str = kpi_summary.get("Avg. Total Power", "N/A").split(" ")[0]
            peak_kva_3p_str = kpi_summary.get("Peak Demand (MD)", "N/A").split(" ")[0]
            avg_pf_val = kpi_summary.get("Avg. Total PF", 0) # Already a float
            imbalance_str = kpi_summary.get("Max Current Imbalance", "N/A").split(" ")[0]
            # --- END FIX ---
            
            st.subheader("Performance Metrics")
            col1, col2, col3, col4 = st.columns(4)
            
            col1.metric("Avg. Total Power", f"{float(avg_power_kw_str):.2f} kW" if avg_power_kw_str != "N/A" else "N/A")
            col2.metric("Peak Demand (MD)", f"{float(peak_kva_3p_str):.2f} kVA" if peak_kva_3p_str != "N/A" else "N/A")
            col3.metric("Avg. Total PF", f"{avg_pf_val:.3f}" if avg_pf_val > 0 else "N/A")
            col4.metric("Max Current Imbalance", f"{float(imbalance_str):.1f} %" if imbalance_str != "N/A" else "N/A", help="Under 5% is good.")

            # --- RESTORED TABS ---
            tab_names_3p = ["📅 Daily Breakdown", "📊 Current & Load Balance", "🩺 Voltage Health", "⚡ Power Analysis", "⚖️ Power Factor", "📝 Settings", "📋 Full Data Table"]
            tabs = st.tabs(tab_names_3p)
            
            with tabs[0]:
                st.subheader("24-Hour Operational Snapshot")
                st.info("Select a specific day to generate a detailed 24-hour subplot of all key electrical parameters. This is essential for comparing shift performance or analyzing specific production runs.")
                # Use data_full for the selector to show all available days
                unique_days = data_full['Datetime'].dt.date.unique()
                
                # Use requested date format
                day_format_func = lambda d: d.strftime('%a %d %b %Y')
                
                selected_day = st.selectbox("Select a day for detailed analysis:", options=unique_days, format_func=day_format_func)
                
                if selected_day:
                    # Filter the *clean* data for the selected day
                    daily_data = data[data['Datetime'].dt.date == selected_day]
                    
                    if daily_data.empty:
                        st.warning("No data found for the selected day in the current time filter. Try expanding the time filter.")
                    else:
                        fig = make_subplots(rows=4, cols=1, shared_xaxes=True, subplot_titles=("Voltage Envelope (V)", "Current Envelope (A)", "Real Power (kW)", "Power Factor"))

                        # Plot Voltage
                        for i in range(1, 4):
                            for stat in ['Min', 'Avg', 'Max']:
                                col = f'L{i} {stat} Voltage (V)'
                                if col in daily_data.columns:
                                    fig.add_trace(go.Scatter(x=daily_data['Datetime'], y=daily_data[col], name=col, mode='lines'), row=1, col=1)
                        fig.update_yaxes(title_text="Voltage (V)", row=1, col=1)
                        
                        # Plot Current
                        for i in range(1, 4):
                            for stat in ['Min', 'Avg', 'Max']:
                                col = f'L{i} {stat} Current (A)'
                                if col in daily_data.columns:
                                    fig.add_trace(go.Scatter(x=daily_data['Datetime'], y=daily_data[col], name=col, mode='lines'), row=2, col=1)
                        fig.update_yaxes(title_text="Current (A)", row=2, col=1)
                        
                        # Plot Real Power
                        for i in range(1, 4):
                            col = f'L{i} Avg Real Power (kW)'
                            if col in daily_data.columns:
                                fig.add_trace(go.Scatter(x=daily_data['Datetime'], y=daily_data[col], name=col, mode='lines'), row=3, col=1)
                        fig.update_yaxes(title_text="Power (kW)", row=3, col=1)

                        # Plot Power Factor
                        for i in range(1, 4):
                            col = f'L{i} Power Factor'
                            if col in daily_data.columns:
                                fig.add_trace(go.Scatter(x=daily_data['Datetime'], y=daily_data[col], name=col, mode='lines'), row=4, col=1)
                        fig.update_yaxes(title_text="Power Factor", row=4, col=1)

                        fig.update_layout(height=1000, title_text=f"Full Operational Breakdown for {selected_day.strftime('%a %d %b %Y')}", showlegend=True)
                        fig.update_xaxes(
                            tickformat="%H:%M", # Show HH:MM for daily breakdown
                            title_text="Time of Day",
                            row=4, col=1
                        )
                        st.plotly_chart(fig, use_container_width=True)
            
            with tabs[1]:
                st.subheader("Current Operational Envelope per Phase")
                st.info("This chart shows the full range of current drawn by the machine on each phase, from minimum to maximum. It is crucial for identifying peak inrush currents during start-up and understanding the full load variation.")
                current_cols_all = [f'{p} {s} Current (A)' for p in ['L1', 'L2', 'L3'] for s in ['Min', 'Avg', 'Max']]
                plot_cols = [c for c in current_cols_all if c in data.columns]
                if plot_cols:
                    fig = px.line(data, x='Datetime', y=plot_cols)
                    
                    # --- GRAPH FORMATTING ---
                    fig.update_layout(
                        xaxis_title="Date & Time",
                        yaxis_title="Current (A)",
                        xaxis_tickformat="%a %d %b\n%H:%M"
                    )
                    fig.update_xaxes(showticklabels=True)
                    # --- END FORMATTING ---
                    
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
                    
                    # --- GRAPH FORMATTING ---
                    fig.update_layout(
                        xaxis_title="Date & Time",
                        yaxis_title="Voltage (V)",
                        xaxis_tickformat="%a %d %b\n%H:%M"
                    )
                    fig.update_xaxes(showticklabels=True)
                    # --- END FORMATTING ---
                    
                    st.plotly_chart(fig, use_container_width=True)
                    with st.expander("Show Voltage Statistics"):
                        st.dataframe(data[plot_cols].describe().T[['mean', 'min', 'max']].rename(columns={'mean':'Average', 'min':'Minimum', 'max':'Maximum'}))

            with tabs[3]:
                st.subheader("Power Analysis")
                st.info("These charts show the Real (useful work), Apparent (total), and Reactive (wasted) power. The top chart shows the total system power, while the bottom chart breaks down the real power by phase to identify imbalances in work being done.")
                total_power_cols = [c for c in ['Total Avg Real Power (kW)', 'Total Avg Apparent Power (kVA)', 'Total Avg Reactive Power (kVAR)'] if c in data.columns]
                if total_power_cols:
                    fig = px.line(data, x='Datetime', y=total_power_cols, title="Total System Power")
                    
                    # --- GRAPH FORMATTING ---
                    fig.update_layout(
                        xaxis_title="Date & Time",
                        yaxis_title="Power (kW, kVA, kVAR)",
                        xaxis_tickformat="%a %d %b\n%H:%M"
                    )
                    fig.update_xaxes(showticklabels=True)
                    # --- END FORMATTING ---
                    
                    st.plotly_chart(fig, use_container_width=True)

                phase_power_cols = [c for c in ['L1 Avg Real Power (kW)', 'L2 Avg Real Power (kW)', 'L3 Avg Real Power (kW)'] if c in data.columns]
                if phase_power_cols:
                    fig2 = px.line(data, x='Datetime', y=phase_power_cols, title="Real Power per Phase")
                    
                    # --- GRAPH FORMATTING ---
                    fig2.update_layout(
                        xaxis_title="Date & Time",
                        yaxis_title="Power (kW)",
                        xaxis_tickformat="%a %d %b\n%H:%M"
                    )
                    fig2.update_xaxes(showticklabels=True)
                    # --- END FORMATTING ---
                    
                    st.plotly_chart(fig2, use_container_width=True)

            with tabs[4]:
                st.subheader("Power Factor per Phase Analysis")
                st.info("This chart shows the efficiency of each phase. The 'Total Power Factor' (often blue) is recalculated as Total Real Power / Total Apparent Power. If it's lower than the individual phases, it indicates system-wide issues like harmonic distortion.")
                
                # --- PLOT FIX: Add Total PF ---
                pf_cols = [c for c in ['L1 Power Factor', 'L2 Power Factor', 'L3 Power Factor', 'Total Power Factor'] if c in data.columns]
                # --- END FIX ---
                
                if pf_cols:
                    fig = px.line(data, x='Datetime', y=pf_cols)
                    
                    # --- GRAPH FORMATTING ---
                    fig.update_layout(
                        xaxis_title="Date & Time",
                        yaxis_title="Power Factor",
                        xaxis_tickformat="%a %d %b\n%H:%M"
                    )
                    fig.update_xaxes(showticklabels=True)
                    # --- END FORMATTING ---
                    
                    st.plotly_chart(fig, use_container_width=True)
                    with st.expander("Show Power Factor Statistics"):
                        # --- PF STATS FIX ---
                        pf_stats_data = {}
                        
                        # Get power threshold
                        power_col = 'Total Avg Real Power (kW)'
                        
                        for col in pf_cols:
                            if col not in data.columns: continue
                            
                            col_series = pd.Series(dtype=float)
                            # Use correct power column for filtering
                            if power_col in data.columns:
                                col_series = data[data[power_col] > POWER_THRESHOLD_KW][col].dropna()
                            
                            if not col_series.empty:
                                stats = col_series.describe()
                                pf_stats_data[col] = {
                                    "Average (Operational)": f"{stats['mean']:.3f}",
                                    "Minimum": f"{stats['min']:.3f}",
                                    "Maximum": f"{stats['max']:.3f}"
                                }
                            else:
                                pf_stats_data[col] = {"Average (Operational)": "N/A"}
                        st.json(pf_stats_data)
                        # --- END FIX ---


            with tabs[5]:
                st.subheader("Measurement Settings")
                st.dataframe(parameters)
            
            with tabs[6]:
                st.subheader("Full Raw Data Table (All Status Codes)")
                st.info("This table shows the complete, unfiltered data, including any rows with error status codes. All graphs and KPIs are calculated *after* filtering out non-zero status rows.")
                st.dataframe(data_raw)
                
                # Download button downloads the CLEAN (Status=0) data_full
                st.download_button(
                    label="📥 Download Clean Data (Excel)",
                    data=to_excel_bytes(data_full),
                    file_name=f"{uploaded_file.name.split('.')[0]}_processed_clean.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    help="This downloads only the data rows with Status=0, which are used for all graphs and analysis."
                )
        
        # --- AI Section (Common to both) ---
        st.sidebar.markdown("---")
        st.sidebar.subheader("Add Custom AI Context")
        additional_context = st.sidebar.text_area("Provide specific details about the machine or process (optional):", help="E.g., 'This is the main dough mixer, model XYZ.' or 'The large spike at 10:00 was a planned startup.'")

        if st.sidebar.button("🤖 Get AI-Powered Analysis"):
            if data.empty:
                st.sidebar.error("Cannot run analysis on an empty dataset. Widen your time filter.")
            else:
                with st.spinner("🧠 AI is analyzing the data... This may take a moment."):
                    # 1. KPIs from clean, filtered data
                    # (kpi_summary was already generated above by generate_kpis())
                    summary_metrics_text = "\n".join([f"- {key}: {value}" for key, value in kpi_summary.items() if "N/A" not in str(value)])
                    # 2. Trend from clean, filtered data
                    trend_summary_text = generate_trend_summary(data, wiring_system)
                    # 3. Stats from clean, filtered data
                    detailed_stats_text = generate_detailed_analysis_text(data, wiring_system)
                    # 4. Measurement settings
                    params_info_text = parameters.to_string()
                    # 5. Transformation Log
                    transform_log_text = generate_transform_summary(uploaded_file.name, data_raw, data_full)

                    
                    ai_response = get_gemini_analysis(
                        summary_metrics_text,
                        detailed_stats_text,
                        trend_summary_text,
                        params_info_text,
                        transform_log_text,
                        additional_context
                    )
                    st.session_state['ai_analysis'] = ai_response
                    
                    # --- Save data for PDF generation ---
                    st.session_state['kpi_summary_selected'] = kpi_summary
                    # Generate and save full-period KPIs
                    st.session_state['kpi_summary_full'] = generate_kpis(data_full, wiring_system)
                    st.session_state['pdf_ready'] = True
                    # --- END ---


        if 'ai_analysis' in st.session_state:
            st.markdown("---")
            st.header("🤖 AI-Powered Analysis")
            st.markdown(st.session_state['ai_analysis'])
            
            # --- DOWNLOAD BUTTONS ---
            st.markdown("---") # Add a separator
            
            dl_col1, dl_col2 = st.columns(2)
            
            # Button 1: Download AI text
            dl_col1.download_button(
                label="📄 Download AI Report (.txt)",
                data=st.session_state['ai_analysis'],
                file_name=f"{uploaded_file.name.split('.')[0]}_ai_analysis.txt",
                mime="text/plain",
                help="Downloads only the text-based AI analysis."
            )
            
            # Button 2: Download Full PDF
            if st.session_state.get('pdf_ready', False):
                with st.spinner("Building PDF Report..."):
                    pdf_bytes = generate_pdf_report(
                        file_name=uploaded_file.name,
                        parameters=parameters,
                        wiring_system=wiring_system,
                        kpi_summary_selected=st.session_state['kpi_summary_selected'],
                        kpi_summary_full=st.session_state['kpi_summary_full'],
                        ai_analysis=st.session_state['ai_analysis'],
                        data_full=data_full # Pass full, clean data for graphs
                    )
                
                dl_col2.download_button(
                    label="📕 Download Full PDF Report",
                    data=pdf_bytes,
                    file_name=f"{uploaded_file.name.split('.')[0]}_analysis_report.pdf",
                    mime="application/pdf",
                    help="Downloads the complete report with AI analysis, KPIs, and all graphs."
                )
            # --- END DOWNLOADS ---

    elif uploaded_file is not None:
        # This triggers if process_result is None
        st.warning("Could not process the uploaded file. Please ensure it is a valid, non-empty Hioki CSV export.")
# Download button downloads the CLEAN (Status=0) data_full
                st.download_button(
                    label="📥 Download Clean Data (Excel)",
                    data=to_excel_bytes(data_full),
                    file_name=f"{uploaded_file.name.split('.')[0]}_processed_clean.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    help="This downloads only the data rows with Status=0, which are used for all graphs and analysis."
                )
        
        # --- AI Section (Common to both) ---
        st.sidebar.markdown("---")
        st.sidebar.subheader("Add Custom AI Context")
        additional_context = st.sidebar.text_area("Provide specific details about the machine or process (optional):", help="E.g., 'This is the main dough mixer, model XYZ.' or 'The large spike at 10:00 was a planned startup.'")

        if st.sidebar.button("🤖 Get AI-Powered Analysis"):
            if data.empty:
                st.sidebar.error("Cannot run analysis on an empty dataset. Widen your time filter.")
            else:
                with st.spinner("🧠 AI is analyzing the data... This may take a moment."):
                    # 1. KPIs from clean, filtered data
                    # (kpi_summary was already generated above by generate_kpis())
                    summary_metrics_text = "\n".join([f"- {key}: {value}" for key, value in kpi_summary.items() if "N/A" not in str(value)])
                    # 2. Trend from clean, filtered data
                    trend_summary_text = generate_trend_summary(data, wiring_system)
                    # 3. Stats from clean, filtered data
                    detailed_stats_text = generate_detailed_analysis_text(data, wiring_system)
                    # 4. Measurement settings
                    params_info_text = parameters.to_string()
                    # 5. Transformation Log
                    transform_log_text = generate_transform_summary(uploaded_file.name, data_raw, data_full)

                    
                    ai_response = get_gemini_analysis(
                        summary_metrics_text,
                        detailed_stats_text,
                        trend_summary_text,
                        params_info_text,
                        transform_log_text,
                        additional_context
                    )
                    st.session_state['ai_analysis'] = ai_response
                    
                    # --- Save data for PDF generation ---
                    st.session_state['kpi_summary_selected'] = kpi_summary
                    # Generate and save full-period KPIs
                    st.session_state['kpi_summary_full'] = generate_kpis(data_full, wiring_system)
                    st.session_state['pdf_ready'] = True
                    # --- END ---


        if 'ai_analysis' in st.session_state:
            st.markdown("---")
            st.header("🤖 AI-Powered Analysis")
            st.markdown(st.session_state['ai_analysis'])
            
            # --- DOWNLOAD BUTTONS ---
            st.markdown("---") # Add a separator
            
            dl_col1, dl_col2 = st.columns(2)
            
            # Button 1: Download AI text
            dl_col1.download_button(
                label="📄 Download AI Report (.txt)",
                data=st.session_state['ai_analysis'],
                file_name=f"{uploaded_file.name.split('.')[0]}_ai_analysis.txt",
                mime="text/plain",
                help="Downloads only the text-based AI analysis."
            )
            
            # Button 2: Download Full PDF
            if st.session_state.get('pdf_ready', False):
                with st.spinner("Building PDF Report..."):
                    pdf_bytes = generate_pdf_report(
                        file_name=uploaded_file.name,
                        parameters=parameters,
                        wiring_system=wiring_system,
                        kpi_summary_selected=st.session_state['kpi_summary_selected'],
                        kpi_summary_full=st.session_state['kpi_summary_full'],
                        ai_analysis=st.session_state['ai_analysis'],
                        data_full=data_full # Pass full, clean data for graphs
                    )
                
                dl_col2.download_button(
                    label="📕 Download Full PDF Report",
                    data=pdf_bytes,
                    file_name=f"{uploaded_file.name.split('.')[0]}_analysis_report.pdf",
                    mime="application/pdf",
                    help="Downloads the complete report with AI analysis, KPIs, and all graphs."
                )
            # --- END DOWNLOADS ---

    elif uploaded_file is not None:
        # This triggers if process_result is None
        st.warning("Could not process the uploaded file. Please ensure it is a valid, non-empty Hioki CSV export.")
