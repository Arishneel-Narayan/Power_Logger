import streamlit as st
import pandas as pd
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from typing import Tuple, Optional, Dict
import requests
import io
import markdown # For HTML report
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

def generate_ai_data_context(data: pd.DataFrame, wiring_system: str) -> str:
    """
    Generates a single, comprehensive markdown string containing all peak event
    and statistical data for the AI, preventing timestamp/value mix-ups.
    """
    if data.empty:
        return "No data available for analysis."

    # --- Define Key Metric Columns ---
    pf_col, power_col, apparent_col = "", "", ""
    phase_cols = {}
    
    if wiring_system == '3P4W':
        pf_col = 'Total Power Factor'
        power_col = 'Total Avg Real Power (kW)'
        apparent_col = 'Total Avg Apparent Power (kVA)'
        phase_cols = {
            'L1 Current (A)': 'L1 Avg Current (A)',
            'L2 Current (A)': 'L2 Avg Current (A)',
            'L3 Current (A)': 'L3 Avg Current (A)',
            'L1 Voltage (V)': 'L1 Avg Voltage (V)',
            'L2 Voltage (V)': 'L2 Avg Voltage (V)',
            'L3 Voltage (V)': 'L3 Avg Voltage (V)',
            'L1 Power Factor': 'L1 Power Factor',
            'L2 Power Factor': 'L2 Power Factor',
            'L3 Power Factor': 'L3 Power Factor',
        }
    elif wiring_system == '1P2W':
        pf_col = 'Power Factor'
        power_col = 'Avg Real Power (kW)'
        apparent_col = 'Avg Apparent Power (kVA)'
        phase_cols = {
            'Current (A)': 'Avg Current (A)',
            'Voltage (V)': 'Avg Voltage (V)',
        }

    # --- 1. Peak Event Summary (Unambiguous) ---
    summary_lines = ["## Peak Event Summary"]
    date_format = '%a %d %b %Y, %H:%M:%S'

    # Find Peak kVA (MD)
    if apparent_col in data.columns and not data[apparent_col].dropna().empty:
        peak_kva_val = data[apparent_col].max()
        peak_kva_time = data.loc[data[apparent_col].idxmax(), 'Datetime'].strftime(date_format)
        summary_lines.append(f"- **Peak Demand (MD):** {peak_kva_val:.2f} kVA (at {peak_kva_time})")
    else:
        summary_lines.append("- **Peak Demand (MD):** N/A")

    # Find Peak kW (Real Power)
    if power_col in data.columns and not data[power_col].dropna().empty:
        peak_kw_val = data[power_col].max()
        peak_kw_time = data.loc[data[power_col].idxmax(), 'Datetime'].strftime(date_format)
        summary_lines.append(f"- **Peak Real Power:** {peak_kw_val:.2f} kW (at {peak_kw_time})")
    else:
        summary_lines.append("- **Peak Real Power:** N/A")

    # --- 2. Detailed Statistical Table ---
    summary_lines.append("\n## Detailed Statistical Summary")
    
    stats_header = "| Metric | Mean | Min | Max | Std Dev (Volatility) |"
    stats_divider = "|:---|---:|---:|---:|---:|"
    stats_rows = [stats_header, stats_divider]

    # Helper to get stats for a column
    def get_stats_row(metric_name: str, col_name: str, data: pd.DataFrame, is_pf: bool = False):
        if col_name in data.columns and not data[col_name].dropna().empty:
            series = data[col_name].dropna()
            
            # For PF, filter by operational threshold
            if is_pf and power_col in data.columns and not data[data[power_col] > POWER_THRESHOLD_KW].empty:
                series = data[data[power_col] > POWER_THRESHOLD_KW][col_name].dropna()
                if series.empty:
                    return f"| {metric_name} | N/A (No load) | N/A | N/A | N/A |"
            
            stats = series.describe()
            return f"| {metric_name} | {stats['mean']:.2f} | {stats['min']:.2f} | {stats['max']:.2f} | {stats['std']:.2f} |"
        return f"| {metric_name} | N/A | N/A | N/A | N/A |"

    # Add system-wide stats
    stats_rows.append(get_stats_row("Total Real Power (kW)", power_col, data))
    stats_rows.append(get_stats_row("Total Apparent Power (kVA)", apparent_col, data))
    stats_rows.append(get_stats_row("Total Power Factor", pf_col, data, is_pf=True))

    # Add phase-specific stats
    for metric_name, col_name in phase_cols.items():
        is_pf = 'Power Factor' in metric_name
        stats_rows.append(get_stats_row(metric_name, col_name, data, is_pf=is_pf))

    summary_lines.extend(stats_rows)
    return "\n".join(summary_lines)


def get_pulse_analysis(ai_data_context: str,
                        params_info: str, 
                        transform_log: str, 
                        additional_context: str = "") -> str:
    """Contacts the PULSE AI for an expert analysis."""
    
    # UPDATED SYSTEM PROMPT
    system_prompt = """You are PULSE (Power Usage Learning and Support Engine), an expert quantitative analyst for FMF Foods Ltd. Your task is to analyze power data for the process optimization engineer.

    Your analysis MUST be:
    1.  **Numbers-Based:** Be quantitative. Use numbers from the tables provided.
    2.  **Concise:** Use short, simple sentences. Use bullet points and markdown tables.
    3.  **Explanatory:** Explain trends (e.g., "High Std Dev means high load volatility").
    4.  **Referential:** You MUST reference the specific 'Peak Demand (MD)' and 'Peak Real Power' events from the 'Peak Event Summary'.
    5.  **Standards-Based:** Cite relevant standards (e.g., IEEE, IEC, NEMA) to support observations.
    
    Your analysis MUST be based SOLELY on the 'CLEANED (Status=0) DATA'.

    Core Principles:
    - **Pattern Analysis:** Use the 'Detailed Statistical Summary' table to analyze key metrics.
    - **Peak Events:** Use the 'Peak Event Summary' to understand the *specific* peak demand (kVA) and peak power (kW) events.
    - **Cost Reduction:** Focus on reducing 'Peak Demand (MD)' and improving 'Total Power Factor'.
    - **Quantitative Significance:** Refer to the absolute values in the tables.
    - **CRITICAL ENGINEERING FEEDBACK:** De-prioritize current imbalance recommendations if the *absolute* difference between phase currents (in Amps) is minor (e.g., less than 50A), even if the *percentage* seems high.
    
    Provide a short, informative report in Markdown format with three sections:
    1.  **PULSE Executive Summary** (2-3 key bullet points)
    2.  **Key Observations** (Use a markdown table and bullet points based on the 'Detailed Statistical Summary')
    3.  **Actionable Recommendations** (Use numbered bullet points, citing standards)
    
    Address the user as a fellow process optimization engineer."""
    
    user_prompt = f"""
    Good morning, Please analyze the following power consumption data for an industrial machine at our Suva facility.
    
    **Data Transformation Log (CRITICAL CONTEXT):**
    {transform_log}
    
    **PULSE Data Analysis (Selected Period):**
    {ai_data_context}
    
    **Measurement Parameters:**
    {params_info}
    """
    if additional_context:
        user_prompt += f"\n**Additional Engineer's Context:**\n{additional_context}"
    user_prompt += "\nBased on all this information, please generate your concise, quantitative, standards-based report."
    
    try:
        api_key = st.secrets["GEMINI_API_KEY"]
    except (KeyError, FileNotFoundError):
        return "Error: PULSE API key not found. Please add it to your Streamlit Secrets."
    
    # Use gemini-2.5-pro and corrected URL spelling
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
            return f"Error from PULSE API: {result['error']['message']}"
            
        candidate = result.get('candidates', [{}])[0]
        
        # Check for safety ratings or finish reason
        if candidate.get('finishReason') not in ['STOP', 'MAX_TOKENS']:
             return f"Error: PULSE API call finished unexpectedly. Reason: {candidate.get('finishReason', 'Unknown')}"

        if not candidate.get('content', {}).get('parts', []):
            return "Error: PULSE API returned an empty response. This may be due to safety filters."
            
        content = candidate['content']['parts'][0]
        return content.get('text', "Error: Could not extract analysis from the PULSE API response.")
        
    except requests.exceptions.HTTPError as http_err:
        return f"HTTP error occurred: {http_err} - {response.text}"
    except requests.exceptions.RequestException as req_err:
        return f"A network error occurred: {req_err}"
    except Exception as e:
        return f"An unexpected error occurred while contacting PULSE: {e}"

# --- 3. Helper Function for Excel Download ---

def to_excel_bytes(df: pd.DataFrame) -> bytes:
    """Converts a DataFrame to an in-memory Excel file (bytes)."""
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Processed_Data')
    return output.getvalue()

# --- 4. HTML Report Generation Function ---

def generate_html_report(
    file_name: str,
    parameters: pd.DataFrame,
    wiring_system: str,
    kpi_summary_selected: dict,
    kpi_summary_full: dict,
    pulse_analysis: str,
    data_full: pd.DataFrame # Use full, clean data for graphs
) -> bytes:
    """Generates a self-contained HTML report with embedded graphs."""
    
    # Helper to create a KPI table
    def kpi_to_html_table(kpi_dict: dict) -> str:
        rows = ""
        for key, value in kpi_dict.items():
            # Format numbers, leave strings as-is
            val_str = value
            if isinstance(value, (int, float)):
                if key == "Avg. Total PF":
                    val_str = f"{value:.3f}"
                elif key == "Max Current Imbalance":
                     val_str = f"{value:.1f} %"
                else:
                    val_str = f"{value:.2f}"
            rows += f"<tr><th>{key}</th><td>{val_str}</td></tr>"
        return f"<table>{rows}</table>"

    # Helper to generate and embed a graph
    def get_graph_html(fig, title: str) -> str:
        if fig is None:
            return f"<h3>{title}</h3><p>Data not available to generate this graph.</p>"
        graph_html = pio.to_html(fig, full_html=False, include_plotlyjs='cdn')
        return f"<h3>{title}</h3>{graph_html}"

    # --- Generate all graphs from FULL data ---
    graphs_html = ""
    
    if wiring_system == '1P2W':
        plot_cols = [col for col in ['Avg Real Power (kW)', 'Avg Apparent Power (kVA)', 'Avg Reactive Power (kVAR)'] if col in data_full.columns]
        fig_power = None
        if plot_cols:
            fig_power = px.line(data_full, x='Datetime', y=plot_cols, template="plotly") # Add template
            fig_power.update_layout(xaxis_title="Date & Time", yaxis_title="Power (kW, kVA, kVAR)", xaxis_tickformat="%a %d %b\n%H:%M")
        graphs_html += get_graph_html(fig_power, "Power Consumption Over Time (Full Period)")

        fig_pf = None
        if 'Power Factor' in data_full.columns:
            fig_pf = px.line(data_full, x='Datetime', y='Power Factor', template="plotly") # Add template
            fig_pf.add_hline(y=0.95, line_dash="dash", line_color="red")
            fig_pf.update_layout(xaxis_title="Date & Time", yaxis_title="Power Factor", xaxis_tickformat="%a %d %b\n%H:%M")
        graphs_html += get_graph_html(fig_pf, "Power Factor Over Time (Full Period)")
    
    elif wiring_system == '3P4W':
        fig_power_total = None
        total_power_cols = [c for c in ['Total Avg Real Power (kW)', 'Total Avg Apparent Power (kVA)', 'Total Avg Reactive Power (kVAR)'] if c in data_full.columns]
        if total_power_cols:
            fig_power_total = px.line(data_full, x='Datetime', y=total_power_cols, template="plotly") # Add template
            fig_power_total.update_layout(xaxis_title="Date & Time", yaxis_title="Power (kW, kVA, kVAR)", xaxis_tickformat="%a %d %b\n%H:%M")
        graphs_html += get_graph_html(fig_power_total, "Total System Power (Full Period)")

        fig_current = None
        current_cols_all = [f'{p} {s} Current (A)' for p in ['L1', 'L2', 'L3'] for s in ['Min', 'Avg', 'Max']]
        plot_cols_current = [c for c in current_cols_all if c in data_full.columns]
        if plot_cols_current:
            fig_current = px.line(data_full, x='Datetime', y=plot_cols_current, template="plotly") # Add template
            fig_current.update_layout(xaxis_title="Date &Time", yaxis_title="Current (A)", xaxis_tickformat="%a %d %b\n%H:%M")
        graphs_html += get_graph_html(fig_current, "Current Operational Envelope (Full Period)")

        fig_voltage = None
        voltage_cols_all = [f'{p} {s} Voltage (V)' for p in ['L1', 'L2', 'L3'] for s in ['Min', 'Avg', 'Max']]
        plot_cols_voltage = [c for c in voltage_cols_all if c in data_full.columns]
        if plot_cols_voltage:
            fig_voltage = px.line(data_full, x='Datetime', y=plot_cols_voltage, template="plotly") # Add template
            fig_voltage.update_layout(xaxis_title="Date & Time", yaxis_title="Voltage (V)", xaxis_tickformat="%a %d %b\n%H:%M")
        graphs_html += get_graph_html(fig_voltage, "Voltage Operational Envelope (Full Period)")
            
        fig_pf_phase = None
        pf_cols = [c for c in ['L1 Power Factor', 'L2 Power Factor', 'L3 Power Factor', 'Total Power Factor'] if c in data_full.columns]
        if pf_cols:
            fig_pf_phase = px.line(data_full, x='Datetime', y=pf_cols, template="plotly") # Add template
            fig_pf_phase.update_layout(xaxis_title="Date &Time", yaxis_title="Power Factor", xaxis_tickformat="%a %d %b\n%H:%M")
        graphs_html += get_graph_html(fig_pf_phase, "Power Factor per Phase (Full Period)")
    
    # --- Convert Markdown to HTML ---
    pulse_analysis_html = markdown.markdown(pulse_analysis, extensions=['tables'])
    
    # --- Get Other HTML Elements ---
    kpi_selected_html = kpi_to_html_table(kpi_summary_selected)
    kpi_full_html = kpi_to_html_table(kpi_summary_full)
    parameters_html = parameters.to_html()
    
    # --- Assemble Final HTML ---
    html_template = f"""
    <html>
    <head>
        <title>FMF PULSE Analysis: {file_name}</title>
        <style>
            body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; margin: 0 auto; max-width: 1200px; padding: 20px; }}
            h1, h2, h3 {{ color: #004a99; }}
            h1 {{ border-bottom: 2px solid #004a99; padding-bottom: 10px; }}
            h2 {{ border-bottom: 1px solid #ccc; padding-bottom: 5px; }}
            table {{ border-collapse: collapse; width: 60%; margin-bottom: 20px; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; font-weight: bold; }}
            pre {{ background-color: #f8f8f8; padding: 10px; border: 1px solid #ddd; border-radius: 5px; }}
            code {{ font-family: monospace; }}
            .container {{ display: grid; grid-template-columns: 1fr 1fr; gap: 40px; }}
            .full-width {{ grid-column: 1 / -1; }}
            .kpi-section {{ break-inside: avoid; }}
            /* Styles for AI-generated tables */
            .pulse-analysis table {{ width: 100%; }}
            .pulse-analysis th {{ background-color: #e0ebf5; }}
        </style>
    </head>
    <body>
        <h1>FMF PULSE Analysis</h1>
        <p><strong>File:</strong> {file_name}</p>
        <p><strong>Generated:</strong> {pd.Timestamp.now(tz="Pacific/Fiji").strftime("%a %d %b %Y, %H:%M")}</p>
        
        <div class="full-width">
            <h2>1. PULSE Analysis</h2>
            <div class="pulse-analysis">
                {pulse_analysis_html}
            </div>
        </div>

        <h2>2. Key Performance Indicators (KPIs)</h2>
        <div class="container">
            <div class="kpi-section">
                <h3>Metrics for Selected Period</h3>
                {kpi_selected_html}
            </div>
            <div class="kpi-section">
                <h3>Metrics for Full Period (All Status=0 Data)</h3>
                {kpi_full_html}
            </div>
        </div>

        <div class="full-width">
            <h2>3. Full Period Graphs</h2>
            {graphs_html}
        </div>
        
        <div class="full-width">
            <h2>4. Measurement Settings</h2>
            {parameters_html}
        </div>
        
    </body>
    </html>
    """
    
    return html_template.encode('utf-8')


# --- 5. KPI Generation Function (Restored) ---
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
            "Avg. Total PF": avg_pf  # <-- Store raw float
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
            "Peak Demand (MD)": f"{peak_kva_3p:.2f} kVA",
            "Avg. Total PF": avg_pf,  # <-- Store raw float
            "Max Current Imbalance": f"{imbalance:.1f} %"
        }
    return kpi_summary


# --- 6. Streamlit UI and Analysis Section ---
st.set_page_config(layout="wide", page_title="FMF PULSE Analysis")
st.title("⚡ FMF PULSE Analysis Dashboard")
st.markdown(f"**P**ower **U**sage **L**earning and **S**upport **E**ngine")

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
                    fig_power = px.line(data, x='Datetime', y=plot_cols, template="plotly") # Add template
                    
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
                    fig_pf = px.line(data, x='Datetime', y='Power Factor', template="plotly") # Add template
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
                        fig = make_subplots(
                            rows=4, cols=1, shared_xaxes=True, 
                            subplot_titles=("Voltage Envelope (V)", "Current Envelope (A)", "Real Power (kW)", "Power Factor")
                        )
                        fig.update_layout(template="plotly") # Add template

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
                    fig = px.line(data, x='Datetime', y=plot_cols, template="plotly") # Add template
                    
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
                    fig = px.line(data, x='Datetime', y=plot_cols, template="plotly") # Add template
                    
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
                    fig = px.line(data, x='Datetime', y=total_power_cols, title="Total System Power", template="plotly") # Add template
                    
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
                    fig2 = px.line(data, x='Datetime', y=phase_power_cols, title="Real Power per Phase", template="plotly") # Add template
                    
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
                    fig = px.line(data, x='Datetime', y=pf_cols, template="plotly") # Add template
                    
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
        
        # --- PULSE AI Section (Common to both) ---
        st.sidebar.markdown("---")
        st.sidebar.subheader("Add Custom PULSE Context")
        additional_context = st.sidebar.text_area("Provide specific details about the machine or process (optional):", help="E.g., 'This is the main dough mixer, model XYZ.' or 'The large spike at 10:00 was a planned startup.'")

        if st.sidebar.button("🤖 Get PULSE Analysis"):
            if data.empty:
                st.sidebar.error("Cannot run analysis on an empty dataset. Widen your time filter.")
            else:
                with st.spinner("🧠 PULSE is analyzing the data... This may take a moment."):
                    # 1. Generate the single, unambiguous data context
                    ai_data_context = generate_ai_data_context(data, wiring_system)
                    
                    # 2. Measurement settings
                    params_info_text = parameters.to_string()
                    
                    # 3. Transformation Log
                    transform_log_text = generate_transform_summary(uploaded_file.name, data_raw, data_full)
                    
                    pulse_response = get_pulse_analysis(
                        ai_data_context,
                        params_info_text,
                        transform_log_text,
                        additional_context
                    )
                    st.session_state['pulse_analysis'] = pulse_response
                    
                    # --- Save data for HTML report generation ---
                    st.session_state['kpi_summary_selected'] = kpi_summary
                    # Generate and save full-period KPIs
                    st.session_state['kpi_summary_full'] = generate_kpis(data_full, wiring_system)
                    st.session_state['report_ready'] = True
                    # --- END ---


        if 'pulse_analysis' in st.session_state:
            st.markdown("---")
            st.header("🤖 PULSE Analysis")
            st.markdown(st.session_state['pulse_analysis'])
            
            # --- DOWNLOAD BUTTONS ---
            st.markdown("---") # Add a separator
            
            dl_col1, dl_col2 = st.columns(2)
            
            # Button 1: Download AI text
            dl_col1.download_button(
                label="📄 Download PULSE Report (.txt)",
                data=st.session_state['pulse_analysis'],
                file_name=f"{uploaded_file.name.split('.')[0]}_pulse_analysis.txt",
                mime="text/plain",
                help="Downloads only the text-based PULSE analysis."
            )
            
            # Button 2: Download Full HTML Report
            if st.session_state.get('report_ready', False):
                with st.spinner("Building HTML Report..."):
                    html_bytes = generate_html_report(
                        file_name=uploaded_file.name,
                        parameters=parameters,
                        wiring_system=wiring_system,
                        kpi_summary_selected=st.session_state['kpi_summary_selected'],
                        kpi_summary_full=st.session_state['kpi_summary_full'],
                        pulse_analysis=st.session_state['pulse_analysis'],
                        data_full=data_full # Pass full, clean data for graphs
                    )
                
                dl_col2.download_button(
                    label="📕 Download Full HTML Report",
                    data=html_bytes,
                    file_name=f"{uploaded_file.name.split('.')[0]}_pulse_analysis_report.html",
                    mime="text/html",
                    help="Downloads the complete report with PULSE analysis, KPIs, and all graphs. Open in browser and 'Print to PDF'."
                )
            # --- END DOWNLOADS ---

    elif uploaded_file is not None:
        # This triggers if process_result is None
        st.warning("Could not process the uploaded file. Please ensure it is a valid, non-empty Hioki CSV export.")
