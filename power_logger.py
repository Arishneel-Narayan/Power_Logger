import streamlit as st
import pandas as pd
import plotly.express as px
from typing import Tuple, Optional

# --- Data Processing Functions ---

def rename_power_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Renames the columns of a Hioki power analyzer DataFrame to plain English.
    """
    rename_map = {
        'Status': 'Machine Status', 'Freq_Avg[Hz]': 'Average Frequency (Hz)',
        'U1_Avg[V]': 'Average Voltage (V)', 'Ufnd1_Avg[V]': 'Fundamental Voltage (V)',
        'Udeg1_Avg[deg]': 'Voltage Phase Angle (deg)', 'I1_Avg[A]': 'Average Current (A)',
        'Ifnd1_Avg[A]': 'Fundamental Current (A)', 'Ideg1_Avg[deg]': 'Current Phase Angle (deg)',
        'P1_Avg[W]': 'Average Real Power (W)', 'S1_Avg[VA]': 'Average Apparent Power (VA)',
        'Q1_Avg[var]': 'Average Reactive Power (VAR)', 'PF1_Avg': 'Average Power Factor',
        'WP+1[Wh]': 'Consumed Real Energy (Wh)', 'WP-1[Wh]': 'Exported Real Energy (Wh)',
        'WQLAG1[varh]': 'Lagging Reactive Energy (VARh)', 'WQLEAD1[varh]': 'Leading Reactive Energy (VARh)',
        'Ecost1': 'Estimated Cost', 'WP+dem1[Wh]': 'Consumed Energy (Demand Period)',
        'WP-dem1[Wh]': 'Exported Energy (Demand Period)', 'WQLAGdem1[varh]': 'Lagging Reactive Energy (Demand Period)',
        'WQLEADdem1[varh]': 'Leading Reactive Energy (Demand Period)', 'Pdem+1[W]': 'Power Demand Consumed (W)',
        'Pdem-1[W]': 'Power Demand Exported (W)', 'QdemLAG1[var]': 'Lagging Reactive Power (Demand)',
        'QdemLEAD1[var]': 'Leading Reactive Power (Demand)', 'PFdem1': 'Power Factor (Demand)', 'Pulse': 'Pulse Count'
    }
    return df.rename(columns=rename_map)

def process_hioki_csv(uploaded_file, header_keyword: str = 'Date') -> Optional[Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Loads, cleans, and processes a Hioki power analyzer CSV file.
    """
    try:
        df_raw = pd.read_csv(uploaded_file, header=None, on_bad_lines='skip', encoding='utf-8')
    except Exception as e:
        st.error(f"Error reading CSV file: {e}")
        return None

    search_column = df_raw.iloc[:, 0].astype(str)
    header_indices = search_column[search_column.eq(header_keyword)].index
    
    if header_indices.empty:
        st.error(f"Error: Header keyword '{header_keyword}' not found in the first column.")
        return None
    header_row_index = header_indices[0]

    params_df = df_raw.iloc[:header_row_index, :2].copy()
    params_df.columns = ['Parameter', 'Value']
    params_df.set_index('Parameter', inplace=True)
    params_df.dropna(inplace=True)

    data_df = df_raw.iloc[header_row_index:].copy()
    data_df.columns = data_df.iloc[0]
    data_df = data_df.iloc[1:].reset_index(drop=True)

    if len(data_df.columns) > 1:
        cols_to_check = data_df.columns.drop('Date', errors='ignore')
        data_df.replace(r'^\s*$', pd.NA, regex=True, inplace=True)
        data_df.dropna(subset=cols_to_check, how='all', inplace=True)

    data_df['Datetime'] = pd.to_datetime(data_df['Date'] + ' ' + data_df['Etime'], errors='coerce')
    data_df = data_df.dropna(subset=['Datetime'])

    data_df = data_df.sort_values(by='Datetime').reset_index(drop=True)

    numeric_cols = data_df.columns.drop(['Date', 'Etime', 'Status', 'Datetime'], errors='ignore')
    for col in numeric_cols:
        data_df[col] = pd.to_numeric(data_df[col], errors='coerce')
        
    data_df = rename_power_columns(data_df)
    
    # --- FIX: Handle reversed wiring for Power Factor and Demand ---
    # Power factor is a ratio and should always be positive.
    if 'Average Power Factor' in data_df.columns:
        data_df['Average Power Factor'] = data_df['Average Power Factor'].abs()

    # Create a robust total demand column
    pdem_plus = data_df['Power Demand Consumed (W)'].fillna(0)
    pdem_minus = data_df['Power Demand Exported (W)'].fillna(0)
    data_df['Total Power Demand (kW)'] = (pdem_plus.abs() + pdem_minus.abs()) / 1000

    for col in ['Average Real Power (W)', 'Average Apparent Power (VA)', 'Average Reactive Power (VAR)']:
        if col in data_df.columns:
            new_col_name = col.replace('(W)', '(kW)').replace('(VA)', '(kVA)').replace('(VAR)', '(kVAR)')
            data_df[new_col_name] = data_df[col] / 1000

    return params_df, data_df

# --- Streamlit App Layout ---

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
    parameters, data = process_hioki_csv(uploaded_file)

    if parameters is not None and data is not None and not data.empty:
        st.sidebar.success("File processed successfully!")
        
        st.header("Analysis Overview")

        # Robust Energy Calculation
        delta_plus_wh = 0
        energy_plus_data = data['Consumed Real Energy (Wh)'].dropna()
        if len(energy_plus_data) > 1:
            delta_plus_wh = energy_plus_data.iloc[-1] - energy_plus_data.iloc[0]

        delta_minus_wh = 0
        energy_minus_data = data['Exported Real Energy (Wh)'].dropna()
        if len(energy_minus_data) > 1:
            delta_minus_wh = energy_minus_data.iloc[-1] - energy_minus_data.iloc[0]

        total_consumed_kwh = (abs(delta_plus_wh) + abs(delta_minus_wh)) / 1000

        # Wiring Check and Warning
        if total_consumed_kwh > 0 and (abs(delta_minus_wh) / (abs(delta_plus_wh) + abs(delta_minus_wh))) > 0.1:
            st.warning("""
            **⚠️ Potential Wiring Issue Detected** A significant amount of energy was recorded in the 'Exported' channel. This typically happens when a Current Transformer (CT) clamp is installed backward.
            
            **Impact:**
            - The **Total Consumed Energy** metric is calculated correctly by summing both channels.
            - **Average Real Power** and **Power Factor** may appear as negative values in plots and tables.
            
            Please verify the physical setup for future measurements.
            """, icon="⚠️")

        duration = data['Datetime'].max() - data['Datetime'].min()
        days, hours, rem = duration.days, duration.seconds // 3600, duration.seconds % 3600
        minutes = rem // 60
        duration_str = f"{days}d {hours}h {minutes}m"

        avg_power_kw = data['Average Real Power (kW)'].abs().mean()
        max_demand_kw = data['Total Power Demand (kW)'].max()
        avg_pf = data['Average Power Factor'].mean()

        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Measurement Duration", duration_str)
        col2.metric("Total Consumed Energy", f"{total_consumed_kwh:.2f} kWh")
        col3.metric("Average Power", f"{avg_power_kw:.2f} kW")
        col4.metric("Peak Demand", f"{max_demand_kw:.2f} kW")
        col5.metric("Average Power Factor", f"{avg_pf:.3f}", delta=f"{avg_pf - 0.95:.3f}", delta_color="inverse", help="Target is > 0.95. A negative delta is bad.")
        
        st.markdown("---")

        tab_list = ["⚡ Power & Current", "⚖️ Power Factor Analysis", "📈 Demand Analysis", "📋 Raw Data", "📝 Summary Parameters", "📖 Variable Explanations"]
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(tab_list)

        with tab1:
            st.subheader("Power Consumption Over Time")
            st.info("Note: A negative 'Average Real Power' value indicates a likely reversed CT clamp. The charts show the raw data.", icon="ℹ️")
            fig_power = px.line(data, x='Datetime', y=['Average Real Power (kW)', 'Average Apparent Power (kVA)', 'Average Reactive Power (kVAR)'],
                                title="Real, Apparent, and Reactive Power", labels={"value": "Power", "variable": "Power Type"})
            fig_power.update_layout(yaxis_title="Power (kVA, kW, kVAR)", hovermode="x unified")
            st.plotly_chart(fig_power, use_container_width=True)

        with tab2:
            st.subheader("Power Factor Analysis")
            st.markdown("This chart shows the corrected, absolute Power Factor. A value of 1.0 is perfect, while below 0.95 is inefficient.")
            fig_pf = px.line(data, x='Datetime', y='Average Power Factor', title="Corrected Power Factor Over Time")
            fig_pf.add_hline(y=0.95, line_dash="dash", line_color="red", annotation_text="Target PF (0.95)")
            st.plotly_chart(fig_pf, use_container_width=True)

        with tab3:
            st.subheader("Peak Demand Analysis")
            st.markdown("This chart shows the total power demand, corrected for potential wiring issues. The peak of this graph determines utility demand charges.")
            fig_demand = px.line(data, x='Datetime', y='Total Power Demand (kW)', title="Total Power Demand Profile")
            
            demand_data = data['Total Power Demand (kW)'].dropna()
            if not demand_data.empty:
                peak_index = demand_data.idxmax()
                peak_demand_time = data.loc[peak_index, 'Datetime']
                
                fig_demand.add_vline(x=peak_demand_time, line_dash="dash", line_color="red")
                fig_demand.add_annotation(x=peak_demand_time, y=max_demand_kw, text=f"Peak Demand: {max_demand_kw:.2f} kW",
                                          showarrow=True, arrowhead=1, yshift=10, bgcolor="rgba(255, 255, 255, 0.8)")
            st.plotly_chart(fig_demand, use_container_width=True)

        with tab4:
            st.subheader("Cleaned Raw Data Table")
            st.dataframe(data)

        with tab5:
            st.subheader("Measurement Summary Parameters")
            st.dataframe(parameters)

        with tab6:
            st.subheader("Variable Explanations")
            st.markdown("""
            - **Average Real Power (kW):** The 'useful' power performing work. *If this value is negative, it likely indicates a reversed CT clamp.*
            - **Average Apparent Power (kVA):** The total power supplied by the utility (Real + Reactive).
            - **Average Reactive Power (kVAR):** The 'wasted' power required for motors.
            - **Average Power Factor:** The ratio of Real to Apparent Power. A direct score of your electrical efficiency, corrected to always be positive.
            - **Consumed Real Energy (kWh):** The total energy you are billed for, corrected for wiring issues.
            - **Total Power Demand (kW):** The basis for peak demand charges, corrected for wiring issues.
            """)
    elif uploaded_file is not None:
         st.warning("Could not process the uploaded file. Please ensure it is a valid, non-empty Hioki CSV export.")

