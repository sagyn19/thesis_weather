import pandas as pd
import streamlit as st


def clean_data(df):
    """
    Cleans column names, builds the timestamp index, filters old data,
    combines redundant sensors, and interpolates small gaps.
    """
    # 1. Clean whitespace from column names (from your notebook)
    df.columns = df.columns.str.strip()

    # 2. Build the Timestamp (from your notebook)
    time_mapping = {'Year': 'year', 'Month': 'month', 'Date': 'day', 'Hour': 'hour', 'Minute': 'minute'}

    if not set(time_mapping.keys()).issubset(df.columns):
        st.error("❌ Uploaded CSV is missing required time columns (Year, Month, Date, Hour, Minute).")
        st.stop()

    df['Timestamp'] = pd.to_datetime(df[list(time_mapping.keys())].rename(columns=time_mapping))
    df = df.set_index('Timestamp').sort_index()

    # 3. Remove ghost data (from your notebook)
    df = df[df.index.year >= 2015]
    if df.empty:
        st.error("❌ Data error: No valid data found after 2015.")
        st.stop()

    # 4. Combine Sensors
    if 'CH1Temp (degC)' in df.columns and 'CH3Temp (degC)' in df.columns:
        df['Temperature'] = df[['CH1Temp (degC)', 'CH3Temp (degC)']].min(axis=1)

    if 'CH1RH (%)' in df.columns and 'CH3RH (%)' in df.columns:
        df['Humidity'] = df[['CH1RH (%)', 'CH3RH (%)']].mean(axis=1)

    # 5. Fix Small Gaps (Interpolate up to 3 hours, then drop broken rows)
    cols_to_fix = ['Temperature', 'Humidity', 'Baro (mBar/hPa)', 'Wspeed (mph)', 'Wgust (mph)']
    existing_cols = [c for c in cols_to_fix if c in df.columns]  # Safety check

    df[existing_cols] = df[existing_cols].interpolate(method='linear', limit=3)
    df = df.dropna(subset=['Temperature'])

    return df