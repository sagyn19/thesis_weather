import pandas as pd
import numpy as np
import streamlit as st


def process_features(df):
    """
    Applies the exact physics, resampling, and feature engineering
    from the Master's thesis Jupyter Notebook.
    """
    df = df.copy()

    # --- 1. PRE-PROCESSING ---
    df['Wgust (mph)'] = df['Wgust (mph)'].fillna(0)
    df['RainCount (mm)'] = df['RainCount (mm)'].fillna(0)
    df['Wspeed (mph)'] = df['Wspeed (mph)'].interpolate(method='time')

    # Sun Spikes & Sensor Fusion
    t1, t3 = 'CH1Temp (degC)', 'CH3Temp (degC)'
    h1, h3 = 'CH1RH (%)', 'CH3RH (%)'

    df.loc[df[t1] > 45, t1] = np.nan
    df.loc[df[t3] > 45, t3] = np.nan

    df['Temperature'] = df[[t1, t3]].min(axis=1)
    df['Humidity'] = df[h1].fillna(df[h3])

    df['Temperature'] = df['Temperature'].interpolate(method='time', limit_direction='both')
    df['Humidity'] = df['Humidity'].interpolate(method='time', limit_direction='both')

    # Wind Vectorization
    wd_rad = df['W_Dir'] * np.pi / 180
    df['Wind_Sin'] = np.sin(wd_rad)
    df['Wind_Cos'] = np.cos(wd_rad)

    # --- 2. RESAMPLING TO HOURLY ---
    agg_rules = {
        'Temperature': 'mean', 'Humidity': 'mean', 'Baro (mBar/hPa)': 'mean',
        'Wspeed (mph)': 'mean', 'Wgust (mph)': 'max', 'RainCount (mm)': 'sum',
        'Wind_Sin': 'mean', 'Wind_Cos': 'mean'
    }

    df_hourly = df.resample('1h').agg(agg_rules)

    df_hourly.rename(columns={
        'Baro (mBar/hPa)': 'Pressure', 'Wspeed (mph)': 'WindSpeed',
        'Wgust (mph)': 'WindGust', 'RainCount (mm)': 'Rain_1h'
    }, inplace=True)

    df_hourly = df_hourly.interpolate(method='time', limit=2).dropna()

    if df_hourly.empty:
        st.error("❌ Not enough valid data left after processing. Please upload a longer date range.")
        st.stop()

    # --- 3. FEATURE ENGINEERING ---
    # A. Time Components
    df_hourly['Hour_Sin'] = np.sin(2 * np.pi * df_hourly.index.hour / 24)
    df_hourly['Hour_Cos'] = np.cos(2 * np.pi * df_hourly.index.hour / 24)
    df_hourly['Day_Sin'] = np.sin(2 * np.pi * df_hourly.index.dayofyear / 365.25)
    df_hourly['Day_Cos'] = np.cos(2 * np.pi * df_hourly.index.dayofyear / 365.25)

    # B. Lags & Windows
    df_hourly['Temp_Lag_24h'] = df_hourly['Temperature'].shift(24)
    df_hourly['Pressure_Lag_24h'] = df_hourly['Pressure'].shift(24)

    df_hourly['Rolling_Mean_24h'] = df_hourly['Temperature'].rolling(window=24).mean()
    df_hourly['Rolling_Min_24h'] = df_hourly['Temperature'].rolling(window=24).min()
    df_hourly['Rolling_Max_24h'] = df_hourly['Temperature'].rolling(window=24).max()

    df_hourly['Rain_Last_24h'] = df_hourly['Rain_1h'].rolling(window=24).sum()
    df_hourly.drop(columns=['Rain_1h'], inplace=True)

    df_hourly['Pressure_Delta_3h'] = df_hourly['Pressure'].diff(3).fillna(0)

    # C. Dew Point (Magnus Formula)
    a, b = 17.27, 237.7
    rh = df_hourly['Humidity'].clip(lower=0.1)
    alpha = ((a * df_hourly['Temperature']) / (b + df_hourly['Temperature'])) + np.log(rh / 100.0)
    df_hourly['Dew_Point'] = (b * alpha) / (a - alpha)

    # 4. FINAL EXTRACTION
    # Drop rows where lags couldn't be calculated (the first 24 hours of the file)
    df_hourly.dropna(inplace=True)

    # Return ONLY the very last row (The "Right Now" moment to predict from)
    return df_hourly.tail(1)