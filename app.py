import streamlit as st
import pandas as pd

from features import process_features
from predictor import make_prediction

# We will uncomment these as we finish the final steps!
# from features import process_features
# from predictor import make_prediction

st.set_page_config(page_title="Weather Predictor", page_icon="🌤️")
st.title("🌤️ Local Weather Forecaster")

# --- 1. STRICT DATA INSTRUCTIONS ---
REQUIRED_FEATURES = [
    'CH1Temp (degC)', 'CH3Temp (degC)', 'CH1RH (%)', 'CH3RH (%)',
    'Baro (mBar/hPa)', 'Wgust (mph)', 'Wspeed (mph)', 'W_Dir', 'RainCount (mm)'
]

st.sidebar.header("📋 Data Requirements")
st.sidebar.info("Your CSV must contain Date/Time data, plus exactly these features:")
for feature in REQUIRED_FEATURES:
    st.sidebar.markdown(f"- `{feature}`")

st.sidebar.header("Configuration")
model_choice = st.sidebar.selectbox("Select Model", ["XGBoost", "Linear Regression"])
target_choice = st.sidebar.radio("What do you want to predict?", ["Temperature", "Pressure"])

# --- 2. FILE UPLOAD ---
uploaded_file = st.file_uploader("Upload Weather CSV", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    df.columns = df.columns.str.strip()

    # --- 3. SMART TIME HANDLING (Dual-Mode) ---
    time_mapping = {'Year': 'year', 'Month': 'month', 'Date': 'day', 'Hour': 'hour', 'Minute': 'minute'}

    if set(time_mapping.keys()).issubset(df.columns):
        # MODE A: Auto-stitch split time columns
        df['Timestamp'] = pd.to_datetime(df[list(time_mapping.keys())].rename(columns=time_mapping))
        st.success("✨ Auto-stitched your split time columns!")
    else:
        # MODE B: Single-column format
        st.info("We need to locate your timeline.")
        time_col = st.selectbox("Which column contains the Date/Time?", df.columns)
        try:
            df['Timestamp'] = pd.to_datetime(df[time_col])
        except Exception:
            st.error("❌ Could not read the dates in that column. Please select a valid Date/Time column.")
            st.stop()

    # --- 4. THE POLITE RECEPTIONIST (Interactive Column Mapping) ---
    missing_cols = [col for col in REQUIRED_FEATURES if col not in df.columns]

    if missing_cols:
        st.warning("⚠️ Your CSV uses different column names. Please match them below:")

        rename_map = {}
        # Exclude 'Timestamp' from the dropdown so the user doesn't accidentally pick it
        available_cols = [c for c in df.columns if c != 'Timestamp']

        for required_col in missing_cols:
            user_choice = st.selectbox(f"Which column is '{required_col}'?", ["Select..."] + available_cols)
            if user_choice != "Select...":
                rename_map[user_choice] = required_col

        # Automatically rename the user's columns to match your XGBoost model
        df = df.rename(columns=rename_map)

        # Final safety check
        still_missing = [col for col in REQUIRED_FEATURES if col not in df.columns]
        if still_missing:
            st.error("Please map all required columns to continue.")
            st.stop()

    # --- 5. PREPARE THE DATA ---
    # Strip away any extra junk columns, keeping ONLY the Timestamp and what we need
    df = df[['Timestamp'] + REQUIRED_FEATURES].sort_values(by='Timestamp')

    st.write("### ✅ Data Accepted & Filtered")
    st.dataframe(df.tail(3))

        # --- 6. FORECAST SETTINGS ---
    st.write("### ⏱️ Forecast Settings")
    available_times = df['Timestamp'].dt.strftime('%Y-%m-%d %H:%M').tolist()

    col1, col2 = st.columns(2)
    with col1:
        selected_time_str = st.selectbox("Start Forecast From:", available_times, index=len(available_times) - 1)
    with col2:
        # Let the user choose how far into the future they want to predict
        forecast_horizon = st.slider("Forecast Horizon (Hours):", min_value=1, max_value=24, value=12)

    # --- 7. RECURSIVE FORECASTING ENGINE ---
    if st.button(f"Predict {forecast_horizon} Hours Ahead 🚀"):
        selected_time = pd.to_datetime(selected_time_str)
        
        # 1. Isolate the historical seed data
        historical_df = df[df['Timestamp'] <= selected_time].copy()
        historical_df = historical_df.set_index('Timestamp')
        
        st.success(f"Processing data and predicting {forecast_horizon} hours into the future...")
        
        # Create a progress bar for the UI
        progress_text = "Running recursive inference loop..."
        my_bar = st.progress(0, text=progress_text)
        
        # 2. Setup the loop variables
        future_dates = []
        predictions = []
        working_df = historical_df.copy()
        
        # 3. The Recursive Loop
        for step in range(forecast_horizon):
            # A. Process features for the current timeline
            final_row = process_features(working_df)
            
            if final_row.empty:
                st.error("❌ Not enough historical data. The model needs at least 24 hours of prior weather data.")
                st.stop()
                
            # B. Predict the very next hour
            pred_val = make_prediction(final_row, model_choice, target_choice)
            predictions.append(pred_val)
            
            # C. Calculate the timestamp for this new prediction
            next_time = working_df.index.max() + pd.Timedelta(hours=1)
            future_dates.append(next_time)
            
            # D. The "Springboard" Trick: Create a dummy row to feed back into the model
            # We copy the last known raw weather conditions, but overwrite the target variable with our new prediction!
            new_row = working_df.iloc[-1].copy()
            new_row.name = next_time
            
            if target_choice == "Temperature":
                new_row['CH1Temp (degC)'] = pred_val
                new_row['CH3Temp (degC)'] = pred_val
            else:
                new_row['Baro (mBar/hPa)'] = pred_val
                
            # Append new hybrid row back into our working dataframe for the next loop iteration
            working_df.loc[next_time] = new_row
            
            # Update UI progress
            my_bar.progress((step + 1) / forecast_horizon, text=f"Predicted Hour {step+1} of {forecast_horizon}...")
            
        my_bar.empty() # Clear the progress bar when done
        
        # --- 8. VISUALIZATION ---
        st.write("### 📈 Forecast Results")
        
        # Determine which raw column to pull actuals from for the chart
        raw_col = 'CH1Temp (degC)' if target_choice == "Temperature" else 'Baro (mBar/hPa)'
        
        # Get the last 48 hours of actual historical data to anchor the graph
        actuals = historical_df[[raw_col]].resample('1h').mean().tail(48)
        actuals.columns = ['Actual Historical']
        
        # Create a dataframe for the future predictions
        forecast_df = pd.DataFrame({'Model Forecast': predictions}, index=future_dates)
        
        # Combine them so they plot on the same graph seamlessly
        chart_df = pd.concat([actuals, forecast_df])
        
        # Render the chart in Streamlit
        st.line_chart(chart_df)
        
        # Output the final target metric
        target_time = future_dates[-1]
        unit = "°C" if target_choice == "Temperature" else "hPa"
        st.metric(label=f"Final Forecasted {target_choice} at {target_time.strftime('%Y-%m-%d %H:%M')}", value=f"{predictions[-1]:.2f} {unit}")
else:
    st.info("Please upload a CSV file to begin forecasting.")