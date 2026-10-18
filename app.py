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

    # --- 6. PREDICTION TARGET ---
    st.write("### ⏱️ Choose Prediction Target")
    available_times = df['Timestamp'].dt.strftime('%Y-%m-%d %H:%M').tolist()
    selected_time_str = st.selectbox("Predict the weather 24h after:", available_times, index=len(available_times) - 1)

    # --- 7. EXECUTE ---
    if st.button(f"Predict {target_choice} 🚀"):
        selected_time = pd.to_datetime(selected_time_str)
        sliced_df = df[df['Timestamp'] <= selected_time].copy()
        sliced_df = sliced_df.set_index('Timestamp')

        st.success(f"Processing data up to {selected_time_str}...")

        # --- THE BACKEND ---
        final_row = process_features(sliced_df)
        if final_row.empty:
            st.error(
                "❌ Not enough historical data. The model needs at least 24 hours of prior weather data to calculate its features. Please select a later starting time.")
            st.stop()

        prediction = make_prediction(final_row, model_choice, target_choice)

        target_time = selected_time + pd.Timedelta(hours=24)

        # Set the correct unit symbol
        unit = "°C" if target_choice == "Temperature" else "hPa"
        st.metric(label=f"{target_choice} Forecast for {target_time.strftime('%Y-%m-%d %H:%M')}", value=f"{prediction:.2f} {unit}")