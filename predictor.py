import joblib
import streamlit as st


@st.cache_resource
def load_model(model_choice, target_choice):
    """Loads the specific model based on what the user wants to predict."""
    try:
        if model_choice == "XGBoost":
            if target_choice == "Temperature":
                return joblib.load('models/xgb_model_temp.pkl')
            else:
                return joblib.load('models/xgb_model_pres.pkl')
        else:
            # If you also train linear regression models, you would load them here
            if target_choice == "Temperature":
                return joblib.load('linear_temp.pkl')
            else:
                return joblib.load('linear_pressure.pkl')

    except FileNotFoundError:
        st.error(f"❌ Error: Could not find the saved model file. Please ensure your `.pkl` files are in the folder!")
        st.stop()


def make_prediction(final_row, model_choice, target_choice):
    model = load_model(model_choice, target_choice)
    prediction = model.predict(final_row)
    return prediction[0]