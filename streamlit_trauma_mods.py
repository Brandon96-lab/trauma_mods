import numpy as np
import pandas as pd
import streamlit as st
import joblib

# =========================
# Page config
# =========================
st.set_page_config(
    page_title="MODS Risk Prediction (LR Model)",
    page_icon="🏥",
    layout="wide"
)

# =========================
# Load model
# =========================
@st.cache_resource
def load_model():
    model = joblib.load("mods_lr_model.joblib")
    scaler = joblib.load("mods_lr_scaler.joblib")
    return model, scaler

model, scaler = load_model()

# =========================
# Title
# =========================
st.title("🏥 Risk Prediction of MODS in Trauma Patients")
st.markdown("**Logistic Regression–Based Clinical Prediction Model**")

col1, col2 = st.columns([1, 2])

# =========================
# Inputs
# =========================
with col1:
    st.subheader("Patient Characteristics")

    platelets = st.slider("Platelet Count (×10⁹/L)", 0, 1000, 200)
    riss = st.slider("RISS", 0.0, 100.0, 25.0)
    sbp = st.slider("Systolic Blood Pressure (mmHg)", 40, 200, 110)
    bun = st.slider("BUN (mg/dL)", 1.0, 200.0, 20.0)
    temp = st.slider("Maximum Temperature (°C)", 34.0, 42.0, 37.5)
    age = st.slider("Age", 18, 100, 60)
    renal = st.slider("Renal Score", 0, 4, 0)
    invasive = st.selectbox("Invasive Line Use", ["No", "Yes"])
    mechvent = st.selectbox("Mechanical Ventilation", ["No", "Yes"])
    sofa = st.slider("SOFA Score (Day 1)", 0, 24, 6)

    if st.button("Predict MODS Risk"):
        # Input dataframe
        X_input = pd.DataFrame([[
            platelets, riss, sbp, bun, temp, age,
            renal,
            1 if invasive == "Yes" else 0,
            1 if mechvent == "Yes" else 0,
            sofa
        ]], columns=[
            'platelets_min',
            'riss',
            'sbp_min',
            'bun_max',
            'temperature_max',
            'admission_age',
            'renal',
            'invasive_line_1stday',
            'mechvent',
            'sofa_1stday'
        ])

        X_scaled = scaler.transform(X_input)
        prob = model.predict_proba(X_scaled)[0, 1]

        with col2:
            st.subheader("Predicted Risk")

            st.markdown(
                f"<h2 style='text-align:center;'>"
                f"Predicted Probability of MODS: "
                f"<span style='color:{'red' if prob >= 0.5 else 'green'};'>"
                f"{prob:.1%}</span></h2>",
                unsafe_allow_html=True
            )

            if prob < 0.2:
                st.success("Low Risk")
            elif prob < 0.5:
                st.warning("Moderate Risk")
            else:
                st.error("High Risk")

# =========================
# Disclaimer
# =========================
st.markdown("---")
st.warning("""
**DISCLAIMER**

This tool is intended for **research use only**.  
It should not replace clinical judgment or decision-making.

Further external and prospective validation is required.
""")
