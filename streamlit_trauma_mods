
import numpy as np
import pandas as pd
import streamlit as st
import joblib

# =========================
# Page configuration
# =========================
st.set_page_config(
    page_title="MODS Prediction (New Model)",
    page_icon="🏥",
    layout="wide"
)

# =========================
# Custom CSS
# =========================
st.markdown("""
<style>
.main { padding: 2rem 3rem; }
.stButton>button { width: 100%; }
h1, h2, h3 { color: #0e4c92; }
</style>
""", unsafe_allow_html=True)

# =========================
# Load model & scaler
# =========================
@st.cache_resource
def load_model():
    model = joblib.load("mods_ensemble_model.joblib")
    scaler = joblib.load("mods_scaler.joblib")
    return model, scaler

model, scaler = load_model()

# =========================
# Title
# =========================
st.title("🏥 Prediction of MODS within 7 Days")
st.markdown("**Trauma Patients with Sepsis – New Machine Learning Model**")

# =========================
# Layout
# =========================
col1, col2 = st.columns([1, 2])

# =========================
# Input features
# =========================
with col1:
    st.subheader("Patient Parameters")

    platelets = st.slider("Platelet Count (×10⁹/L)", 0, 1000, 200)
    riss = st.slider("RISS", 0.0, 100.0, 25.0)
    sbp = st.slider("Systolic Blood Pressure (mmHg)", 40, 200, 110)
    bun = st.slider("BUN (mg/dL)", 1.0, 200.0, 20.0)
    temp = st.slider("Maximum Temperature (°C)", 34.0, 42.0, 37.5)
    age = st.slider("Age", 18, 100, 60)
    renal = st.slider("Renal Score", 0, 4, 0)
    invasive = st.selectbox("Invasive Line Use", ["No", "Yes"])
    mechvent = st.selectbox("Mechanical Ventilation", ["No", "Yes"])
    sofa = st.slider("SOFA Score (1st day)", 0, 24, 6)

    if st.button("Predict MODS Risk"):
        # =========================
        # Prepare input
        # =========================
        input_df = pd.DataFrame([[
            platelets,
            riss,
            sbp,
            bun,
            temp,
            age,
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

        # Scale input
        input_scaled = scaler.transform(input_df)

        # Predict
        prob = model.predict_proba(input_scaled)[0, 1]

        # =========================
        # Output
        # =========================
        with col2:
            st.subheader("Prediction Result")

            st.markdown(
                f"<h2 style='text-align:center;'>"
                f"Probability of MODS: "
                f"<span style='color:{'red' if prob >= 0.5 else 'green'};'>"
                f"{prob:.2%}</span></h2>",
                unsafe_allow_html=True
            )

            if prob < 0.2:
                st.success("Low Risk of MODS")
            elif prob < 0.5:
                st.warning("Moderate Risk of MODS")
            else:
                st.error("High Risk of MODS")

# =========================
# Disclaimer
# =========================
st.markdown("---")
st.warning("""
**DISCLAIMER**

This prediction tool is developed for **research purposes only**.  
It should **NOT** be used as the sole basis for clinical decision-making.

- The model is intended to support, not replace, clinician judgment.
- External and prospective validation are required before clinical deployment.
- Always consult qualified healthcare professionals.
""")

st.markdown(
    "<div style='text-align:center; color:#666;'>© 2025 MODS Prediction Model</div>",
    unsafe_allow_html=True
)
