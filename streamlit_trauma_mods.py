
import numpy as np
import pandas as pd
import streamlit as st
import joblib

# ===============================
# Page configuration
# ===============================
st.set_page_config(
    page_title="MODS Prediction (RF Model)",
    page_icon="🏥",
    layout="wide"
)

# ===============================
# Custom CSS
# ===============================
st.markdown("""
<style>
    h1, h2, h3 {
        color: #0e4c92;
    }
    .stButton>button {
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

# ===============================
# Load model
# ===============================
@st.cache_resource
def load_model():
    return joblib.load("rf_mods_model.joblib")

model = load_model()

# ===============================
# Title
# ===============================
st.title("🏥 MODS Prediction in Trauma Patients with Sepsis")
st.caption("Random Forest–based prediction model")

# ===============================
# Layout
# ===============================
col1, col2 = st.columns([1, 2])

# ===============================
# Input section
# ===============================
with col1:
    st.subheader("Patient Parameters")

    platelets = st.slider("Platelet Count (×10⁹/L)", 0, 1000, 200)
    riss = st.slider("RISS", 0, 100, 25)
    sbp = st.slider("Systolic Blood Pressure (mmHg)", 40, 200, 110)
    bun = st.slider("BUN (mg/dL)", 0, 200, 20)
    temp = st.slider("Temperature (°C)", 30.0, 43.0, 37.0, step=0.1)
    age = st.slider("Age", 18, 100, 60)
    renal = st.slider("Renal Score", 0, 4, 0)
    invasive_line = st.selectbox("Invasive Line (Day 1)", ["No", "Yes"])
    mechvent = st.selectbox("Mechanical Ventilation", ["No", "Yes"])
    sofa = st.slider("SOFA Score (Day 1)", 0, 24, 6)

    predict_button = st.button("Predict MODS Risk")

# ===============================
# Prediction
# ===============================
if predict_button:
    # Prepare input
    input_data = pd.DataFrame([[
        platelets,
        riss,
        sbp,
        bun,
        temp,
        age,
        renal,
        1 if invasive_line == "Yes" else 0,
        1 if mechvent == "Yes" else 0,
        sofa
    ]], columns=[
        "platelets_min",
        "riss",
        "sbp_min",
        "bun_max",
        "temperature_max",
        "admission_age",
        "renal",
        "invasive_line_1stday",
        "mechvent",
        "sofa_1stday"
    ])

    # Predict probability
    prob = model.predict_proba(input_data)[0, 1]

    with col2:
        st.subheader("Prediction Result")

        st.markdown(
            f"""
            <h2 style='text-align:center;'>
            MODS Probability: 
            <span style='color:{'red' if prob >= 0.5 else 'green'}'>
            {prob:.2%}
            </span>
            </h2>
            """,
            unsafe_allow_html=True
        )

        if prob < 0.10:
            st.success("Low Risk of MODS")
        elif prob < 0.50:
            st.warning("Moderate Risk of MODS")
        else:
            st.error("High Risk of MODS")

# ===============================
# Disclaimer
# ===============================
st.markdown("---")
st.warning("""
**DISCLAIMER**

This tool is intended for research purposes only.

- The prediction is based on a Random Forest model trained on retrospective data.
- It should NOT be used as the sole basis for clinical decisions.
- Further external and prospective validation is required.

Always consult qualified healthcare professionals.
""")

# ===============================
# Footer
# ===============================
st.markdown(
    "<p style='text-align:center;color:gray;'>© 2024 MODS Prediction Model | Research Use Only</p>",
    unsafe_allow_html=True
)
