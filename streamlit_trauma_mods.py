

import numpy as np
import pandas as pd
import streamlit as st
import joblib
import shap
import matplotlib.pyplot as plt

# ======================
# Page config
# ======================
st.set_page_config(
    page_title="MODS Prediction (XGBoost)",
    page_icon="🏥",
    layout="wide"
)

# ======================
# Custom CSS
# ======================
st.markdown("""
<style>
    .main {
        padding: 2rem 3rem;
    }
    h1, h2, h3 {
        color: #0e4c92;
    }
    .stButton>button {
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

# ======================
# Load model
# ======================
@st.cache_resource
def load_model():
    return joblib.load("xgb_mods_model.joblib")

model = load_model()

# ======================
# Feature definition
# ======================
FEATURES = [
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
]

FEATURES_DISPLAY = [
    "Platelet Count (×10⁹/L)",
    "RISS",
    "Systolic BP (mmHg)",
    "BUN (mg/dL)",
    "Temperature (°C)",
    "Age (years)",
    "Renal Score",
    "Invasive Line Use",
    "Mechanical Ventilation",
    "SOFA Score"
]

# ======================
# Title
# ======================
st.title("🏥 MODS Prediction in Trauma Patients with Sepsis")
st.markdown("### XGBoost-based Clinical Prediction Model")

col1, col2 = st.columns([1, 2])

# ======================
# Input panel
# ======================
with col1:
    st.subheader("Patient Parameters")

    platelets = st.slider(FEATURES_DISPLAY[0], 0, 1000, 200)
    riss = st.slider(FEATURES_DISPLAY[1], 0, 75, 25)
    sbp = st.slider(FEATURES_DISPLAY[2], 50, 200, 110)
    bun = st.slider(FEATURES_DISPLAY[3], 0, 200, 20)
    temp = st.slider(FEATURES_DISPLAY[4], 34.0, 42.0, 37.0, step=0.1)
    age = st.slider(FEATURES_DISPLAY[5], 18, 100, 60)
    renal = st.slider(FEATURES_DISPLAY[6], 0, 4, 0)
    invasive = st.selectbox(FEATURES_DISPLAY[7], ["No", "Yes"])
    mechvent = st.selectbox(FEATURES_DISPLAY[8], ["No", "Yes"])
    sofa = st.slider(FEATURES_DISPLAY[9], 0, 24, 6)

    if st.button("Predict MODS Risk"):
        input_data = pd.DataFrame([[
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
        ]], columns=FEATURES)

        pred_prob = model.predict_proba(input_data)[0, 1]

        # ======================
        # Output
        # ======================
        with col2:
            st.subheader("Prediction Result")

            color = "red" if pred_prob >= 0.5 else "green"
            st.markdown(
                f"<h2 style='text-align:center;color:{color};'>"
                f"MODS Probability: {pred_prob:.2%}</h2>",
                unsafe_allow_html=True
            )

            if pred_prob < 0.1:
                st.success("Low Risk of MODS")
            elif pred_prob < 0.5:
                st.warning("Moderate Risk of MODS")
            else:
                st.error("High Risk of MODS")

            # ======================
            # SHAP explanation
            # ======================
            st.subheader("Model Explanation (SHAP)")

            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(input_data)

            fig, ax = plt.subplots(figsize=(8, 5))
            shap.summary_plot(
                shap_values,
                input_data,
                plot_type="bar",
                show=False
            )
            st.pyplot(fig)
            plt.close(fig)

# ======================
# Disclaimer
# ======================
st.markdown("---")
st.warning("""
**DISCLAIMER**

This prediction tool is developed for research purposes only.
It should not be used as the sole basis for clinical decision-making.

Always consult qualified healthcare professionals before making
diagnostic or treatment decisions.
""")

# ======================
# Footer
# ======================
st.markdown(
    "<div style='text-align:center;color:gray;'>"
    "© 2025 MODS Prediction Model | XGBoost | Research Use Only"
    "</div>",
    unsafe_allow_html=True
)

