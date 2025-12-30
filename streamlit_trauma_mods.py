
import numpy as np
import pandas as pd
import streamlit as st
import joblib
import shap
import matplotlib.pyplot as plt

# ==========================================
# 页面配置与样式
# ==========================================
st.set_page_config(
    page_title="MODS Prediction in Trauma Patients (RF Model)", 
    page_icon="🏥", 
    layout="wide"
)

st.markdown("""
<style>
    .main {
        padding: 2rem 3rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #0e4c92;
        color: white;
        font-weight: bold;
    }
    h1, h2, h3 {
        color: #0e4c92;
    }
    /* 调整输入区背景，使其更明显 */
    [data-testid="stSidebar"] {
        background-color: #f0f2f6;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 模型加载
# ==========================================
@st.cache_resource
def load_model():
    # 确保 'rf_model_new.joblib' 文件在同级目录下
    try:
        model = joblib.load('rf_mods_model.joblib')
        return model
    except FileNotFoundError:
        st.error("Error: Model file 'rf_model_new.joblib' not found.")
        return None

model = load_model()

# ==========================================
# 主界面
# ==========================================

st.title("🏥 Prediction of MODS in Trauma Patients")
st.markdown("Based on **Random Forest** algorithm")

col1, col2 = st.columns([1, 2])

# ==========================================
# 左侧：患者参数输入
# ==========================================
with col1:
    st.subheader("Patient Parameters")
    st.info("Enter clinical data (First 24h)")
    
    # 1. Age
    age = st.number_input("Age (years)", min_value=18, max_value=120, value=50, step=1)
    
    # 2. Temperature
    temp = st.slider("Max Temperature (°C)", 30.0, 45.0, 37.0, step=0.1)
    
    # 3. Systolic BP
    sbp = st.slider("Min Systolic BP (mmHg)", 40, 250, 110)
    
    # 4. Platelet Count
    platelets = st.slider("Min Platelet Count (x10^9/L)", 0, 1000, 200)
    
    # 5. BUN
    bun = st.number_input("Max BUN (mg/dL)", min_value=0.0, max_value=200.0, value=20.0, step=0.1)
    
    # 6. RISS (Revised Injury Severity Score)
    riss = st.slider("RISS Score", 0, 75, 15)
    
    # 7. SOFA Score (Total)
    sofa = st.slider("Total SOFA Score (1st Day)", 0, 24, 5)
    
    # 8. Renal SOFA Component (修正部分)
    # SOFA肾脏分项通常为0-4分
    renal = st.slider(
        "Renal SOFA Score (Component)", 
        min_value=0, 
        max_value=4, 
        value=0,
        help="0: Normal, 1-4: Increasing severity based on Creatinine/Urine output"
    )
    
    # 9. Invasive Line
    inv_line_input = st.selectbox("Invasive Line Used (1st Day)", ("No", "Yes"))
    invasive_line = 1 if inv_line_input == "Yes" else 0
    
    # 10. Mechanical Ventilation
    mech_vent_input = st.selectbox("Mechanical Ventilation", ("No", "Yes"))
    mech_vent = 1 if mech_vent_input == "Yes" else 0

    st.write("") # Spacer
    predict_btn = st.button("Predict Probability", key="predict")

# ==========================================
# 右侧：预测结果与解释
# ==========================================
with col2:
    if predict_btn and model is not None:
        # 构建输入数据 DataFrame
        # 必须严格保持训练时的特征顺序：
        # ['platelets_min', 'riss', 'sbp_min', 'bun_max', 'temperature_max', 'admission_age', 'renal', 'invasive_line_1stday', 'mechvent', 'sofa_1stday']
        
        features_dict = {
            'platelets_min': platelets,
            'riss': riss,
            'sbp_min': sbp,
            'bun_max': bun,
            'temperature_max': temp,
            'admission_age': age,
            'renal': renal,          # 这里现在是 0-4 的整数
            'invasive_line_1stday': invasive_line,
            'mechvent': mech_vent,
            'sofa_1stday': sofa
        }
        
        input_data = pd.DataFrame([features_dict])
        
        # 显示用的特征名映射
        display_names = {
            'platelets_min': 'Platelets',
            'riss': 'RISS',
            'sbp_min': 'Systolic BP',
            'bun_max': 'BUN',
            'temperature_max': 'Temp',
            'admission_age': 'Age',
            'renal': 'Renal SOFA',
            'invasive_line_1stday': 'Inv. Line',
            'mechvent': 'Mech. Vent',
            'sofa_1stday': 'Total SOFA'
        }

        try:
            # 预测
            prediction_prob = model.predict_proba(input_data)[0, 1]
            
            # 显示结果区域
            st.subheader("Prediction Result")
            
            # 颜色逻辑
            if prediction_prob < 0.2:
                color = "green"
                risk_text = "Low Risk"
                bg_color = "#e6f4ea"
            elif prediction_prob < 0.5:
                color = "#ffa500" # Orange
                risk_text = "Moderate Risk"
                bg_color = "#fff8e1"
            else:
                color = "#d93025" # Red
                risk_text = "High Risk"
                bg_color = "#fce8e6"

            # 结果卡片
            st.markdown(
                f"""
                <div style="background-color: {bg_color}; padding: 20px; border-radius: 10px; border: 1px solid #ddd; text-align: center; margin-bottom: 20px;">
                    <h3 style="margin:0; color: #555;">Probability of MODS (within 7d)</h3>
                    <h1 style="color: {color}; font-size: 56px; margin: 10px 0; font-weight: bold;">{prediction_prob:.2%}</h1>
                    <h4 style="color: #333;">Risk Level: <span style="color: {color};">{risk_text}</span></h4>
                </div>
                """, 
                unsafe_allow_html=True
            )
            
            # 解释部分
            st.subheader("Why this prediction?")
            st.markdown("The chart below shows how each feature contributed to pushing the risk **higher (red)** or **lower (blue)**.")

            with st.spinner("Analyzing model decision..."):
                # SHAP Explanation
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(input_data)
                
                # 处理 binary classification 的 shap_values 输出 (通常是 list[array, array])
                # 我们取 index 1 (positive class / MODS发生)
                if isinstance(shap_values, list):
                    shap_vals_target = shap_values[1]
                    base_value = explainer.expected_value[1]
                else:
                    shap_vals_target = shap_values
                    base_value = explainer.expected_value

                # 准备绘图数据
                # 将列名替换为易读名称
                input_data_display = input_data.rename(columns=display_names)
                
                exp = shap.Explanation(
                    values=shap_vals_target[0], 
                    base_values=base_value, 
                    data=input_data_display.iloc[0],
                    feature_names=input_data_display.columns
                )
                
                # 绘制 Waterfall Plot
                fig, ax = plt.subplots(figsize=(10, 5))
                shap.plots.waterfall(exp, show=False, max_display=10)
                st.pyplot(fig)
                plt.close(fig)

        except Exception as e:
            st.error(f"Prediction Error: {e}")
            st.info("Check if feature names/order match the trained model exactly.")

    elif not predict_btn:
        st.info("👈 Please configure patient parameters on the left sidebar/column.")

# ==========================================
# 底部 Disclaimer
# ==========================================
st.markdown("---")
st.warning("""
**DISCLAIMER**: This tool is for **research purposes only**. 
It uses a Random Forest model to estimate the risk of MODS based on first-24h data. 
Results should not replace clinical judgment.
""")
