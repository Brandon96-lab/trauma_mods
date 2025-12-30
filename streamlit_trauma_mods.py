
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
    page_title="MODS Prediction in Trauma Patients (New Model)", 
    page_icon="🏥", 
    layout="wide"
)

# 自定义CSS，保持与之前一致的风格
st.markdown("""
<style>
    .main {
        padding: 2rem 3rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #0e4c92;
        color: white;
    }
    .stAlert {
        padding: 1rem;
        margin-bottom: 1rem;
    }
    h1, h2, h3 {
        color: #0e4c92;
    }
    /* 调整Sidebar背景色 (可选) */
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
    # 请确保将训练好的模型文件 'rf_model_new.joblib' 放在同级目录下
    try:
        model = joblib.load('rf_mods_model.joblib')
        return model
    except FileNotFoundError:
        st.error("Error: Model file 'rf_model_new.joblib' not found. Please upload the model.")
        return None

model = load_model()

# ==========================================
# 主界面
# ==========================================

st.title("🏥 Prediction of MODS in Trauma Patients")
st.markdown("Based on the **Random Forest** algorithm (New Model)")

# 创建布局
col1, col2 = st.columns([1, 2])

# ==========================================
# 左侧：患者参数输入 (Input)
# ==========================================
with col1:
    st.subheader("Patient Parameters")
    st.info("Please enter the patient's clinical data obtained within the first 24h.")
    
    # 按照 feature_name_tot_RE 和 label_features_dict 的顺序对应设计输入框
    
    # 1. Age (admission_age)
    age = st.number_input("Age (years)", min_value=18, max_value=120, value=50, step=1)
    
    # 2. Temperature (temperature_max)
    temp = st.slider("Max Temperature (°C)", 30.0, 45.0, 37.0, step=0.1)
    
    # 3. Systolic BP (sbp_min)
    sbp = st.slider("Min Systolic BP (mmHg)", 40, 250, 110)
    
    # 4. Platelet Count (platelets_min)
    platelets = st.slider("Min Platelet Count (x10^9/L)", 0, 1000, 200)
    
    # 5. BUN (bun_max)
    bun = st.number_input("Max BUN (mg/dL)", min_value=0.0, max_value=200.0, value=20.0, step=0.1)
    
    # 6. RISS (riss) - Revised Injury Severity Score
    riss = st.slider("RISS Score", 0, 75, 15)
    
    # 7. SOFA Score (sofa_1stday)
    sofa = st.slider("SOFA Score (1st Day)", 0, 24, 5)
    
    # 8. Renal Comorbidity/Score (renal)
    # 根据变量名推测可能是肾脏疾病史或肾脏评分。这里设置为二分类（有无肾脏疾病史）
    # 如果原数据是SOFA肾脏分项(0-4)，请改为 slider
    renal_input = st.selectbox("Renal Comorbidity / History", ("No", "Yes"))
    renal = 1 if renal_input == "Yes" else 0
    
    # 9. Invasive Line (invasive_line_1stday)
    inv_line_input = st.selectbox("Invasive Line Used (1st Day)", ("No", "Yes"))
    invasive_line = 1 if inv_line_input == "Yes" else 0
    
    # 10. Mechanical Ventilation (mechvent)
    mech_vent_input = st.selectbox("Mechanical Ventilation", ("No", "Yes"))
    mech_vent = 1 if mech_vent_input == "Yes" else 0

    # 预测按钮
    predict_btn = st.button("Predict Probability", key="predict")

# ==========================================
# 右侧：预测结果与解释 (Output)
# ==========================================
with col2:
    if predict_btn and model is not None:
        # 1. 数据预处理
        # 必须严格按照训练时的特征顺序排列
        # 特征列表：['platelets_min', 'riss', 'sbp_min', 'bun_max', 'temperature_max', 'admission_age', 'renal', 'invasive_line_1stday', 'mechvent', 'sofa_1stday']
        
        input_data = pd.DataFrame([[
            platelets,      # platelets_min
            riss,           # riss
            sbp,            # sbp_min
            bun,            # bun_max
            temp,           # temperature_max
            age,            # admission_age
            renal,          # renal
            invasive_line,  # invasive_line_1stday
            mech_vent,      # mechvent
            sofa            # sofa_1stday
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
        
        # 显示友好的特征名称供展示用
        display_names = {
            'platelets_min': 'Platelet Count',
            'riss': 'RISS',
            'sbp_min': 'Systolic BP',
            'bun_max': 'BUN',
            'temperature_max': 'Temperature',
            'admission_age': 'Age',
            'renal': 'Renal Hx',
            'invasive_line_1stday': 'Inv. Line',
            'mechvent': 'Mech. Vent',
            'sofa_1stday': 'SOFA Score'
        }

        # 2. 进行预测
        try:
            # 预测概率
            prediction_prob = model.predict_proba(input_data)[0, 1]
            
            # 3. 显示预测结果
            st.subheader("Prediction Result")
            
            # 动态颜色设置
            color = "green"
            risk_label = "Low Risk"
            if prediction_prob >= 0.5:
                color = "red"
                risk_label = "High Risk"
            elif prediction_prob >= 0.2:
                color = "#ffcc00" # Orange/Yellow
                risk_label = "Moderate Risk"

            st.markdown(
                f"""
                <div style="background-color: #f9f9f9; padding: 20px; border-radius: 10px; border: 1px solid #ddd; text-align: center;">
                    <h3 style="margin:0;">Probability of MODS (within 7d)</h3>
                    <h1 style="color: {color}; font-size: 48px; margin: 10px 0;">{prediction_prob:.2%}</h1>
                    <h4 style="color: #555;">Risk Level: <b>{risk_label}</b></h4>
                </div>
                """, 
                unsafe_allow_html=True
            )
            
            # 阈值提示
            if prediction_prob < 0.2:
                st.success("The model predicts a low probability of developing MODS.")
            elif prediction_prob < 0.5:
                st.warning("The model predicts a moderate probability. Clinical monitoring advised.")
            else:
                st.error("The model predicts a high probability. Intensive monitoring required.")

            # 4. SHAP 解释 (Feature Importance)
            st.markdown("---")
            st.subheader("Model Explanation (SHAP)")
            
            with st.spinner("Calculating feature importance..."):
                # 创建解释器 (Random Forest 使用 TreeExplainer)
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(input_data)
                
                # 处理 SHAP 值的格式 (Binary classification usually returns a list of arrays)
                # shap_values[1] 对应 positive class (MODS=1)
                shap_val_target = shap_values[1] if isinstance(shap_values, list) else shap_values

                # --- Visualization 1: Waterfall Plot (Force Plot 的现代替代品) ---
                st.write("**Why did the model make this prediction?**")
                
                # 为了绘图，我们需要把 input_data 的列名改成人类可读的
                input_data_display = input_data.rename(columns=display_names)
                
                # 创建 SHAP Explanation 对象 (新版 SHAP 推荐用法)
                exp = shap.Explanation(
                    values=shap_val_target[0], 
                    base_values=explainer.expected_value[1], 
                    data=input_data_display.iloc[0],
                    feature_names=input_data_display.columns
                )
                
                fig_waterfall, ax = plt.subplots(figsize=(10, 5))
                shap.plots.waterfall(exp, show=False)
                st.pyplot(fig_waterfall)
                plt.close(fig_waterfall)
                
                # --- Visualization 2: Summary Plot (Bar Chart) ---
                # 既然是单样本预测，Bar chart 也就是显示绝对值大小
                # st.write("**Feature Impact Magnitude**")
                # fig_bar, ax = plt.subplots(figsize=(8, 4))
                # shap.plots.bar(exp, show=False)
                # st.pyplot(fig_bar)
                # plt.close(fig_bar)

        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
            st.write("Debug info - Shape mismatch or feature name mismatch likely.")

    elif not predict_btn:
        st.info("👈 Adjust patient parameters on the left and click 'Predict'.")

# ==========================================
# 底部声明
# ==========================================
st.markdown("---")
st.warning("""
**DISCLAIMER:**

This online calculator utilizes a machine learning model (**Random Forest**) trained on clinical data to predict the risk of Multiple Organ Dysfunction Syndrome (MODS). 

**Key Limitations & Usage:**
- **Research Use Only:** This tool is not FDA approved and is intended for educational and research validation purposes only.
- **Consult Professionals:** Never disregard professional medical advice or delay seeking it because of something you have read on this website.
- **Model Context:** The model was validated on specific datasets; performance may vary in different populations.
""")

st.markdown("""
<div style='text-align: center; color: #888; font-size: 0.8em;'>
    <p>© 2024 MODS Prediction Research Group</p>
</div>
""", unsafe_allow_html=True)
