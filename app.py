import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import pickle
import shap
import tensorflow as tf
from sklearn.metrics import roc_curve, auc

# --- PAGE CONFIG ---
st.set_page_config(page_title="Diabetes Predictor", layout="wide", page_icon="🩺")

# --- LOAD ASSETS ---
@st.cache_resource
def load_assets():
    # Loading all artifacts exported from your notebook
    # Ensure these files are in the same folder as app.py
    model = tf.keras.models.load_model("diabetes_model.keras")
    scaler = joblib.load("scaler.pkl")
    X_test_scaled = np.load("X_test_scaled.npy")
    with open("history.pkl", "rb") as f:
        history_dict = pickle.load(f)
    y_test = np.load("y_test.npy")
    y_prob = np.load("y_prob.npy")
    return model, scaler, X_test_scaled, history_dict, y_test, y_prob

try:
    # Unpacking all 6 items returned by load_assets
    model, scaler, X_test_scaled, history, y_test, y_prob = load_assets()
except Exception as e:
    st.error(f"⚠️ Error loading files: {e}")
    st.info("Check that 'diabetes_model.keras', 'scaler.pkl', 'X_test_scaled.npy', 'history.pkl', 'y_test.npy', and 'y_prob.npy' exist in this folder.")
    st.stop()

# --- SIDEBAR INPUTS (SLIDERS) ---
st.sidebar.header("📋 Patient Health Metrics")
st.sidebar.markdown("Adjust sliders to match the patient's data.")

def get_user_input():
    # Ranges based on PIMA dataset standards
    preg = st.sidebar.slider("Pregnancies", 0, 17, 3)
    plas = st.sidebar.slider("Glucose (2-hr test)", 0, 200, 117)
    pres = st.sidebar.slider("Blood Pressure (mm Hg)", 0, 122, 72)
    skin = st.sidebar.slider("Skin Thickness (mm)", 0, 99, 23)
    test = st.sidebar.slider("Insulin (mu U/ml)", 0, 846, 30)
    bmi  = st.sidebar.slider("BMI (Weight/Height^2)", 0.0, 67.1, 32.0)
    pedi = st.sidebar.slider("Diabetes Pedigree Function", 0.0, 2.42, 0.37)
    age  = st.sidebar.slider("Age (Years)", 21, 81, 29)
    
    data = {'Preg': preg, 'Plas': plas, 'Pres': pres, 'Skin': skin,
            'Test': test, 'BMI': bmi, 'Pedigree': pedi, 'Age': age}
    return pd.DataFrame(data, index=[0])

input_df = get_user_input()

# --- MAIN PAGE ---
st.title("🩺 Diabetes Risk Deep Learning Analysis")
st.markdown("""
This application utilizes a **Neural Network** (50-50-2 architecture with L1 Regularization) 
to predict the likelihood of diabetes based on eight clinical variables.
""")

if st.button("🚀 Run Diagnostic Analysis"):
    # 1. Prediction Logic
    input_scaled = scaler.transform(input_df)
    prediction_prob_raw = model.predict(input_scaled)
    # Extract the probability for Class 1 (Diabetes)
    risk_score = float(prediction_prob_raw[0, 1]) 

    # 2. Results Header
    st.divider()
    col1, col2 = st.columns([1, 1.5])
    
    with col1:
        if risk_score >= 0.5:
            st.error("### Result: High Risk")
            st.write("The model suggests a **Positive** diagnosis.")
        else:
            st.success("### Result: Low Risk")
            st.write("The model suggests a **Negative** diagnosis.")
        
        st.metric("Risk Probability Score", f"{risk_score:.2%}")

    with col2:
        st.markdown("### 💡 Understanding the Score")
        # Explanation of "Is it good or bad?"
        if risk_score < 0.25:
            st.info(f"**Is this good? YES.** \n\nA score of {risk_score:.2%} is low. This means the model is highly confident the patient is healthy. In medical screening, a low 'positive probability' is the desired healthy outcome.")
        elif 0.25 <= risk_score < 0.50:
            st.warning(f"**Is this good? MODERATE.** \n\nAt {risk_score:.2%}, the patient is currently below the 50% threshold, but features are entering a 'borderline' zone. Monitoring Glucose and BMI is recommended.")
        else:
            st.error(f"**Is this bad? YES.** \n\nA score of {risk_score:.2%} indicates high model certainty that the patient has diabetes. This suggests a strong need for clinical follow-up.")

    # 3. ANALYSIS TABS
    st.divider()
    tab1, tab2, tab3 = st.tabs(["📈 Training & Performance", "🧪 Why this result? (SHAP)", "📊 Data Overview"])

    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            st.write("**Learning Curve (Loss)**")
            fig_loss, ax_loss = plt.subplots()
            ax_loss.plot(history['loss'], label='Training Loss', color='#1f77b4')
            ax_loss.plot(history['val_loss'], label='Validation Loss', color='#ff7f0e')
            ax_loss.set_xlabel("Epochs")
            ax_loss.set_ylabel("Loss")
            ax_loss.legend()
            st.pyplot(fig_loss)
        
        with c2:
            st.write("**Predictive Power (ROC Curve)**")
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            roc_auc = auc(fpr, tpr)
            fig_roc, ax_roc = plt.subplots()
            ax_roc.plot(fpr, tpr, color='darkorange', label=f'AUC = {roc_auc:.3f}')
            ax_roc.plot([0, 1], [0, 1], color='navy', linestyle='--')
            ax_roc.set_xlabel("False Positive Rate")
            ax_roc.set_ylabel("True Positive Rate")
            ax_roc.legend(loc="lower right")
            st.pyplot(fig_roc)

    with tab2:
        st.write("**Feature Impact Analysis (SHAP)**")
        # Explaining the prediction for the current input
        explainer = shap.Explainer(model, X_test_scaled)
        shap_values = explainer(input_scaled)
        
        # Defining names for the SHAP plot
        feature_names = [
            'Pregnancies', 'Glucose', 'Blood Pressure', 'Skin Thickness', 
            'Insulin', 'BMI', 'Diabetes Pedigree Function', 'Age'
        ]
        shap_values.feature_names = feature_names

        fig_shap, ax_shap = plt.subplots(figsize=(10, 5))
        # Visualizing index 1 for Diabetes Class
        shap.plots.bar(shap_values[0, :, 1], show=False)
        st.pyplot(fig_shap)
        st.caption("Right (Positive): Feature increased risk | Left (Negative): Feature decreased risk.")

    with tab3:
        st.write("**Global Dataset Trends (PIMA Indians)**")
        
        try:
            # Load the local CSV instead of a URL
            df_raw = pd.read_csv("diabetes_raw.csv")
            
            # Create a 2x4 grid for the 8 features
            fig_dist, axes = plt.subplots(2, 4, figsize=(16, 8))
            axes = axes.flatten()
            
            # Plot only the feature columns (excluding 'Class' or 'Outcome')
            feature_cols = df_raw.columns[:8] 
            
            for i, col in enumerate(feature_cols):
                sns.kdeplot(df_raw[col], ax=axes[i], fill=True, color="#52b788")
                axes[i].set_title(col, fontsize=12, fontweight='bold')
                axes[i].set_xlabel("")
                axes[i].set_ylabel("")
            
            plt.tight_layout()
            st.pyplot(fig_dist)
            
        except FileNotFoundError:
            st.warning("⚠️ 'diabetes_raw.csv' not found. Please export it from your notebook to see these plots.")
        except Exception as e:
            st.error(f"Error loading distribution: {e}")
else:
    st.info("👈 Set the patient's metrics on the left and click **Run Diagnostic Analysis** to see results.")
