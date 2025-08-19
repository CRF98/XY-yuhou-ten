import streamlit as st
import pandas as pd
import numpy as np
import shap
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from matplotlib import rcParams

# Configure matplotlib for better visualization
rcParams['font.family'] = 'sans-serif'
rcParams['font.size'] = 10

# Set page config with custom icon
st.set_page_config(
    page_title="Model prediction visualization",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling
st.markdown("""
    <style>
    .main {background-color: #f8f9fa;}
    .stButton>button {background-color: #007bff; color: white; border-radius: 8px;}
    .stNumberInput>label {font-weight: bold; color: #2c3e50;}
    .sidebar .sidebar-content {background-color: #e9ecef;}
    h1 {color: #2c3e50; text-align: center;}
    h2 {color: #34495e; border-bottom: 2px solid #17a2b8; padding-bottom: 5px;}
    </style>
""", unsafe_allow_html=True)

# Title and introduction
st.title("Model prediction visualization")
st.markdown("""
    This tool uses feature data to make predictions and provides mechanistic explanations through SHAP visualization.
    Adjust the feature value in the sidebar to observe the changes in the prediction results and SHAP values.
""")

# Load and prepare background data
@st.cache_data
def load_background_data():
    df = pd.read_excel('data/10feature_train.xlsx')
    return df.iloc[:, :-2]

# calc the cut_off
@st.cache_data
def calc_cut_off():
    df_y = pd.read_excel('data/10feature_test.xlsx').iloc[:, -1]
    df_pred_y = pd.read_excel('data/val_result.xlsx').iloc[:, -1]
    fpr, tpr, thresholds = roc_curve(df_y, df_pred_y)
    cut_off = thresholds[np.argmax(tpr - fpr)]
    return cut_off

# Load the pre-trained model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('data/MODEL.h5')

# Initialize data and model
background_data = load_background_data()
model = load_model()

# Default values for features
default_values = background_data.iloc[0, :].to_dict()

cut_off = calc_cut_off()

# Sidebar configuration
st.sidebar.header("Feature Inputs")
st.sidebar.markdown("Adjust values of features:")

# Reset button
if st.sidebar.button("Reset to Defaults", key="reset"):
    st.session_state.update(default_values)

features = list(default_values.keys())
values = {}
cols = st.sidebar.columns(2)

for i, feature in enumerate(features):
    with cols[i % 2]:
        values[feature] = st.number_input(
            feature,
            min_value=float(background_data[feature].min()),
            max_value=float(background_data[feature].max()),
            value=default_values[feature],
            step=0.001,
            format="%.3f",
            key=feature
        )

# Prepare input data
def prepare_input_data():
    return pd.DataFrame([values])

# Main analysis
@st.cache_data
def determine_model_type():
    try:
        # Read target variable
        df_y = pd.read_excel('data/10feature_test.xlsx').iloc[:, -1]
        
        # Determine variable type
        unique_values = df_y.nunique()
        
        # If number of unique values <= 2 or variable type is object/category, it's a classification model
        if unique_values <= 2 or df_y.dtype in ['object', 'category']:
            model_type = "classification"
            # Get class labels
            labels = df_y.unique()
            return model_type, labels
        else:
            model_type = "regression"
            return model_type, None
            
    except Exception as e:
        st.error(f"Error determining model type: {str(e)}")
        return None, None

# Add model type determination in main program
model_type, class_labels = determine_model_type()
if class_labels is not None:
    class_labels = sorted(class_labels)
print(class_labels)

if st.button("Analyze Calculation", key="calculate"):
    input_df = prepare_input_data()
    
    # Prediction
    prediction = model.predict(input_df.values, verbose=0)[0][0]
    with st.container():
        st.header("📈 Prediction Result")    
        col1, col2 = st.columns(2)
        with col1:
            if model_type == "classification":
                # Classification model display
                predicted_class = class_labels[1] if prediction >= cut_off else class_labels[0]
                st.metric(
                    "Probability", 
                    f"{prediction:.4f}", 
                    delta=f"Predicted Class: {predicted_class}",
                    delta_color="inverse"
                )
            else:
                # Regression model display
                st.metric(
                    "Predicted Value", 
                    f"{prediction:.4f}"
                )
        with col2:
            if model_type == "classification":
                st.metric(
                    "Classification Threshold", 
                    f"{cut_off:.4f}"
                )
            else:
                # Display statistical information for regression model
                st.metric(
                    "Prediction Range", 
                    f"{df_y.min():.2f} - {df_y.max():.2f}"
                )
    
    # SHAP explanation
    explainer = shap.DeepExplainer(model, background_data.values)
    shap_values = np.squeeze(np.array(explainer.shap_values(input_df.values)))
    base_value = float(explainer.expected_value[0].numpy())

    # Visualization tabs
    tab1, tab2, tab3 = st.tabs(["Force Plot", "Decision Plot", "Mechanistic Insights"])
    
    with tab1:
        st.subheader("Force Plot")
        col1, col2 = st.columns([3, 1])  # 创建两个列，图像放在较小的列中
        with col1:
            explanation = shap.Explanation(
                values=shap_values, 
                base_values=base_value, 
                feature_names=input_df.columns,
                data=input_df.values.round(3)
            )
            shap.plots.force(explanation, matplotlib=True, show=False, figsize=(20, 4))
            st.pyplot(plt.gcf(), clear_figure=True)

    with tab2:
        st.subheader("Decision Plot")
        col1, col2 = st.columns([2, 2])  # 创建两个列，图像放在较小的列中
        with col1:
            fig, ax = plt.subplots(figsize=(6, 3))  # 设置图像大小
            shap.decision_plot(base_value, shap_values, input_df.columns, show=False)
            st.pyplot(plt.gcf(), clear_figure=True)
    
    with tab3:
        st.subheader("Mechanistic Insights")
        importance_df = pd.DataFrame({'Feature': input_df.columns, 'SHAP Value': shap_values})
        importance_df = importance_df.sort_values('SHAP Value', ascending=False)
        st.dataframe(importance_df.style.background_gradient(cmap='coolwarm', subset=['SHAP Value']))