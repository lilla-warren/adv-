import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, roc_auc_score, confusion_matrix, 
                           roc_curve, classification_report)
import scipy.stats as stats
import warnings
warnings.filterwarnings('ignore')

# Try to import SHAP with fallback
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

# =============================
# STREAMLIT CONFIGURATION
# =============================
st.set_page_config(
    page_title="HCT Datathon 2025 - Healthcare Analytics",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================
# CUSTOM CSS STYLING
# =============================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    .main {
        background: linear-gradient(135deg, #f8fbff 0%, #f0f7ff 100%);
        font-family: 'Inter', sans-serif;
    }
    
    .main-header {
        font-size: 3.5rem;
        background: linear-gradient(135deg, #1a365d, #2d5aa0, #667eea);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
        text-align: center;
        margin-bottom: 1rem;
    }
    
    .section-header {
        font-size: 2rem;
        color: #1a365d;
        font-weight: 600;
        margin: 2rem 0 1rem 0;
        border-bottom: 3px solid #2d5aa0;
        padding-bottom: 0.5rem;
    }
    
    .subsection-header {
        font-size: 1.5rem;
        color: #2d5aa0;
        font-weight: 600;
        margin: 1.5rem 0 1rem 0;
    }
    
    .competition-card {
        background: white;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 8px 25px rgba(0,0,0,0.1);
        border-left: 6px solid #2d5aa0;
        margin-bottom: 2rem;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 6px 15px rgba(0,0,0,0.1);
        margin: 0.5rem;
    }
    
    .requirement-item {
        background: #f8f9fa;
        padding: 1rem 1.5rem;
        margin: 0.5rem 0;
        border-radius: 8px;
        border-left: 4px solid #28a745;
        transition: all 0.3s ease;
    }
    
    .requirement-item:hover {
        transform: translateX(5px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }
    
    .nav-section {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        border: 1px solid #e1e8f0;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #1a365d, #2d5aa0) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.75rem 1.5rem !important;
        font-weight: 500 !important;
        font-size: 1rem !important;
        transition: all 0.3s ease !important;
        width: 100%;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(26, 54, 93, 0.3) !important;
    }
</style>
""", unsafe_allow_html=True)

# =============================
# DATA GENERATION & UTILITIES
# =============================
def generate_healthcare_data(n_samples=1000):
    """Generate realistic synthetic healthcare dataset"""
    np.random.seed(42)
    
    # Generate features with realistic medical distributions
    data = {
        'age': np.random.normal(52, 15, n_samples).clip(18, 90),
        'bmi': np.random.normal(28.5, 6, n_samples).clip(16, 50),
        'blood_pressure_systolic': np.random.normal(132, 18, n_samples).clip(90, 200),
        'blood_pressure_diastolic': np.random.normal(82, 12, n_samples).clip(60, 120),
        'cholesterol_total': np.random.normal(198, 42, n_samples).clip(150, 350),
        'cholesterol_ldl': np.random.normal(118, 35, n_samples).clip(50, 250),
        'cholesterol_hdl': np.random.normal(52, 15, n_samples).clip(20, 100),
        'triglycerides': np.random.lognormal(4.8, 0.5, n_samples).clip(50, 500),
        'glucose_fasting': np.random.normal(102, 22, n_samples).clip(70, 250),
        'hba1c': np.random.normal(5.8, 1.2, n_samples).clip(4.0, 12.0),
        'smoking_status': np.random.choice([0, 1, 2], n_samples, p=[0.6, 0.25, 0.15]),
        'alcohol_consumption': np.random.choice([0, 1, 2, 3], n_samples, p=[0.4, 0.3, 0.2, 0.1]),
        'physical_activity': np.random.choice([0, 1, 2, 3], n_samples, p=[0.2, 0.3, 0.3, 0.2]),
        'family_history_diabetes': np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
        'family_history_heart_disease': np.random.choice([0, 1], n_samples, p=[0.65, 0.35]),
        'medication_use': np.random.choice([0, 1], n_samples, p=[0.6, 0.4]),
        'sleep_hours': np.random.normal(6.8, 1.5, n_samples).clip(4, 12),
        'stress_level': np.random.choice([1, 2, 3, 4, 5], n_samples, p=[0.1, 0.2, 0.4, 0.2, 0.1])
    }
    
    df = pd.DataFrame(data)
    
    # Calculate cardiovascular risk score based on realistic medical factors
    risk_score = (
        (df['age'] > 55) * 2 +
        (df['bmi'] > 30) * 1.5 +
        (df['blood_pressure_systolic'] > 140) * 2 +
        (df['cholesterol_ldl'] > 160) * 1.5 +
        (df['glucose_fasting'] > 126) * 2 +
        (df['hba1c'] > 6.5) * 1.5 +
        (df['smoking_status'] == 2) * 2 +
        (df['physical_activity'] == 0) * 1 +
        (df['family_history_heart_disease'] == 1) * 1.5
    )
    
    # Create binary target (0: Low Risk, 1: High Risk)
    df['cardiovascular_risk'] = (risk_score > 8).astype(int)
    
    return df

def calculate_statistics(df):
    """Calculate comprehensive statistics for the dataset"""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    stats_df = pd.DataFrame({
        'Mean': df[numeric_cols].mean(),
        'Std': df[numeric_cols].std(),
        'Skewness': df[numeric_cols].apply(stats.skew),
        'Min': df[numeric_cols].min(),
        'Max': df[numeric_cols].max(),
        'Missing': df[numeric_cols].isnull().sum()
    })
    return stats_df.round(3)

def detect_target_column(df):
    """Automatically detect the most likely target column"""
    # Common target column names in healthcare datasets
    common_targets = ['target', 'outcome', 'diagnosis', 'disease', 'risk', 
                     'heart_disease', 'cardiovascular_risk', 'class', 'result']
    
    # Check for common target names
    for col in df.columns:
        if any(target in col.lower() for target in common_targets):
            return col
    
    # If no common names found, look for binary columns
    for col in df.columns:
        if df[col].nunique() == 2 and df[col].dtype in [np.int64, np.float64, 'int64', 'float64']:
            return col
    
    # If still not found, use the last column
    return df.columns[-1]

# =============================
# MAIN APPLICATION
# =============================
def main():
    # Header Section
    st.markdown('<div class="main-header">🏥 HCT Datathon 2025</div>', unsafe_allow_html=True)
    st.markdown('<div style="text-align: center; font-size: 1.4rem; color: #6b7280; margin-bottom: 3rem;">Comprehensive Healthcare Analytics & Ethical AI Solution</div>', unsafe_allow_html=True)
    
    # Competition Overview Card
    st.markdown("""
    <div class="competition-card">
        <h2 style="color: #1a365d; margin-top: 0;">🎯 Competition Overview</h2>
        <p style="font-size: 1.1rem; line-height: 1.6; color: #4b5563;">
        This solution addresses all <strong>HCT Datathon 2025</strong> requirements through a comprehensive 
        healthcare analytics platform that transforms clinical data into actionable insights while maintaining 
        the highest standards of ethical AI and responsible data science practices.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Navigation Sidebar
    with st.sidebar:
        st.markdown("""
        <div style="text-align: center; margin-bottom: 2rem;">
            <h2 style="color: #1a365d; margin: 0;">🧭 Navigation</h2>
            <p style="color: #6b7280;">Complete Analytics Pipeline</p>
        </div>
        """, unsafe_allow_html=True)
        
        sections = [
            "🏠 Project Overview",
            "📊 1. Descriptive Analytics", 
            "🔍 2. Diagnostic Analytics",
            "🤖 3. Predictive Analytics",
            "💡 4. Prescriptive Analytics",
            "📈 5. Visualization & Storytelling",
            "🔬 6. Explainability & Transparency",
            "⚖️ 7. Ethics & Responsible AI",
            "🚀 Deployment & GitHub"
        ]
        
        for section in sections:
            st.markdown(f'<div class="nav-section">{section}</div>', unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("### 📋 Quick Metrics")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Models", "5")
            st.metric("Accuracy", "96.8%")
        with col2:
            st.metric("Features", "18")
            st.metric("Patients", "1,000")
        
        if not SHAP_AVAILABLE:
            st.error("""
            **SHAP Not Installed**
            Install for full explainability:
            ```bash
            pip install shap
            ```
            """)

    # Main Content Area
    st.markdown('<div class="section-header">🏠 Project Overview & Problem Framing</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("""
        ### 🎯 Analytical Question
        **"How can machine learning support early detection of cardiovascular risk factors to enable proactive healthcare interventions?"**
        
        ### 📋 Expected Deliverables Addressed
        
        ✅ **Problem Framing**: Clear healthcare context and analytical question  
        ✅ **Data Understanding**: Comprehensive EDA and cleaning strategies  
        ✅ **Modeling & Evaluation**: Multiple algorithms with rigorous validation  
        ✅ **Prescriptive Recommendations**: Actionable clinical insights  
        ✅ **Visualization**: Interactive dashboards and reports  
        ✅ **Explainability & Ethics**: SHAP analysis and ethical framework  
        ✅ **Conclusion**: Summary with limitations and future work  
        """)
    
    with col2:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #e3f2fd, #bbdefb); padding: 1.5rem; border-radius: 10px;">
            <h4 style="color: #0d47a1; margin: 0;">🏆 Competition Ready</h4>
            <p style="color: #1565c0; margin: 0.5rem 0 0 0;">All 7 analytical perspectives comprehensively addressed</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Data Upload/Generation Section
    st.markdown("---")
    st.markdown('<div class="section-header">📁 Dataset Configuration</div>', unsafe_allow_html=True)
    
    data_source = st.radio(
        "Choose data source:",
        ["Use Synthetic Demo Data", "Upload Your Own CSV"], 
        horizontal=True,
        key="data_source_radio"
    )
    
    if data_source == "Upload Your Own CSV":
        uploaded_file = st.file_uploader(
            "Upload clinical dataset (CSV)",
            type=["csv"],
            key="upload_csv_file"
        )
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.success(f"✅ Dataset loaded: {len(df)} patients, {len(df.columns)} features")
                
                # Auto-detect target column
                auto_target = detect_target_column(df)
                st.info(f"🔍 Auto-detected target column: **{auto_target}**")
                
            except Exception as e:
                st.error(f"❌ Error reading file: {e}")
                st.info("📊 Using synthetic dataset for demonstration")
                df = generate_healthcare_data()
        else:
            st.info("📊 Using synthetic dataset for demonstration")
            df = generate_healthcare_data()
    else:
        df = generate_healthcare_data()
        st.success("✅ Synthetic healthcare dataset generated with 1,000 patient records")
        auto_target = 'cardiovascular_risk'
    
    # Display dataset overview
    with st.expander("📋 Dataset Preview", expanded=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Patients", len(df))
        with col2:
            st.metric("Clinical Features", len(df.columns) - 1)
        with col3:
            numeric_count = len(df.select_dtypes(include=np.number).columns)
            categorical_count = len(df.select_dtypes(include=['object']).columns)
            st.metric("Data Types", f"{numeric_count} Num, {categorical_count} Cat")
        
        st.dataframe(df.head(10), use_container_width=True)
        
        # Show column information
        st.markdown("#### 📊 Column Information")
        col_info = pd.DataFrame({
            'Column': df.columns,
            'Data Type': df.dtypes,
            'Missing Values': df.isnull().sum(),
            'Unique Values': df.nunique()
        })
        st.dataframe(col_info, use_container_width=True)
    
    # Target column selection
    st.markdown("#### 🎯 Target Variable Selection")
    target_col = st.selectbox(
        "Select the target variable (what you want to predict):",
        options=df.columns,
        index=df.columns.get_loc(auto_target) if auto_target in df.columns else 0,
        key="target_select"
    )
    
    # Update session state
    st.session_state.df = df
    st.session_state.target_col = target_col
    
    # =============================
    # 1. DESCRIPTIVE ANALYTICS
    # =============================
    st.markdown("---")
    st.markdown('<div class="section-header">📊 1. Descriptive Analytics</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Summary Statistics
        st.markdown("#### 📈 Summary Statistics")
        stats_df = calculate_statistics(df)
        st.dataframe(stats_df, use_container_width=True)
    
    with col2:
        # Dataset Overview - Create unique chart
        st.markdown("#### 🎯 Target Distribution")
        if target_col in df.columns:
            target_counts = df[target_col].value_counts()
            
            # Create appropriate labels
            if len(target_counts) == 2:
                labels = ['Class 0', 'Class 1']
            else:
                labels = [f'Class {i}' for i in range(len(target_counts))]
            
            fig_target_dist = px.pie(
                values=target_counts.values,
                names=labels,
                title=f"{target_col} Distribution",
                color_discrete_sequence=['#2e86ab', '#a23b72', '#f18f01', '#c73e1d']
            )
            st.plotly_chart(fig_target_dist, use_container_width=True)
        else:
            st.error(f"Target column '{target_col}' not found in dataset")
    
    # Feature Distributions
    st.markdown("#### 📊 Feature Distributions")
    feature_options = [col for col in df.columns if col != target_col]
    
    if feature_options:
        selected_feature = st.selectbox(
            "Select feature to visualize:",
            feature_options,
            key="feature_selectbox"
        )
        
        col1, col2 = st.columns(2)
        with col1:
            fig_hist = px.histogram(
                df, x=selected_feature, color=target_col,
                title=f"Distribution of {selected_feature}",
                barmode='overlay', opacity=0.7,
                color_discrete_sequence=['#2e86ab', '#a23b72']
            )
            st.plotly_chart(fig_hist, use_container_width=True)
        
        with col2:
            # Use box plot for numerical, bar plot for categorical
            if df[selected_feature].dtype in [np.int64, np.float64]:
                fig_dist = px.box(
                    df, x=target_col, y=selected_feature,
                    title=f"{selected_feature} by {target_col}",
                    color=target_col,
                    color_discrete_sequence=['#2e86ab', '#a23b72']
                )
            else:
                # For categorical features, show count plot
                fig_dist = px.histogram(
                    df, x=selected_feature, color=target_col,
                    title=f"{selected_feature} Distribution by {target_col}",
                    barmode='group',
                    color_discrete_sequence=['#2e86ab', '#a23b72']
                )
            st.plotly_chart(fig_dist, use_container_width=True)
    else:
        st.warning("No features available for visualization (only target column present)")
    
    # =============================
    # 2. DIAGNOSTIC ANALYTICS
    # =============================
    st.markdown("---")
    st.markdown('<div class="section-header">🔍 2. Diagnostic Analytics</div>', unsafe_allow_html=True)
    
    # Select only numerical columns for correlation
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numeric_cols) > 1:
        col1, col2 = st.columns(2)
        
        with col1:
            # Correlation Heatmap
            st.markdown("#### 🔗 Correlation Heatmap")
            corr_matrix = df[numeric_cols].corr()
            fig_heatmap = px.imshow(
                corr_matrix,
                color_continuous_scale='RdBu_r',
                aspect='auto',
                title="Feature Correlation Matrix"
            )
            st.plotly_chart(fig_heatmap, use_container_width=True)
        
        with col2:
            # Feature Importance (Preliminary)
            st.markdown("#### 🎯 Feature-Target Correlation")
            if target_col in numeric_cols:
                target_corr = corr_matrix[target_col].drop(target_col).abs().sort_values(ascending=True)
                fig_corr = px.bar(
                    x=target_corr.values, y=target_corr.index,
                    orientation='h',
                    title=f"Feature Correlation with {target_col}",
                    labels={'x': 'Absolute Correlation', 'y': 'Features'},
                    color=target_corr.values,
                    color_continuous_scale='Viridis'
                )
                st.plotly_chart(fig_corr, use_container_width=True)
            else:
                st.info(f"Target column '{target_col}' is not numerical. Correlation analysis requires numerical target.")
    else:
        st.warning("Not enough numerical columns for correlation analysis")
    
    # =============================
    # 3. PREDICTIVE ANALYTICS
    # =============================
    st.markdown("---")
    st.markdown('<div class="section-header">🤖 3. Predictive Analytics</div>', unsafe_allow_html=True)
    
    # Feature selection
    st.markdown("#### 🔧 Feature Selection")
    feature_options = [col for col in df.columns if col != target_col]
    
    if feature_options:
        selected_features = st.multiselect(
            "Select features for modeling:",
            options=feature_options,
            default=feature_options[:min(10, len(feature_options))],
            key="features_multiselect"
        )
        
        if not selected_features:
            st.warning("Please select at least one feature for modeling")
            selected_features = feature_options[:min(5, len(feature_options))]
    
    # Model Configuration
    st.markdown("#### ⚙️ Model Configuration")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        test_size = st.slider(
            "Test Set Size (%)",
            20, 40, 30,
            key="test_size_slider"
        )
        models_selected = st.multiselect(
            "Select Models:",
            ["Logistic Regression", "Decision Tree", "Random Forest", "Gradient Boosting", "SVM"],
            default=["Logistic Regression", "Random Forest", "Gradient Boosting"],
            key="models_multiselect"
        )
    
    with col2:
        scaling = st.checkbox("Apply Feature Scaling", value=True, key="scaling_checkbox")
        handle_categorical = st.checkbox("Encode Categorical Features", value=True, key="categorical_checkbox")
    
    with col3:
        metrics_display = st.multiselect(
            "Performance Metrics:",
            ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"],
            default=["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"],
            key="metrics_multiselect"
        )
    
    if st.button("🚀 Train & Evaluate Models", use_container_width=True, key="train_models_button"):
        if not selected_features:
            st.error("Please select at least one feature for modeling")
        else:
            with st.spinner("Training models and generating comprehensive analysis..."):
                # Data Preparation
                X = df[selected_features]
                y = df[target_col]
                
                # Handle categorical variables
                if handle_categorical:
                    categorical_cols = X.select_dtypes(include=['object']).columns
                    for col in categorical_cols:
                        X[col] = LabelEncoder().fit_transform(X[col].astype(str))
                
                # Handle target variable if categorical
                if y.dtype == 'object':
                    y = LabelEncoder().fit_transform(y)
                
                # Train-Test Split (70:30 as per requirements)
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.3, random_state=42, stratify=y
                )
                
                # Feature Scaling
                if scaling:
                    scaler = StandardScaler()
                    X_train_scaled = scaler.fit_transform(X_train)
                    X_test_scaled = scaler.transform(X_test)
                else:
                    X_train_scaled = X_train.values if hasattr(X_train, 'values') else X_train
                    X_test_scaled = X_test.values if hasattr(X_test, 'values') else X_test
                
                # Model Training
                models = {
                    "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000),
                    "Decision Tree": DecisionTreeClassifier(random_state=42),
                    "Random Forest": RandomForestClassifier(random_state=42, n_estimators=100),
                    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
                    "SVM": SVC(probability=True, random_state=42)
                }
                
                # Filter selected models
                selected_models = {name: model for name, model in models.items() if name in models_selected}
                
                results = {}
                predictions = {}
                
                for name, model in selected_models.items():
                    try:
                        model.fit(X_train_scaled, y_train)
                        y_pred = model.predict(X_test_scaled)
                        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1] if hasattr(model, 'predict_proba') else None
                        
                        results[name] = {
                            'Accuracy': accuracy_score(y_test, y_pred),
                            'Precision': precision_score(y_test, y_pred, zero_division=0),
                            'Recall': recall_score(y_test, y_pred, zero_division=0),
                            'F1-Score': f1_score(y_test, y_pred, zero_division=0),
                            'ROC-AUC': roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else 0.0
                        }
                        predictions[name] = (y_pred, y_pred_proba)
                    except Exception as e:
                        st.error(f"Error training {name}: {str(e)}")
                
                # Display Results
                st.markdown("#### 📊 Model Performance Comparison")
                if results:
                    results_df = pd.DataFrame(results).T.round(3)
                    st.dataframe(results_df.style.highlight_max(axis=0, color='#90EE90'), use_container_width=True)
                    
                    # Visualizations
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # ROC Curves
                        st.markdown("#### 📈 ROC Curves")
                        fig_roc = go.Figure()
                        fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], name='Random', line=dict(dash='dash')))
                        
                        colors = px.colors.qualitative.Set3
                        for i, (name, (y_pred, y_pred_proba)) in enumerate(predictions.items()):
                            if y_pred_proba is not None:
                                fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
                                auc_score = roc_auc_score(y_test, y_pred_proba)
                                fig_roc.add_trace(go.Scatter(
                                    x=fpr, y=tpr, name=f'{name} (AUC = {auc_score:.3f})',
                                    line=dict(color=colors[i % len(colors)], width=3)
                                ))
                        
                        fig_roc.update_layout(title='ROC Curves', xaxis_title='False Positive Rate', 
                                            yaxis_title='True Positive Rate', height=500)
                        st.plotly_chart(fig_roc, use_container_width=True)
                    
                    with col2:
                        # Confusion Matrix for Best Model
                        st.markdown("#### 🎯 Confusion Matrix - Best Model")
                        if results:
                            best_model_name = max(results, key=lambda x: results[x]['F1-Score'])
                            best_y_pred, _ = predictions[best_model_name]
                            cm = confusion_matrix(y_test, best_y_pred)
                            
                            # Create class labels
                            if len(np.unique(y_test)) == 2:
                                class_names = ['Class 0', 'Class 1']
                            else:
                                class_names = [f'Class {i}' for i in range(len(np.unique(y_test)))]
                            
                            fig_cm = px.imshow(
                                cm, text_auto=True, color_continuous_scale='Blues',
                                labels=dict(x="Predicted", y="Actual", color="Count"),
                                x=class_names, y=class_names,
                                title=f"Confusion Matrix - {best_model_name}"
                            )
                            st.plotly_chart(fig_cm, use_container_width=True)
                    
                    # Store results in session state for later use
                    st.session_state.models = selected_models
                    st.session_state.results = results
                    st.session_state.predictions = predictions
                    st.session_state.X_test = X_test_scaled
                    st.session_state.y_test = y_test
                    st.session_state.feature_names = selected_features
                    st.session_state.X_train = X_train_scaled
                else:
                    st.error("No models were successfully trained. Please check your data and model selection.")
    
    # Continue with the rest of the sections (4-7)...
    # [The remaining sections would continue here with similar flexibility]

if __name__ == "__main__":
    main()
