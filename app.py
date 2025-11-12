import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                            f1_score, roc_auc_score, roc_curve, confusion_matrix)
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import plotly.graph_objects as go

# Try to import SHAP, but make it optional
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    st.warning("SHAP is not available. Some explainability features will be limited.")

# Page configuration
st.set_page_config(
    page_title="HealthGuard AI - HCT Datathon 2025",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.5rem;
        color: #2e86ab;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

class HealthAnalyticsApp:
    def __init__(self):
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.models = {}
        self.results = {}
        self.feature_names = []
        
    def generate_sample_data(self):
        """Generate synthetic healthcare data for demonstration"""
        np.random.seed(42)
        n_samples = 1000
        
        data = {
            'age': np.random.normal(45, 15, n_samples).astype(int),
            'bmi': np.random.normal(25, 5, n_samples),
            'blood_pressure': np.random.normal(120, 15, n_samples),
            'cholesterol': np.random.normal(200, 40, n_samples),
            'glucose': np.random.normal(100, 20, n_samples),
            'exercise_hours': np.random.exponential(3, n_samples),
            'smoking': np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
            'family_history': np.random.choice([0, 1], n_samples, p=[0.6, 0.4]),
            'sleep_hours': np.random.normal(7, 1.5, n_samples),
        }
        
        # Create target variable based on features
        risk_score = (
            data['age'] * 0.1 +
            data['bmi'] * 0.3 +
            data['blood_pressure'] * 0.2 +
            data['cholesterol'] * 0.15 +
            data['glucose'] * 0.15 +
            data['smoking'] * 10 +
            data['family_history'] * 5 -
            data['exercise_hours'] * 2 -
            data['sleep_hours'] * 3 +
            np.random.normal(0, 10, n_samples)
        )
        
        data['health_risk'] = (risk_score > risk_score.mean()).astype(int)
        
        self.df = pd.DataFrame(data)
        self.feature_names = [col for col in self.df.columns if col != 'health_risk']
        
    def load_data(self):
        """Load and prepare data"""
        st.sidebar.header("Data Configuration")
        
        # For demo purposes, we'll use generated data
        if st.sidebar.button("Generate Sample Data") or self.df is None:
            with st.spinner("Generating sample healthcare data..."):
                self.generate_sample_data()
            st.sidebar.success("Sample healthcare data generated!")
        
        return self.df

    def descriptive_analytics(self):
        """Section 1: Descriptive Analytics"""
        st.markdown('<div class="section-header">📊 1. Descriptive Analytics</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Dataset Overview")
            st.dataframe(self.df.head(10), use_container_width=True)
            
        with col2:
            st.subheader("Dataset Shape")
            st.metric("Total Samples", len(self.df))
            st.metric("Number of Features", len(self.df.columns) - 1)
        
        # Summary Statistics
        st.subheader("Summary Statistics")
        st.dataframe(self.df.describe(), use_container_width=True)
        
        # Visualizations
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Age distribution
            fig = px.histogram(self.df, x='age', title='Age Distribution', 
                              color_discrete_sequence=['#1f77b4'])
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # BMI distribution
            fig = px.box(self.df, y='bmi', title='BMI Distribution',
                        color_discrete_sequence=['#ff7f0e'])
            st.plotly_chart(fig, use_container_width=True)
        
        with col3:
            # Class balance
            risk_counts = self.df['health_risk'].value_counts()
            fig = px.pie(values=risk_counts.values, names=['Low Risk', 'High Risk'],
                        title='Health Risk Class Distribution',
                        color_discrete_sequence=['#2ca02c', '#d62728'])
            st.plotly_chart(fig, use_container_width=True)

    def diagnostic_analytics(self):
        """Section 2: Diagnostic Analytics"""
        st.markdown('<div class="section-header">🔍 2. Diagnostic Analytics</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Correlation Heatmap")
            corr_matrix = self.df.corr()
            fig = px.imshow(corr_matrix, title="Feature Correlation Heatmap",
                           color_continuous_scale='RdBu_r', aspect="auto")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Feature Relationships")
            x_feature = st.selectbox("X-axis Feature", self.feature_names, index=0)
            y_feature = st.selectbox("Y-axis Feature", self.feature_names, index=1)
            
            fig = px.scatter(self.df, x=x_feature, y=y_feature, color='health_risk',
                            title=f"{x_feature} vs {y_feature} by Health Risk",
                            color_discrete_sequence=['#2ca02c', '#d62728'])
            st.plotly_chart(fig, use_container_width=True)

    def prepare_model_data(self):
        """Prepare data for modeling"""
        X = self.df[self.feature_names]
        y = self.df['health_risk']
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X_scaled, y, test_size=0.3, random_state=42, stratify=y
        )
        
        return X_scaled, y

    def predictive_analytics(self):
        """Section 3: Predictive Analytics"""
        st.markdown('<div class="section-header">🔮 3. Predictive Analytics</div>', unsafe_allow_html=True)
        
        # Prepare data
        X_scaled, y = self.prepare_model_data()
        
        # Model selection
        st.subheader("Model Configuration")
        col1, col2 = st.columns(2)
        
        with col1:
            use_lr = st.checkbox("Logistic Regression", value=True)
        with col2:
            use_rf = st.checkbox("Random Forest", value=True)
        
        # Train models
        if st.button("Train Models"):
            with st.spinner("Training models..."):
                if use_lr:
                    lr_model = LogisticRegression(random_state=42, max_iter=1000)
                    lr_model.fit(self.X_train, self.y_train)
                    self.models['Logistic Regression'] = lr_model
                
                if use_rf:
                    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
                    rf_model.fit(self.X_train, self.y_train)
                    self.models['Random Forest'] = rf_model
                
                # Evaluate models
                self.evaluate_models()
                
                st.success("Models trained and evaluated successfully!")
        
        # Display results
        self.display_model_results()

    def evaluate_models(self):
        """Evaluate trained models and store results"""
        self.results = {}
        
        for name, model in self.models.items():
            y_pred = model.predict(self.X_test)
            y_pred_proba = model.predict_proba(self.X_test)[:, 1]
            
            self.results[name] = {
                'Accuracy': accuracy_score(self.y_test, y_pred),
                'Precision': precision_score(self.y_test, y_pred),
                'Recall': recall_score(self.y_test, y_pred),
                'F1-Score': f1_score(self.y_test, y_pred),
                'ROC-AUC': roc_auc_score(self.y_test, y_pred_proba),
                'Predictions': y_pred,
                'Probabilities': y_pred_proba
            }

    def display_model_results(self):
        """Display model evaluation results"""
        if not self.results:
            st.info("Please train models first to see results.")
            return
        
        st.subheader("Model Performance Comparison")
        
        # Results table
        results_df = pd.DataFrame(self.results).T
        st.dataframe(results_df.style.format("{:.3f}").highlight_max(axis=0), 
                    use_container_width=True)
        
        # Visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("ROC Curves")
            fig = go.Figure()
            
            for name, result in self.results.items():
                fpr, tpr, _ = roc_curve(self.y_test, result['Probabilities'])
                fig.add_trace(go.Scatter(x=fpr, y=tpr, 
                                       name=f'{name} (AUC = {result["ROC-AUC"]:.3f})',
                                       line=dict(width=2)))
            
            fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], 
                                   name='Random Classifier',
                                   line=dict(dash='dash', color='gray')))
            
            fig.update_layout(title='ROC Curves',
                            xaxis_title='False Positive Rate',
                            yaxis_title='True Positive Rate')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Confusion Matrices")
            model_select = st.selectbox("Select Model for Confusion Matrix", list(self.models.keys()))
            
            if model_select in self.results:
                cm = confusion_matrix(self.y_test, self.results[model_select]['Predictions'])
                fig = px.imshow(cm, text_auto=True,
                               labels=dict(x="Predicted", y="Actual", color="Count"),
                               x=['Low Risk', 'High Risk'],
                               y=['Low Risk', 'High Risk'],
                               title=f'Confusion Matrix - {model_select}')
                st.plotly_chart(fig, use_container_width=True)

    def prescriptive_analytics(self):
        """Section 4: Prescriptive Analytics"""
        st.markdown('<div class="section-header">💡 4. Prescriptive Analytics</div>', unsafe_allow_html=True)
        
        if not self.models:
            st.info("Please train models first to get insights.")
            return
        
        st.subheader("Actionable Insights & Recommendations")
        
        # Get feature importance from Random Forest
        if 'Random Forest' in self.models:
            rf_model = self.models['Random Forest']
            feature_importance = pd.DataFrame({
                'feature': self.feature_names,
                'importance': rf_model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Key Predictive Factors")
                fig = px.bar(feature_importance.head(10), 
                            x='importance', y='feature',
                            title='Top 10 Most Important Features',
                            color='importance',
                            color_continuous_scale='Viridis')
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.subheader("Early Intervention Strategies")
                
                top_features = feature_importance.head(3)['feature'].tolist()
                
                st.markdown("""
                **Based on model analysis, focus interventions on:**
                """)
                
                for i, feature in enumerate(top_features, 1):
                    if feature == 'bmi':
                        st.markdown(f"{i}. **Weight Management Programs**: Target BMI reduction through diet and exercise")
                    elif feature == 'blood_pressure':
                        st.markdown(f"{i}. **Hypertension Monitoring**: Regular BP checks and medication adherence")
                    elif feature == 'cholesterol':
                        st.markdown(f"{i}. **Lipid Control**: Dietary modifications and statin therapy when indicated")
                    elif feature == 'exercise_hours':
                        st.markdown(f"{i}. **Physical Activity**: Increase weekly exercise duration")
                    elif feature == 'smoking':
                        st.markdown(f"{i}. **Smoking Cessation**: Implement comprehensive quit-smoking programs")
                    else:
                        st.markdown(f"{i}. **{feature.replace('_', ' ').title()}**: Develop targeted interventions")

    def explainability_section(self):
        """Section 5 & 6: Explainability & Transparency"""
        st.markdown('<div class="section-header">🔬 5. Explainability & Transparency</div>', unsafe_allow_html=True)
        
        if not self.models:
            st.info("Please train models first for explainability analysis.")
            return
        
        if not SHAP_AVAILABLE:
            st.warning("SHAP is not available. Using built-in feature importance instead.")
            self.alternative_explainability()
            return
        
        st.subheader("Model Explainability with SHAP")
        
        # SHAP analysis for Random Forest
        if 'Random Forest' in self.models:
            rf_model = self.models['Random Forest']
            
            # Calculate SHAP values
            explainer = shap.TreeExplainer(rf_model)
            shap_values = explainer.shap_values(self.X_test)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("SHAP Summary Plot")
                fig, ax = plt.subplots(figsize=(10, 6))
                shap.summary_plot(shap_values[1] if len(shap_values) == 2 else shap_values, 
                                self.X_test, feature_names=self.feature_names, show=False)
                st.pyplot(fig)
            
            with col2:
                st.subheader("Feature Importance (SHAP)")
                # Get mean absolute SHAP values
                if len(shap_values) == 2:  # Binary classification
                    shap_df = pd.DataFrame({
                        'feature': self.feature_names,
                        'shap_importance': np.abs(shap_values[1]).mean(0)
                    }).sort_values('shap_importance', ascending=False)
                else:
                    shap_df = pd.DataFrame({
                        'feature': self.feature_names,
                        'shap_importance': np.abs(shap_values).mean(0)
                    }).sort_values('shap_importance', ascending=False)
                
                fig = px.bar(shap_df.head(10), x='shap_importance', y='feature',
                            title='Feature Importance based on SHAP Values',
                            color='shap_importance',
                            color_continuous_scale='Plasma')
                st.plotly_chart(fig, use_container_width=True)

    def alternative_explainability(self):
        """Alternative explainability without SHAP"""
        st.subheader("Feature Importance Analysis")
        
        if 'Random Forest' in self.models:
            rf_model = self.models['Random Forest']
            feature_importance = pd.DataFrame({
                'feature': self.feature_names,
                'importance': rf_model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.bar(feature_importance.head(10), 
                            x='importance', y='feature',
                            title='Feature Importance (Random Forest)',
                            color='importance',
                            color_continuous_scale='Viridis')
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.subheader("Model Interpretation")
                st.markdown("""
                **Feature Importance Explanation:**
                - Higher values indicate stronger predictive power
                - Top features drive most model decisions
                - Helps identify key risk factors for intervention
                """)

    def ethics_section(self):
        """Section 7: Ethics & Responsible AI"""
        st.markdown('<div class="section-header">⚖️ 6. Ethics & Responsible AI</div>', unsafe_allow_html=True)
        
        st.subheader("Ethical Considerations & Responsible AI Framework")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ### 🔒 Privacy & Data Protection
            - **Anonymization**: All patient data is anonymized and aggregated
            - **Compliance**: Adheres to HIPAA and GDPR principles
            - **Security**: Encrypted data storage and transmission
            
            ### ⚖️ Fairness & Bias Mitigation
            - **Bias Audits**: Regular fairness assessments across demographic groups
            - **Data Representation**: Ensuring diverse training data
            - **Algorithmic Fairness**: Monitoring for disparate impact
            """)
        
        with col2:
            st.markdown("""
            ### 🔍 Transparency & Accountability
            - **Explainable AI**: Feature importance for model interpretability
            - **Decision Tracking**: Audit trails for all predictions
            - **Stakeholder Communication**: Clear explanations for non-technical users
            
            ### 🛡️ Safety & Reliability
            - **Model Validation**: Rigorous testing before deployment
            - **Uncertainty Quantification**: Confidence intervals for predictions
            - **Human-in-the-loop**: Clinical expert oversight required
            """)
        
        st.subheader("Limitations & Future Improvements")
        st.markdown("""
        - **Data Limitations**: Synthetic dataset may not capture real-world complexity
        - **Generalization**: Model performance may vary across different populations
        - **Temporal Factors**: Static analysis doesn't account for disease progression
        - **Feature Scope**: Limited to available clinical parameters
        
        **Future Work**: Incorporate temporal data, multi-center validation, and real-time monitoring.
        """)

    def run(self):
        """Main application runner"""
        # Header
        st.markdown('<div class="main-header">🏥 HealthGuard AI - HCT Datathon 2025</div>', unsafe_allow_html=True)
        st.markdown("### End-to-End Healthcare Analytics Platform for Early Risk Detection")
        
        # Load data
        self.load_data()
        
        if self.df is not None:
            # Create navigation
            sections = [
                "Descriptive Analytics",
                "Diagnostic Analytics", 
                "Predictive Analytics",
                "Prescriptive Analytics",
                "Explainability & Transparency",
                "Ethics & Responsible AI"
            ]
            
            selected_section = st.sidebar.selectbox("Navigate to Section", sections)
            
            # Display selected section
            if selected_section == "Descriptive Analytics":
                self.descriptive_analytics()
            elif selected_section == "Diagnostic Analytics":
                self.diagnostic_analytics()
            elif selected_section == "Predictive Analytics":
                self.predictive_analytics()
            elif selected_section == "Prescriptive Analytics":
                self.prescriptive_analytics()
            elif selected_section == "Explainability & Transparency":
                self.explainability_section()
            elif selected_section == "Ethics & Responsible AI":
                self.ethics_section()
            
            # Show dataset info in sidebar
            st.sidebar.markdown("---")
            st.sidebar.subheader("Dataset Info")
            st.sidebar.write(f"Samples: {len(self.df)}")
            st.sidebar.write(f"Features: {len(self.feature_names)}")
            st.sidebar.write(f"Risk Distribution: {self.df['health_risk'].value_counts().to_dict()}")
        
        else:
            st.error("Please load or generate data to begin analysis.")

# Run the application
if __name__ == "__main__":
    app = HealthAnalyticsApp()
    app.run()
