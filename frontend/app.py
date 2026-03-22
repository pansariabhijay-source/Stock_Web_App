"""
Main Streamlit application for Stock Prediction Dashboard.
"""
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add utils to path
sys.path.append(str(Path(__file__).parent))

from utils.api_client import APIClient
from utils.charts import (
    plot_shap_waterfall,
    plot_feature_importance
)
from utils.data_loader import load_stock_data, prepare_features

# Page configuration
st.set_page_config(
    page_title="Stock Prediction Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize API client
@st.cache_resource
def get_api_client():
    """Get cached API client instance."""
    api_url = st.secrets.get("API_URL", "http://localhost:8000")
    return APIClient(api_url)

api_client = get_api_client()


def main():
    """Main application."""
    st.title("📈 Stock Prediction Dashboard")
    st.markdown("Production-grade ML system for stock price prediction")
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Select Page",
        ["Prediction", "Explainability", "How to Use"]
    )
    
    # Health check
    try:
        health = api_client.health_check()
        if health["status"] == "healthy":
            st.sidebar.success("✅ API Connected")
        else:
            st.sidebar.warning("⚠️ Models not loaded")
    except Exception as e:
        st.sidebar.error(f"❌ API Error: {e}")
        st.error("Cannot connect to API. Please ensure the backend is running.")
        return
    
    # Route to appropriate page
    if page == "Prediction":
        show_prediction_page()
    elif page == "Explainability":
        show_explainability_page()
    elif page == "How to Use":
        show_how_to_use_page()


def show_prediction_page():
    """Prediction interface page."""
    st.header("Stock Price Prediction")
    
    # Get model info to determine expected feature count
    try:
        model_info = api_client.get_active_model_info()
        expected_feature_count = model_info.get("num_features", 16)
        feature_names = model_info.get("feature_names", [])
    except Exception as e:
        st.warning(f"Could not fetch model info: {e}. Using default feature count.")
        expected_feature_count = 16
        feature_names = []
    
    # Sidebar: Manual Entry
    with st.sidebar:
        st.subheader("Manual Feature Entry")
        st.info("Enter feature values manually")
        
        if feature_names:
            with st.expander("Expected Features", expanded=False):
                st.write(f"Model expects **{expected_feature_count}** features:")
                for i, name in enumerate(feature_names, 1):
                    st.write(f"{i}. {name}")
        
        # Manual entry form
        features = []
        # Check if features were loaded from data
        default_values = {}
        if "loaded_features" in st.session_state:
            loaded_features = st.session_state.loaded_features
            if len(loaded_features) == expected_feature_count:
                for i, val in enumerate(loaded_features):
                    default_values[i] = float(val)
                # Clear after use
                if "loaded_features" in st.session_state:
                    del st.session_state.loaded_features
                if "loaded_feature_names" in st.session_state:
                    del st.session_state.loaded_feature_names
        
        for i in range(expected_feature_count):
            label = feature_names[i] if i < len(feature_names) else f"Feature {i+1}"
            default_val = default_values.get(i, 0.0)
            val = st.number_input(
                label,
                value=default_val,
                key=f"feature_{i}",
                help=f"Feature {i+1} of {expected_feature_count}",
                step=0.01
            )
            features.append(val)
        
        if st.button("Predict", type="primary", use_container_width=True):
            if len(features) != expected_feature_count:
                st.error(f"Expected {expected_feature_count} features, but got {len(features)}")
            else:
                make_prediction(features)
    
    # Main area: Load Data and Train Models
    st.subheader("Load Data & Train Models")
    st.markdown("""
    **📋 Important:** Upload a CSV file containing stock data with the following required columns:
    - `Open Price`, `High Price`, `Low Price`, `Close Price`
    - `Total Traded Quantity` (optional)
    - `Turnover` (optional)
    
    The system will automatically train new models with your data. After training completes, 
    use the manual feature entry in the sidebar to make predictions.
    """)
    
    data_file = st.file_uploader(
        "Upload CSV file with stock data",
        type=["csv"],
        help="Upload CSV file containing stock price data with required columns"
    )
    
    if data_file:
        try:
            df = pd.read_csv(data_file)
            st.success(f"✅ Loaded {len(df)} rows, {len(df.columns)} columns")
            
            # Validate required columns
            required_cols = ["Open Price", "High Price", "Low Price", "Close Price"]
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                st.error(f"❌ Missing required columns: {', '.join(missing_cols)}")
                st.info("Please ensure your CSV file contains: " + ", ".join(required_cols))
            else:
                # Show data preview
                with st.expander("Data Preview", expanded=True):
                    st.dataframe(df.head(10))
                    st.caption(f"Total rows: {len(df)}")
                
                # Training section
                st.divider()
                st.subheader("Train Models")
                
                if len(df) < 100:
                    st.warning("⚠️ Warning: At least 100 rows are recommended for training. Current: " + str(len(df)))
                
                if st.button("🚀 Train Models with This Data", type="primary", use_container_width=True):
                    with st.spinner("Training models... This may take several minutes. Please wait..."):
                        try:
                            # Convert DataFrame to list of dictionaries
                            data_dict = df.to_dict('records')
                            
                            # Train models
                            training_result = api_client.train_models(data_dict)
                            
                            # Store results
                            st.session_state.training_result = training_result
                            st.session_state.training_complete = True
                            
                            st.success("✅ Training completed successfully!")
                            st.balloons()
                            
                            # Display results
                            st.subheader("Training Results")
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                st.metric("Model Version", training_result["version"])
                            with col2:
                                st.metric("Features", training_result["num_features"])
                            with col3:
                                st.metric("Training Samples", training_result["train_size"])
                            
                            # Model performance
                            st.subheader("Model Performance")
                            results = training_result["results"]
                            
                            metrics_df = pd.DataFrame({
                                "Model": ["LightGBM", "XGBoost", "LSTM"],
                                "Test RMSE": [
                                    results["lightgbm"]["test_rmse"],
                                    results["xgboost"]["test_rmse"],
                                    results["lstm"]["test_rmse"]
                                ],
                                "Test MAE": [
                                    results["lightgbm"]["test_mae"],
                                    results["xgboost"]["test_mae"],
                                    results["lstm"]["test_mae"]
                                ],
                                "Test R²": [
                                    results["lightgbm"]["test_r2"],
                                    results["xgboost"]["test_r2"],
                                    results["lstm"]["test_r2"]
                                ]
                            })
                            st.dataframe(metrics_df, use_container_width=True)
                            
                            st.info("💡 Models are now trained and ready! Use the sidebar to enter features manually and make predictions.")
                            
                        except Exception as e:
                            st.error(f"❌ Training failed: {str(e)}")
                            st.exception(e)
        except Exception as e:
            st.error(f"❌ Error loading file: {e}")
            st.exception(e)
    
    
    # Results section
    st.divider()
    st.subheader("Results")
    if "prediction_result" in st.session_state:
        display_prediction_result(st.session_state.prediction_result)
    else:
        st.info("Enter features manually in the sidebar or load data above to make a prediction.")


def make_prediction(features: list, feature_names: list = None):
    """Make prediction via API."""
    with st.spinner("Making prediction..."):
        try:
            result = api_client.predict(
                features=features,
                return_components=True,
                return_confidence=True
            )
            
            st.session_state.prediction_result = result
            st.session_state.feature_names = feature_names
            
            st.rerun()
        except Exception as e:
            error_msg = str(e)
            # Show more detailed error information
            if "500" in error_msg or "Internal Server Error" in error_msg:
                st.error("⚠️ Server Error: The backend encountered an error. Please check:")
                st.info("1. Ensure models are loaded (check Model Info page)\n2. Verify feature count matches expected model input\n3. Check backend logs for details")
            elif "404" in error_msg:
                st.error("⚠️ Endpoint not found. Please check if the backend is running.")
            else:
                st.error(f"❌ Prediction failed: {error_msg}")


def display_prediction_result(result: dict):
    """Display prediction results."""
    prediction = result.get("prediction", 0)
    confidence = result.get("confidence", {})
    components = result.get("components", {})
    
    # Main prediction
    st.metric("Predicted Price", f"${prediction:,.2f}")
    
    # Confidence interval
    if confidence:
        lower = confidence.get("lower", prediction * 0.95)
        upper = confidence.get("upper", prediction * 1.05)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Lower Bound", f"${lower:,.2f}")
        with col2:
            st.metric("Upper Bound", f"${upper:,.2f}")
        
        # Confidence visualization
        st.progress(0.95, text="95% Confidence Interval")
    
    # Component breakdown
    if components:
        with st.expander("Model Components"):
            xgb_pred = components.get("xgboost", 0)
            nn_residual = components.get("neural_network_residual", 0)
            weights = components.get("weights", {})
            
            st.write(f"**XGBoost Prediction:** ${xgb_pred:,.2f} (Weight: {weights.get('xgboost', 0):.1%})")
            st.write(f"**NN Residual:** ${nn_residual:,.2f} (Weight: {weights.get('neural_network', 0):.1%})")
            st.write(f"**Ensemble:** ${prediction:,.2f}")


def show_explainability_page():
    """Explainability page with SHAP visualizations."""
    st.header("Model Explainability")
    
    st.info("Understand why the model made a specific prediction using SHAP values")
    
    # Get model info to determine expected feature count
    try:
        model_info = api_client.get_active_model_info()
        expected_feature_count = model_info.get("num_features", 16)
        feature_names = model_info.get("feature_names", [])
    except Exception as e:
        st.warning(f"Could not fetch model info: {e}. Using default feature count.")
        expected_feature_count = 16
        feature_names = []
    
    # Sidebar: Manual Entry
    with st.sidebar:
        st.subheader("Manual Feature Entry")
        st.info("Enter feature values manually")
        
        if feature_names:
            with st.expander("Expected Features", expanded=False):
                st.write(f"Model expects **{expected_feature_count}** features:")
                for i, name in enumerate(feature_names, 1):
                    st.write(f"{i}. {name}")
        
        # Manual entry form
        features = []
        # Check if features were loaded from data
        default_values = {}
        if "loaded_features" in st.session_state:
            loaded_features = st.session_state.loaded_features
            if len(loaded_features) == expected_feature_count:
                for i, val in enumerate(loaded_features):
                    default_values[i] = float(val)
                # Clear after use
                if "loaded_features" in st.session_state:
                    del st.session_state.loaded_features
                if "loaded_feature_names" in st.session_state:
                    del st.session_state.loaded_feature_names
        
        for i in range(expected_feature_count):
            label = feature_names[i] if i < len(feature_names) else f"Feature {i+1}"
            default_val = default_values.get(i, 0.0)
            val = st.number_input(
                label,
                value=default_val,
                key=f"explain_feature_{i}",
                help=f"Feature {i+1} of {expected_feature_count}",
                step=0.01
            )
            features.append(val)
        
        if st.button("Explain Prediction", type="primary", use_container_width=True):
            if len(features) != expected_feature_count:
                st.error(f"Expected {expected_feature_count} features, but got {len(features)}")
            else:
                explain_prediction(features, feature_names if feature_names else None)
    
    # Main area: Instructions
    st.subheader("How to Use Explainability")
    st.markdown("""
    **To explain predictions:**
    
    1. **First, train models** with your data in the **Prediction** tab
    2. **Enter feature values** manually in the sidebar (same features used for prediction)
    3. Click **"Explain Prediction"** button in the sidebar
    4. View the explanation results below
    
    The explanation shows:
    - **Feature Importance:** Which features most influence the prediction
    - **SHAP Values:** How each feature contributes to the prediction
    - **Waterfall Plot:** Visual breakdown of feature contributions
    """)
    
    st.info("💡 **Tip:** Use the same feature values you used for prediction to understand why that prediction was made.")
    
    # Display explanation if available
    st.divider()
    st.subheader("Explanation Results")
    if "explanation_result" in st.session_state:
        display_explanation()
    else:
        st.info("Enter features manually in the sidebar or load data above to extract features, then click 'Explain Prediction'.")


def explain_prediction(features: list, feature_names: list = None):
    """Get SHAP explanation via API."""
    with st.spinner("Generating explanation..."):
        try:
            result = api_client.explain(features=features)
            st.session_state.explanation_result = result
            st.session_state.explanation_feature_names = feature_names or result.get("feature_names", [])
            st.rerun()
        except Exception as e:
            st.error(f"Explanation failed: {e}")


def display_explanation():
    """Display SHAP explanation results."""
    if "explanation_result" not in st.session_state:
        return
    
    result = st.session_state.explanation_result
    feature_names = st.session_state.explanation_feature_names or result.get("feature_names", [])
    shap_values = result.get("shap_values", [])
    
    # Feature importance
    st.subheader("Feature Importance")
    importance = result.get("feature_importance", {})
    
    if importance:
        fig = plot_feature_importance(importance)
        st.plotly_chart(fig, use_container_width=True)
    
    # SHAP waterfall
    if shap_values and feature_names:
        st.subheader("SHAP Waterfall Plot")
        fig = plot_shap_waterfall(shap_values, feature_names, result.get("base_value", 0))
        st.plotly_chart(fig, use_container_width=True)
    
    # Detailed table
    with st.expander("Detailed SHAP Values"):
        df = pd.DataFrame({
            "Feature": feature_names,
            "SHAP Value": shap_values
        })
        df = df.sort_values("SHAP Value", key=abs, ascending=False)
        st.dataframe(df, use_container_width=True)


def show_how_to_use_page():
    """How to Use guide page."""
    st.header("📖 How to Use This Application")
    
    st.markdown("""
    ## Welcome to the Stock Prediction Dashboard
    
    This application uses advanced machine learning models (LightGBM, XGBoost, and LSTM) 
    to predict stock prices. Follow these steps to get started.
    """)
    
    st.divider()
    
    st.subheader("🚀 Step 1: Prepare Your Data")
    st.markdown("""
    Prepare a CSV file with your stock data. The file **must** contain the following columns:
    
    **Required Columns:**
    - `Open Price` - Opening price of the stock
    - `High Price` - Highest price during the day
    - `Low Price` - Lowest price during the day  
    - `Close Price` - Closing price of the stock
    
    **Optional Columns:**
    - `Total Traded Quantity` - Volume of shares traded
    - `Turnover` - Total turnover value
    
    **Data Requirements:**
    - Minimum 100 rows of data (more is better)
    - Data should be in chronological order
    - No missing values in required columns
    """)
    
    st.code("""
Example CSV format:
Date,Open Price,High Price,Low Price,Close Price,Total Traded Quantity,Turnover
2024-01-01,100.5,102.3,99.8,101.2,1000000,101200000
2024-01-02,101.2,103.1,100.5,102.8,1200000,123360000
...
    """, language="csv")
    
    st.divider()
    
    st.subheader("🎯 Step 2: Train Models with Your Data")
    st.markdown("""
    1. Go to the **Prediction** tab
    2. Click **"Upload CSV file"** in the "Load Data & Train Models" section
    3. Select your prepared CSV file
    4. Review the data preview to ensure it loaded correctly
    5. Click **"🚀 Train Models with This Data"** button
    
    **What happens during training:**
    - The system engineers features from your raw data (technical indicators, moving averages, etc.)
    - Three models are trained: LightGBM, XGBoost, and LSTM
    - Models are validated and tested
    - The best model version is automatically saved and activated
    
    **Training time:** Typically 2-5 minutes depending on data size
    """)
    
    st.info("💡 **Tip:** Make sure your data has at least 100 rows for best results. More data = better model performance!")
    
    st.divider()
    
    st.subheader("📊 Step 3: Make Predictions")
    st.markdown("""
    After training is complete, you can make predictions:
    
    1. **Use Manual Entry (Sidebar):**
       - Enter feature values manually in the sidebar
       - The sidebar shows all required features
       - Click **"Predict"** button
       
    2. **Feature Values:**
       - The model expects specific features (typically 16 features)
       - These include: Open Price, High Price, Low Price, Close Price, 
         technical indicators (RSI, MACD, Bollinger Bands), moving averages, etc.
       - You can see the expected features in the "Expected Features" expander
    
    **Prediction Results:**
    - Predicted stock price
    - Confidence intervals (upper and lower bounds)
    - Individual model predictions (LightGBM, XGBoost, LSTM)
    - Ensemble prediction (weighted combination)
    """)
    
    st.divider()
    
    st.subheader("🔍 Step 4: Understand Predictions (Explainability)")
    st.markdown("""
    To understand **why** the model made a specific prediction:
    
    1. Go to the **Explainability** tab
    2. Enter feature values manually in the sidebar (same as prediction)
    3. Click **"Explain Prediction"** button
    
    **You'll see:**
    - **Feature Importance:** Which features most influence the prediction
    - **SHAP Waterfall Plot:** Visual breakdown of how each feature contributes
    - **Detailed SHAP Values:** Exact contribution of each feature
    
    This helps you understand:
    - Which factors drive the prediction
    - Whether the prediction is reliable
    - What to focus on when analyzing stocks
    """)
    
    st.divider()
    
    st.subheader("📋 Workflow Summary")
    st.markdown("""
    ```
    1. Prepare CSV file with stock data
       ↓
    2. Upload data → Train models
       ↓
    3. Enter features manually in sidebar
       ↓
    4. Click "Predict" → View results
       ↓
    5. (Optional) Use Explainability tab to understand predictions
    ```
    """)
    
    st.divider()
    
    st.subheader("❓ Frequently Asked Questions")
    
    with st.expander("What if I don't have all the required columns?"):
        st.markdown("""
        The system requires at minimum: Open Price, High Price, Low Price, and Close Price.
        Other columns like Total Traded Quantity and Turnover are optional but recommended.
        """)
    
    with st.expander("How much data do I need?"):
        st.markdown("""
        Minimum: 100 rows. Recommended: 500+ rows for better model performance.
        More historical data generally leads to more accurate predictions.
        """)
    
    with st.expander("How long does training take?"):
        st.markdown("""
        Typically 2-5 minutes for datasets with 100-1000 rows. Larger datasets may take longer.
        The system trains 3 models (LightGBM, XGBoost, LSTM) which takes time but ensures accuracy.
        """)
    
    with st.expander("Can I use the same data multiple times?"):
        st.markdown("""
        Yes! Each training creates a new model version. You can train with updated data anytime.
        The latest trained model is automatically used for predictions.
        """)
    
    with st.expander("What if training fails?"):
        st.markdown("""
        Common issues:
        - Missing required columns → Check your CSV format
        - Insufficient data → Need at least 100 rows
        - Invalid data types → Ensure prices are numeric
        - Check the error message for specific details
        """)
    
    with st.expander("How accurate are the predictions?"):
        st.markdown("""
        Accuracy depends on:
        - Quality and quantity of training data
        - Market conditions (volatile markets are harder to predict)
        - Feature quality
        
        The system shows RMSE, MAE, and R² metrics after training so you can evaluate model performance.
        """)
    
    st.divider()
    
    st.subheader("🆘 Need Help?")
    st.markdown("""
    - Check that your CSV file has the correct format
    - Ensure all required columns are present
    - Verify data has no missing values in required columns
    - Make sure you have at least 100 rows of data
    - Check the backend logs if training fails
    """)
    
    st.success("🎉 You're all set! Start by uploading your data in the Prediction tab.")


if __name__ == "__main__":
    main()

