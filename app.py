import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from modeling import build_model
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.preprocessing import StandardScaler
from modeling import build_model, compare_models, plot_feature_importance, save_model
from eda import ElectricityDataAnalyzer, run_eda
from outlier_handling import OutlierDetector, detect_and_handle_outliers
from forecasting import ElectricityDemandForecaster, create_forecast_scenario, compare_forecast_scenarios



# Page configuration
st.set_page_config(page_title="Electricity Demand Forecasting", layout="wide")

# Load processed data
@st.cache_data
def load_data():
    df = pd.read_csv('merged_data.csv')
    # Convert timestamp to datetime if it's not already
    if df['timestamp'].dtype == 'object':
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df

# Fix for displaying dataframes with timestamps in Streamlit
def prepare_df_for_display(df):
    display_df = df.copy()
    if 'timestamp' in display_df.columns and pd.api.types.is_datetime64_any_dtype(display_df['timestamp']):
        display_df['timestamp'] = display_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
    return display_df

merged_df = load_data()

# Create sidebar navigation instead of tabs
st.sidebar.title("Navigation")

# Navigation options in the sidebar
selected_page = st.sidebar.radio(
    "",
    ["📊 Data Overview", "📈 EDA", "🔍 Outlier Analysis", "🤖 Model Performance", "🔮 Forecasting"]
)

# Add separator between navigation and filters
st.sidebar.markdown("---")

# Sidebar for filters
st.sidebar.header("Data Filters")

# Date range selector
min_date = merged_df['timestamp'].min()
max_date = merged_df['timestamp'].max()
start_date = st.sidebar.date_input("Start Date", min_date, min_value=min_date, max_value=max_date)
end_date = st.sidebar.date_input("End Date", max_date, min_value=min_date, max_value=max_date)

# Convert to datetime for filtering
start_date = pd.Timestamp(start_date)
end_date = pd.Timestamp(end_date)
filtered_df = merged_df[(merged_df['timestamp'] >= start_date) & (merged_df['timestamp'] <= end_date)]

# Weather condition filter if available
if 'weather_condition' in merged_df.columns:
    weather_options = ['All'] + list(merged_df['weather_condition'].unique())
    selected_weather = st.sidebar.selectbox("Weather Condition", weather_options)
    if selected_weather != 'All':
        filtered_df = filtered_df[filtered_df['weather_condition'] == selected_weather]

# Main title
st.title('Electricity Demand Forecasting Dashboard')

# Main content based on sidebar selection
if selected_page == "📊 Data Overview":
    st.header("Data Summary")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Records", f"{filtered_df.shape[0]:,}")
    with col2:
        st.metric("Total Features", filtered_df.shape[1])
    with col3:
        st.metric("Date Range", f"{filtered_df['timestamp'].min().date()} to {filtered_df['timestamp'].max().date()}")
    
    st.subheader("Sample Data")
    st.dataframe(prepare_df_for_display(filtered_df.head()))
    
    st.subheader("Statistical Summary")
    numeric_columns = [col for col in filtered_df.columns if col != 'timestamp' and col != 'timestamp_dt']
    st.dataframe(filtered_df[numeric_columns].describe())
    
    # Missing values analysis
    st.subheader("Missing Values")
    missing_data = filtered_df.isna().sum().reset_index()
    missing_data.columns = ['Feature', 'Missing Count']
    missing_data['Missing Percentage'] = (missing_data['Missing Count'] / len(filtered_df)) * 100
    st.dataframe(missing_data)

elif selected_page == "📈 EDA":
    st.header("Exploratory Data Analysis")
    
    # Initialize the EDA analyzer
    analyzer = ElectricityDataAnalyzer(filtered_df)
    
    # Create sub-tabs for organization
    eda_tabs = st.tabs(["Time Series", "Patterns", "Distribution", "Correlation", "Advanced"])
    
    # Time Series Analysis
    with eda_tabs[0]:
        st.subheader("Electricity Demand Over Time")
        
        # Add resample option
        resample_options = {
            'None': None,
            'Hourly': 'h', 
            'Daily': 'D', 
            'Weekly': 'W', 
            'Monthly': 'M'
        }
        resample_freq = st.selectbox("Resample Frequency", 
                                    options=list(resample_options.keys()),
                                    format_func=lambda x: x)
        
        # Get time series plot
        fig = analyzer.plot_time_series(resample_freq=resample_options[resample_freq])
        st.pyplot(fig)
    
    # Time-based Patterns
    with eda_tabs[1]:
        st.subheader("Time-based Patterns")
        
        # Get time patterns analysis
        time_patterns = analyzer.analyze_time_patterns()
        
        # Create tabs for different patterns
        pattern_tabs = st.tabs(["Hourly", "Daily", "Monthly", "Seasonal"])
        
        with pattern_tabs[0]:
            if 'hourly' in time_patterns:
                st.dataframe(time_patterns['hourly'])
                fig, ax = plt.subplots(figsize=(10, 6))
                time_patterns['hourly']['mean'].plot(kind='bar', ax=ax)
                ax.set_title('Average Electricity Demand by Hour')
                ax.set_xlabel('Hour of Day')
                ax.set_ylabel('Demand')
                st.pyplot(fig)
        
        with pattern_tabs[1]:
            if 'daily' in time_patterns:
                st.dataframe(time_patterns['daily'])
                fig, ax = plt.subplots(figsize=(10, 6))
                time_patterns['daily']['mean'].plot(kind='bar', ax=ax)
                ax.set_title('Average Electricity Demand by Day of Week')
                ax.set_xlabel('Day of Week (0=Monday)')
                ax.set_ylabel('Demand')
                st.pyplot(fig)
                
                # Weekend vs Weekday
                if 'weekend_vs_weekday' in time_patterns:
                    st.subheader("Weekend vs. Weekday")
                    st.dataframe(time_patterns['weekend_vs_weekday'])
                    fig, ax = plt.subplots(figsize=(8, 6))
                    time_patterns['weekend_vs_weekday']['mean'].plot(kind='bar', ax=ax)
                    ax.set_title('Weekend vs. Weekday Demand')
                    ax.set_xticklabels(['Weekday', 'Weekend'])
                    st.pyplot(fig)
        
        with pattern_tabs[2]:
            if 'monthly' in time_patterns:
                st.dataframe(time_patterns['monthly'])
                fig, ax = plt.subplots(figsize=(10, 6))
                time_patterns['monthly']['mean'].plot(kind='bar', ax=ax)
                ax.set_title('Average Electricity Demand by Month')
                ax.set_xlabel('Month')
                ax.set_ylabel('Demand')
                st.pyplot(fig)
                
        with pattern_tabs[3]:
            if 'seasonal' in time_patterns:
                st.dataframe(time_patterns['seasonal'])
                fig, ax = plt.subplots(figsize=(10, 6))
                time_patterns['seasonal']['mean'].plot(kind='bar', ax=ax)
                ax.set_title('Average Electricity Demand by Season')
                ax.set_xlabel('Season')
                ax.set_ylabel('Demand')
                st.pyplot(fig)
    
    # Distribution Analysis
    with eda_tabs[2]:
        st.subheader("Distribution Analysis")
        
        # Get distribution plots
        fig = analyzer.plot_distribution()
        st.pyplot(fig)
        
        # Basic stats
        st.subheader("Statistical Summary")
        stats_df = analyzer.get_basic_stats()
        st.dataframe(stats_df)
    
    # Correlation Analysis
    with eda_tabs[3]:
        st.subheader("Correlation Analysis")
        
        # Get correlation matrix
        fig, corr_matrix = analyzer.plot_correlation_matrix()
        st.pyplot(fig)
        
        # Top correlations
        st.subheader("Top Correlations with Electricity Demand")
        if 'electricity_demand' in corr_matrix.columns:
            # Get correlations with electricity_demand
            demand_corr = corr_matrix['electricity_demand'].sort_values(ascending=False)
            # Drop self-correlation
            demand_corr = demand_corr.drop('electricity_demand')
            # Display as a bar chart
            fig, ax = plt.subplots(figsize=(10, 6))
            demand_corr.plot(kind='bar', ax=ax)
            ax.set_title('Features Correlation with Electricity Demand')
            ax.set_ylabel('Correlation Coefficient')
            st.pyplot(fig)
    
    # Advanced Analysis
    with eda_tabs[4]:
        st.subheader("Advanced Time Series Analysis")
        
        col1, col2 = st.columns(2)
        with col1:
            # Time Series Decomposition
            st.subheader("Time Series Decomposition")
            if st.checkbox("Run Decomposition Analysis", key="decomp_checkbox"):
                with st.spinner("Decomposing time series..."):
                    try:
                        # Allow user to select period
                        period = st.slider("Seasonality Period (hours)", 
                                         min_value=6, max_value=48, value=24, step=6, key="seasonality_period_slider")
                        fig, decomp = analyzer.perform_time_series_decomposition(period=period)
                        st.pyplot(fig)
                    except Exception as e:
                        st.error(f"Error in decomposition: {e}")
        
        with col2:
            # Stationarity Test
            st.subheader("Stationarity Test (ADF)")
            if st.checkbox("Check Stationarity"):
                with st.spinner("Running ADF test..."):
                    adf_result, is_stationary = analyzer.check_stationarity()
                    st.write(f"Series is {'stationary' if is_stationary else 'non-stationary'}")
                    st.write("ADF Test Results:")
                    st.write(f"Test Statistic: {adf_result['Test Statistic']:.4f}")
                    st.write(f"p-value: {adf_result['p-value']:.4f}")
                    st.write("Critical Values:")
                    for key, value in adf_result['Critical Values'].items():
                        st.write(f"  {key}: {value:.4f}")
        
        # Autocorrelation
        st.subheader("Autocorrelation Analysis")
        if st.checkbox("Show Autocorrelation Plots", key="acf_checkbox"):
            lag_value = st.slider("Number of Lags", min_value=12, max_value=96, value=48, step=12, key="acf_lags_slider")
            fig = analyzer.plot_autocorrelation(lags=lag_value)
            st.pyplot(fig)
        
        # Weather Impact Analysis
        if 'temperature' in filtered_df.columns:
            st.subheader("Weather Impact Analysis")
            fig, corr = analyzer.analyze_weather_impact()
            st.write(f"Correlation between temperature and demand: {corr:.4f}")
            st.pyplot(fig)
     

elif selected_page == "🔍 Outlier Analysis":
    st.header("Outlier Analysis")
    
    # Initialize the outlier detector
    detector = OutlierDetector(filtered_df)
    
    # Select column for outlier detection
    outlier_column = st.selectbox(
        "Select column for outlier detection",
        [col for col in filtered_df.columns if filtered_df[col].dtype in ['int64', 'float64']]
    )
    
    # Multiple outlier detection methods
    st.subheader("Outlier Detection Methods")
    method_tabs = st.tabs(["IQR", "Z-Score", "Isolation Forest", "LOF", "Comparison"])
    
    # IQR Method
    with method_tabs[0]:
        st.write("### Interquartile Range (IQR) Method")
        iqr_factor = st.slider("IQR Factor", 1.0, 3.0, 1.5, 0.1, key="iqr_factor_slider")
        
        if st.button("Run IQR Detection", key="iqr_btn"):
            outlier_indices = detector.detect_iqr_outliers(outlier_column, factor=iqr_factor)
            st.write(f"Found {len(outlier_indices)} outliers ({detector.outliers_info['iqr']['percentage']:.2f}% of data)")
            
            fig = detector.plot_outliers(outlier_column, method='iqr')
            st.pyplot(fig)
            
            if len(outlier_indices) > 0:
                st.write("Sample outliers:")
                st.dataframe(prepare_df_for_display(filtered_df.loc[outlier_indices].head(10)))
    
    # Z-Score Method
    with method_tabs[1]:
        st.write("### Z-Score Method")
        z_threshold = st.slider("Z-Score Threshold", 1.0, 5.0, 3.0, 0.1, key="zscore_threshold_slider")
        
        if st.button("Run Z-Score Detection", key="zscore_btn"):
            outlier_indices = detector.detect_zscore_outliers(outlier_column, threshold=z_threshold)
            st.write(f"Found {len(outlier_indices)} outliers ({detector.outliers_info['zscore']['percentage']:.2f}% of data)")
            
            fig = detector.plot_outliers(outlier_column, method='zscore')
            st.pyplot(fig)
            
            if len(outlier_indices) > 0:
                st.write("Sample outliers:")
                st.dataframe(prepare_df_for_display(filtered_df.loc[outlier_indices].head(10)))
    
    # Isolation Forest
    with method_tabs[2]:
        st.write("### Isolation Forest Method")
        iso_contamination = st.slider("Contamination", 0.01, 0.1, 0.05, 0.01, key="isolation_forest_contamination")
        
        if st.button("Run Isolation Forest Detection", key="iso_btn"):
            # Select features for multivariate outlier detection
            numeric_cols = filtered_df.select_dtypes(include=['number']).columns.tolist()
            features = st.multiselect(
                "Select features for outlier detection", 
                numeric_cols,
                default=[outlier_column, 'temperature'] if 'temperature' in numeric_cols else [outlier_column]
            )
            
            if len(features) >= 1:
                outlier_indices = detector.detect_isolation_forest(features, contamination=iso_contamination)
                st.write(f"Found {len(outlier_indices)} outliers ({detector.outliers_info['isolation_forest']['percentage']:.2f}% of data)")
                
                fig = detector.plot_outliers(outlier_column, method='isolation_forest')
                st.pyplot(fig)
                
                if len(outlier_indices) > 0:
                    st.write("Sample outliers:")
                    st.dataframe(prepare_df_for_display(filtered_df.loc[outlier_indices].head(10)))
            else:
                st.warning("Please select at least one feature for Isolation Forest")
    
    # LOF Method
    with method_tabs[3]:
        st.write("### Local Outlier Factor (LOF) Method")
        lof_contamination = st.slider("Contamination", 0.01, 0.1, 0.05, 0.01, key="lof_contamination")
        
        if st.button("Run LOF Detection", key="lof_btn"):
            # Select features for multivariate outlier detection
            numeric_cols = filtered_df.select_dtypes(include=['number']).columns.tolist()
            features = st.multiselect(
                "Select features for outlier detection", 
                numeric_cols,
                default=[outlier_column, 'temperature'] if 'temperature' in numeric_cols else [outlier_column]
            )
            
            if len(features) >= 1:
                outlier_indices = detector.detect_lof_outliers(features, contamination=lof_contamination)
                st.write(f"Found {len(outlier_indices)} outliers ({detector.outliers_info['lof']['percentage']:.2f}% of data)")
                
                fig = detector.plot_outliers(outlier_column, method='lof')
                st.pyplot(fig)
                
                if len(outlier_indices) > 0:
                    st.write("Sample outliers:")
                    st.dataframe(prepare_df_for_display(filtered_df.loc[outlier_indices].head(10)))
            else:
                st.warning("Please select at least one feature for LOF")
    
    # Method Comparison
    with method_tabs[4]:
        st.write("### Method Comparison")
        
        if st.button("Run All Methods for Comparison", key="compare_btn"):
            # Run all methods
            detector.detect_iqr_outliers(outlier_column)
            detector.detect_zscore_outliers(outlier_column)
            
            # For multivariate methods, use default features
            numeric_cols = filtered_df.select_dtypes(include=['number']).columns.tolist()
            multi_features = [outlier_column, 'temperature'] if 'temperature' in numeric_cols else [outlier_column]
            detector.detect_isolation_forest(multi_features)
            detector.detect_lof_outliers(multi_features)
            
            # Get summary
            summary = detector.get_summary()
            st.dataframe(summary)
            
            # Plot all outliers
            fig = detector.plot_outliers(outlier_column, method='all')
            st.pyplot(fig)
    
    # Outlier Handling
    st.subheader("Outlier Handling")
    if len(detector.outliers_info) > 0:
        col1, col2 = st.columns(2)
        
        with col1:
            # Select method for handling
            handling_method = st.selectbox(
                "Select outlier detection method",
                options=list(detector.outliers_info.keys())
            )
        
        with col2:
            # Select strategy
            handling_strategy = st.selectbox(
                "Select handling strategy",
                options=["cap", "remove", "mean", "median"]
            )
        
        if st.button("Handle Outliers", key="handle_btn"):
            try:
                original_df = filtered_df.copy()
                cleaned_df = detector.handle_outliers(handling_method, column=outlier_column, strategy=handling_strategy)
                
                st.write(f"Applied {handling_strategy} strategy to {len(detector.outliers_info[handling_method]['indices'])} outliers")
                
                # Compare before and after
                fig = detector.compare_before_after(original_df, cleaned_df, outlier_column)
                st.pyplot(fig)
                
                # Impact on statistics
                st.write("Impact on Statistics:")
                original_stats = original_df[outlier_column].describe()
                cleaned_stats = cleaned_df[outlier_column].describe()
                
                stats_comparison = pd.DataFrame({
                    'Before': original_stats,
                    'After': cleaned_stats,
                    'Difference': cleaned_stats - original_stats
                })
                st.dataframe(stats_comparison)
                
            except Exception as e:
                st.error(f"Error handling outliers: {e}")
    else:
        st.info("Run outlier detection first using one of the methods above.")
   

elif selected_page == "🤖 Model Performance":
    st.header("Regression Model Performance")
    
    # Model selection and configuration
    st.subheader("Model Configuration")
    col1, col2 = st.columns(2)
    with col1:
        model_type = st.selectbox(
            "Select Model Type", 
            options=["linear", "ridge", "lasso", "rf", "gb"],
            format_func=lambda x: {
                "linear": "Linear Regression", 
                "ridge": "Ridge Regression",
                "lasso": "Lasso Regression",
                "rf": "Random Forest",
                "gb": "Gradient Boosting"
            }.get(x, x)
        )
    with col2:
        tune_hyperparameters = st.checkbox("Tune Hyperparameters", value=False, 
                                          help="Enable GridSearchCV to find optimal hyperparameters (takes longer to run)")
    
    # Build model using filtered data if user clicks button
    if st.button("Run Model on Filtered Data"):
        with st.spinner(f"Training {model_type} model{'with hyperparameter tuning' if tune_hyperparameters else ''}..."):
            try:
                # Create a copy of filtered_df with timestamp_dt moved to timestamp for modeling
                model_df = filtered_df.copy()
                if 'timestamp_dt' in model_df.columns:
                    model_df['timestamp'] = model_df['timestamp_dt']
                
                # Call enhanced build_model function with all parameters
                model, y_test, y_pred, mse, rmse, r2, mae, mape, feature_importance = build_model(
                    model_df, 
                    model_type=model_type,
                    tune_hyperparameters=tune_hyperparameters
                )
                
                # Display metrics in two rows for better visibility
                st.subheader("Model Performance Metrics")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("R² Score", f"{r2:.4f}")
                with col2:
                    st.metric("RMSE", f"{rmse:.4f}")
                with col3:
                    st.metric("MSE", f"{mse:.4f}")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("MAE", f"{mae:.4f}")
                with col2:
                    st.metric("MAPE", f"{mape:.2f}%")
                with col3:
                    if tune_hyperparameters and hasattr(model, 'best_params_'):
                        st.write("Best Parameters:")
                        st.json(model.best_params_)
                
                # Actual vs Predicted plot
                st.subheader("Actual vs Predicted Values")
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.scatter(y_test, y_pred, alpha=0.6)
                ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
                ax.set_xlabel('Actual Demand')
                ax.set_ylabel('Predicted Demand')
                ax.set_title('Actual vs Predicted Electricity Demand')
                st.pyplot(fig)
                
                # Residual analysis
                st.subheader("Residual Analysis")
                residuals = y_test - y_pred
                
                col1, col2 = st.columns(2)
                with col1:
                    # Residuals vs Fitted plot
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.scatter(y_pred, residuals, alpha=0.6)
                    ax.axhline(y=0, color='r', linestyle='-')
                    ax.set_xlabel('Predicted Values')
                    ax.set_ylabel('Residuals')
                    ax.set_title('Residuals vs Fitted Values')
                    st.pyplot(fig)
                    
                with col2:
                    # Residual distribution
                    fig, ax = plt.subplots(figsize=(10, 6))
                    sns.histplot(residuals, kde=True, ax=ax)
                    ax.set_title('Distribution of Residuals')
                    st.pyplot(fig)
                
                # Feature Importance - use the function from modeling.py
                if feature_importance:
                    st.subheader("Feature Importance")
                    fig = plot_feature_importance(feature_importance)
                    st.pyplot(fig)
                    
                # Model saving option
                if st.button('Save Model'):
                    try:
                        model_filename = f'electricity_demand_{model_type}_model.pkl'
                        save_model(model, model_filename)
                        st.success(f"Model saved as {model_filename}")
                    except Exception as e:
                        st.error(f"Error saving model: {e}")
                    
            except Exception as e:
                st.error(f"Error during model training: {e}")
                st.error("Stack trace:")
                st.exception(e)
    
    # Add model comparison section
    st.subheader("Model Comparison")
    if st.button("Compare Different Models"):
        with st.spinner("Training multiple models for comparison..."):
            try:
                # Create a copy of filtered_df with timestamp_dt moved to timestamp for modeling
                model_df = filtered_df.copy()
                if 'timestamp_dt' in model_df.columns:
                    model_df['timestamp'] = model_df['timestamp_dt']
                
                # Compare standard models without hyperparameter tuning
                results_df = compare_models(model_df, models=['linear', 'ridge', 'lasso', 'rf', 'gb'])
                
                # Display results table
                st.dataframe(results_df)
                
                # Plot comparison
                fig, ax = plt.subplots(figsize=(12, 6))
                results_df[['rmse', 'mae']].plot(kind='bar', ax=ax)
                ax.set_title("Model Error Metrics Comparison")
                ax.set_ylabel("Error (lower is better)")
                plt.tight_layout()
                st.pyplot(fig)
                
                # R² comparison (higher is better)
                fig, ax = plt.subplots(figsize=(10, 4))
                ax.bar(results_df.index, results_df['r2'])
                ax.set_title("R² Score by Model (higher is better)")
                ax.set_ylabel("R² Score")
                plt.tight_layout()
                st.pyplot(fig)
                
            except Exception as e:
                st.error(f"Error comparing models: {e}")
    else:
        st.info("Click 'Compare Different Models' to train and compare multiple regression models.")
        
        
elif selected_page == "🔮 Forecasting":
    st.header("Electricity Demand Forecasting")
    
    # Initialize forecaster
    forecaster = ElectricityDemandForecaster()
    
    st.subheader("Forecast Configuration")
    
    # Select forecasting method
    forecast_method = st.radio(
        "Select Forecasting Method",
        options=["Statistical (Prophet)", "Machine Learning"],
        horizontal=True,
        key="forecast_method"
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Forecast horizon
        horizon = st.slider("Forecast Horizon (hours)", 
                          min_value=24, max_value=168, value=48, step=24,
                          key="forecast_horizon")
        
        # Start date for forecast
        last_date = filtered_df['timestamp'].max()
        forecast_start = st.date_input("Forecast Start Date", 
                                     value=last_date.date(), 
                                     min_value=last_date.date(),
                                     key="forecast_start_date")
    
    with col2:
        if forecast_method == "Statistical (Prophet)":
            # Prophet-specific settings
            yearly_seasonality = st.checkbox("Yearly Seasonality", value=True, key="prophet_yearly")
            weekly_seasonality = st.checkbox("Weekly Seasonality", value=True, key="prophet_weekly")
            daily_seasonality = st.checkbox("Daily Seasonality", value=True, key="prophet_daily")
        
        else:  # Machine Learning
            # ML model selection
            ml_model_type = st.selectbox(
                "Select ML Model",
                options=["rf", "gb", "linear", "ridge", "lasso"],
                format_func=lambda x: {
                    "rf": "Random Forest",
                    "gb": "Gradient Boosting",
                    "linear": "Linear Regression",
                    "ridge": "Ridge Regression",
                    "lasso": "Lasso Regression"
                }.get(x, x),
                key="ml_model_type"
            )
    
    # Temperature scenario
    st.subheader("Temperature Scenario")
    temp_scenario = st.selectbox(
        "Select Temperature Scenario",
        options=["Normal", "High", "Low", "Custom"],
        key="temp_scenario"
    )
    
    if temp_scenario == "Custom":
        base_temp = st.slider("Base Temperature (°C)", 
                            min_value=0, max_value=30, value=20, step=1,
                            key="base_temp")
        temp_variation = st.slider("Day/Night Variation (±°C)", 
                                 min_value=0, max_value=10, value=5, step=1,
                                 key="temp_variation")
    
    # Generate forecast
    if st.button("Generate Forecast", key="generate_forecast_btn"):
        with st.spinner("Training model and generating forecast..."):
            try:
                # Create forecast start datetime
                forecast_start_dt = pd.Timestamp(forecast_start)
                
                # Create future features dataframe
                if temp_scenario == "Normal":
                    # Use average temperatures from historical data for same time of year
                    temperature_pattern = None  # Will use default pattern
                elif temp_scenario == "High":
                    temperature_pattern = None  # Will use default pattern + 5
                    base_temp = 25
                    temp_variation = 5
                elif temp_scenario == "Low":
                    temperature_pattern = None  # Will use default pattern - 5
                    base_temp = 15
                    temp_variation = 5
                else:  # Custom
                    temperature_pattern = None  # Will use provided base_temp and temp_variation
                
                future_df = forecaster.generate_future_features(
                    start_date=forecast_start_dt,
                    periods=horizon,
                    freq='H',
                    temperature_pattern=temperature_pattern
                )
                
                # If custom or special scenarios, adjust temperature
                if temp_scenario != "Normal":
                    # Simple temperature pattern: base + variation * sin(hour)
                    future_df['temperature'] = base_temp + temp_variation * np.sin(future_df['hour'] * np.pi / 12)
                
                # Fit model and generate forecast
                if forecast_method == "Statistical (Prophet)":
                    # Prepare Prophet training data
                    train_df = filtered_df.copy()
                    
                    # Fit Prophet model
                    forecaster.fit_prophet_model(
                        train_df,
                        yearly_seasonality=yearly_seasonality,
                        weekly_seasonality=weekly_seasonality,
                        daily_seasonality=daily_seasonality
                    )
                    
                    # Generate forecast
                    forecast = forecaster.forecast_with_prophet(
                        periods=horizon,
                        future_df=future_df
                    )
                    
                    # Show forecast plot with components
                    st.subheader("Forecast Results")
                    fig, components_fig = forecaster.plot_forecast(
                        forecast_df=forecast,
                        historical_df=filtered_df,
                        plot_components=True
                    )
                    st.pyplot(fig)
                    
                    # Show components
                    st.subheader("Forecast Components")
                    st.pyplot(components_fig)
                    
                else:  # Machine Learning
                    # Fit ML model
                    forecaster.fit_ml_model(filtered_df, model_type=ml_model_type)
                    
                    # Generate forecast
                    forecast = forecaster.forecast_with_ml_model(future_df)
                    
                    # Show forecast plot
                    st.subheader("Forecast Results")
                    fig = forecaster.plot_forecast(
                        forecast_df=forecast,
                        historical_df=filtered_df
                    )
                    st.pyplot(fig)
                
                # Display forecast data
                st.subheader("Forecast Data")
                display_cols = ['ds', 'yhat', 'yhat_lower', 'yhat_upper']
                if 'trend' in forecast.columns:
                    display_cols.append('trend')
                
                forecast_display = forecast[display_cols].copy()
                forecast_display = forecast_display.rename(columns={
                    'ds': 'Date',
                    'yhat': 'Forecast',
                    'yhat_lower': 'Lower Bound (95%)',
                    'yhat_upper': 'Upper Bound (95%)',
                    'trend': 'Trend'
                })
                st.dataframe(forecast_display)
                
                # Create scenarios
                st.subheader("Forecast Scenarios")
                st.write("Compare different demand scenarios")
                
                scenarios = {
                    "Base Forecast": forecast,
                    "High Demand (+20%)": create_forecast_scenario(forecast, "High Demand", 1.2),
                    "Low Demand (-20%)": create_forecast_scenario(forecast, "Low Demand", 0.8)
                }
                
                # Plot scenario comparison
                fig = compare_forecast_scenarios(scenarios, filtered_df)
                st.pyplot(fig)
                
                # Download forecast as CSV
                st.download_button(
                    label="Download Forecast CSV",
                    data=forecast_display.to_csv(index=False),
                    file_name=f"electricity_demand_forecast_{forecast_start_dt.strftime('%Y-%m-%d')}.csv",
                    mime="text/csv",
                    key="download_forecast"
                )
                
            except Exception as e:
                st.error(f"Error generating forecast: {e}")
                st.error("Stack trace:")
                st.exception(e)
    
    # Information about forecasting
    with st.expander("About Forecasting Methods"):
        st.markdown("""
        ### Statistical Forecasting (Prophet)
        Facebook Prophet is a forecasting procedure that works best with time series that have strong seasonal effects and several seasons of historical data. Prophet is robust to missing data and trend changes, and typically handles outliers well.
        
        ### Machine Learning Forecasting
        The ML-based forecasting uses the same models from the Model Performance tab but applied to future data points. The models capture relationships between features like temperature, hour of day, day of week, etc. to predict future electricity demand.
        
        ### Temperature Scenarios
        - **Normal**: Uses typical temperature patterns based on historical data.
        - **High**: Simulates a hotter than normal scenario.
        - **Low**: Simulates a cooler than normal scenario.
        - **Custom**: Allows you to manually set temperature parameters.
        """)        
    
        
# Footer with additional information
st.markdown("---")
st.markdown("**Electricity Demand Forecasting Project** | Data Science Dashboard")