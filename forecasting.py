import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os
from datetime import datetime, timedelta
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from prophet import Prophet
from sklearn.preprocessing import StandardScaler
from modeling import build_model

class ElectricityDemandForecaster:
    """
    Class for forecasting electricity demand using multiple methods.
    """
    
    def __init__(self):
        """Initialize forecasting models."""
        self.prophet_model = None
        self.ml_model = None
        self.ml_model_type = None
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        self.features = ['temperature', 'hour', 'day_of_week', 'is_weekend', 'month']
        self.categorical_features = ['season']
        
    def prepare_prophet_data(self, df):
        """Prepare data for Prophet forecasting model."""
        # Prophet requires 'ds' (date) and 'y' (target) columns
        prophet_df = df.copy()
        prophet_df = prophet_df.rename(columns={'timestamp': 'ds', 'electricity_demand': 'y'})
        return prophet_df
    
    def fit_prophet_model(self, df, yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=True):
        """Fit Facebook Prophet model for time series forecasting."""
        # Prepare data
        prophet_df = self.prepare_prophet_data(df)
        
        # Initialize and fit model
        self.prophet_model = Prophet(
            yearly_seasonality=yearly_seasonality,
            weekly_seasonality=weekly_seasonality,
            daily_seasonality=daily_seasonality,
            interval_width=0.95  # 95% confidence interval
        )
        
        # Add weather regressors if available
        if 'temperature' in df.columns:
            self.prophet_model.add_regressor('temperature')
            prophet_df['temperature'] = df['temperature']
        
        # Fit model
        self.prophet_model.fit(prophet_df)
        return self.prophet_model
    
    def forecast_with_prophet(self, periods=24, future_df=None, include_history=False):
        """Generate forecast using Prophet model."""
        if self.prophet_model is None:
            raise ValueError("Prophet model not fitted. Call fit_prophet_model first.")
        
        if future_df is not None:
            # Use provided future dataframe
            future = self.prepare_prophet_data(future_df)
        else:
            # Create future dataframe
            future = self.prophet_model.make_future_dataframe(periods=periods, freq='H')
            
        # Add regressors if they were used in training
        if 'temperature' in self.prophet_model.extra_regressors:
            if future_df is not None and 'temperature' in future_df.columns:
                future['temperature'] = future_df['temperature']
            else:
                # For demo, use last temperature values or seasonal patterns
                temp_values = future_df['temperature'].values if future_df is not None else [20] * periods
                future['temperature'] = temp_values
        
        # Make prediction
        forecast = self.prophet_model.predict(future)
        
        if not include_history:
            # Only return future predictions
            last_date = self.prophet_model.history['ds'].max()
            forecast = forecast[forecast['ds'] > last_date]
            
        return forecast
    
    def fit_ml_model(self, df, model_type='rf'):
        """Fit ML-based model for forecasting."""
        self.ml_model_type = model_type
        self.ml_model, _, _, _, _, _, _, _, _ = build_model(df, model_type=model_type)
        return self.ml_model
    
    def forecast_with_ml_model(self, future_df):
        """Generate forecast using ML model."""
        if self.ml_model is None:
            raise ValueError("ML model not fitted. Call fit_ml_model first.")
        
        # Extract features
        feature_cols = [col for col in self.features if col in future_df.columns]
        feature_cols += [col for col in self.categorical_features if col in future_df.columns]
        
        X_future = future_df[feature_cols]
        
        # Make prediction
        y_pred = self.ml_model.predict(X_future)
        
        # Create forecast dataframe
        forecast = pd.DataFrame({
            'ds': future_df['timestamp'],
            'yhat': y_pred,
            # Add placeholder for confidence intervals (ML models need extra work for this)
            'yhat_lower': y_pred * 0.9,  # Simple approximation
            'yhat_upper': y_pred * 1.1   # Simple approximation
        })
        
        return forecast
    
    def generate_future_features(self, start_date, periods=24, freq='H', temperature_pattern=None):
        """Generate future feature values for forecasting."""
        # Create date range
        future_dates = pd.date_range(start=start_date, periods=periods, freq=freq)
        
        # Create dataframe with timestamps
        future_df = pd.DataFrame({'timestamp': future_dates})
        
        # Extract time features
        future_df['hour'] = future_df['timestamp'].dt.hour
        future_df['day_of_week'] = future_df['timestamp'].dt.dayofweek
        future_df['is_weekend'] = future_df['day_of_week'].isin([5, 6]).astype(int)
        future_df['month'] = future_df['timestamp'].dt.month
        
        # Add season based on month (simple rule-based approach)
        season_map = {
            1: 'Winter', 2: 'Winter', 3: 'Spring', 4: 'Spring', 5: 'Summer', 
            6: 'Summer', 7: 'Summer', 8: 'Autumn', 9: 'Autumn', 
            10: 'Autumn', 11: 'Winter', 12: 'Winter'
        }
        future_df['season'] = future_df['month'].map(season_map)
        
        # Generate temperature values
        if temperature_pattern is not None:
            future_df['temperature'] = temperature_pattern
        else:
            # Simple temperature pattern based on hour
            future_df['temperature'] = 15 + 5 * np.sin(future_df['hour'] * np.pi / 12)
            # Add monthly variation
            future_df['temperature'] += (future_df['month'] - 1) * 1.5 - 9
        
        return future_df
    
    def evaluate_forecast(self, actual_df, forecast_df, plot=False):
        """Evaluate forecast accuracy using various metrics."""
        # Merge actual and forecast data
        eval_df = actual_df.copy()
        eval_df = eval_df.rename(columns={'timestamp': 'ds', 'electricity_demand': 'y'})
        
        merged = pd.merge(eval_df[['ds', 'y']], 
                           forecast_df[['ds', 'yhat']], 
                           on='ds', 
                           how='inner')
        
        if len(merged) == 0:
            raise ValueError("No overlapping dates between actual and forecast data")
        
        # Calculate metrics
        mae = mean_absolute_error(merged['y'], merged['yhat'])
        rmse = np.sqrt(mean_squared_error(merged['y'], merged['yhat']))
        r2 = r2_score(merged['y'], merged['yhat'])
        mape = np.mean(np.abs((merged['y'] - merged['yhat']) / merged['y'])) * 100
        
        metrics = {
            'MAE': mae,
            'RMSE': rmse,
            'R²': r2,
            'MAPE (%)': mape
        }
        
        # Plot actual vs forecast
        if plot:
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(merged['ds'], merged['y'], 'b-', label='Actual')
            ax.plot(merged['ds'], merged['yhat'], 'r-', label='Forecast')
            ax.fill_between(
                forecast_df['ds'], 
                forecast_df['yhat_lower'], 
                forecast_df['yhat_upper'],
                color='r', 
                alpha=0.2, 
                label='95% Confidence Interval'
            )
            ax.set_title('Forecast vs Actual Values')
            ax.set_xlabel('Date')
            ax.set_ylabel('Electricity Demand')
            ax.legend()
            plt.tight_layout()
            return metrics, fig
        
        return metrics
    
    def save_model(self, filename='electricity_forecaster.pkl'):
        """Save trained forecaster to disk."""
        model_data = {
            'prophet_model': self.prophet_model,
            'ml_model': self.ml_model,
            'ml_model_type': self.ml_model_type
        }
        joblib.dump(model_data, filename)
        print(f"Forecaster saved to {filename}")
    
    def load_model(self, filename='electricity_forecaster.pkl'):
        """Load trained forecaster from disk."""
        if os.path.exists(filename):
            model_data = joblib.load(filename)
            self.prophet_model = model_data['prophet_model']
            self.ml_model = model_data['ml_model']
            self.ml_model_type = model_data['ml_model_type']
            return True
        else:
            print(f"Model file {filename} not found")
            return False
            
    def plot_forecast(self, forecast_df, historical_df=None, plot_components=False):
        """Create visualization of the forecast."""
        # Main forecast plot
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot historical data if provided
        if historical_df is not None:
            historical = historical_df.copy()
            if 'timestamp' in historical.columns and 'electricity_demand' in historical.columns:
                historical = historical.rename(columns={'timestamp': 'ds', 'electricity_demand': 'y'})
            ax.plot(historical['ds'], historical['y'], 'k.', alpha=0.5, label='Historical')
        
        # Plot forecast
        ax.plot(forecast_df['ds'], forecast_df['yhat'], 'b-', label='Forecast')
        ax.fill_between(
            forecast_df['ds'], 
            forecast_df['yhat_lower'], 
            forecast_df['yhat_upper'],
            color='blue', 
            alpha=0.2, 
            label='95% Confidence Interval'
        )
        
        ax.set_title('Electricity Demand Forecast')
        ax.set_xlabel('Date')
        ax.set_ylabel('Demand')
        ax.legend()
        
        # Return component plots if using Prophet
        if plot_components and self.prophet_model is not None:
            components_fig = self.prophet_model.plot_components(forecast_df)
            return fig, components_fig
            
        return fig

# Utility functions
def create_forecast_scenario(base_forecast, scenario_name="High Demand", adjustment_factor=1.2):
    """Create a forecast scenario by adjusting the base forecast."""
    scenario = base_forecast.copy()
    scenario['yhat'] = scenario['yhat'] * adjustment_factor
    scenario['yhat_lower'] = scenario['yhat_lower'] * adjustment_factor
    scenario['yhat_upper'] = scenario['yhat_upper'] * adjustment_factor
    scenario['scenario'] = scenario_name
    return scenario

def compare_forecast_scenarios(scenarios, historical_df=None):
    """Compare multiple forecast scenarios visually."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    colors = ['blue', 'red', 'green', 'purple', 'orange']
    
    # Plot historical data if provided
    if historical_df is not None:
        historical = historical_df.copy()
        if 'timestamp' in historical.columns and 'electricity_demand' in historical.columns:
            historical = historical.rename(columns={'timestamp': 'ds', 'electricity_demand': 'y'})
        ax.plot(historical['ds'], historical['y'], 'k.', alpha=0.2, label='Historical')
    
    # Plot each scenario
    for i, (scenario_name, forecast_df) in enumerate(scenarios.items()):
        color = colors[i % len(colors)]
        ax.plot(forecast_df['ds'], forecast_df['yhat'], color=color, label=scenario_name)
        ax.fill_between(
            forecast_df['ds'], 
            forecast_df['yhat_lower'], 
            forecast_df['yhat_upper'],
            color=color, 
            alpha=0.1
        )
    
    ax.set_title('Comparison of Forecast Scenarios')
    ax.set_xlabel('Date')
    ax.set_ylabel('Electricity Demand')
    ax.legend()
    
    return fig