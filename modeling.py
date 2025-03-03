from sklearn.model_selection import train_test_split, TimeSeriesSplit, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import os

def build_model(merged_df, model_type='linear', tune_hyperparameters=False):
    """
    Build a regression model for electricity demand forecasting
    
    Parameters:
    -----------
    merged_df : pandas DataFrame
        The merged and preprocessed dataframe
    model_type : str
        Type of model to build: 'linear', 'ridge', 'lasso', 'rf' (random forest), or 'gb' (gradient boosting)
    tune_hyperparameters : bool
        Whether to perform hyperparameter tuning
        
    Returns:
    --------
    model : fitted model object
    y_test : array-like
        Actual test values
    y_pred : array-like
        Predicted test values
    mse : float
        Mean squared error
    rmse : float
        Root mean squared error
    r2 : float
        R-squared score
    """
    # Ensure timestamp column is datetime type for time-based split
    if 'timestamp' in merged_df.columns:
        merged_df['timestamp'] = pd.to_datetime(merged_df['timestamp'])
        # Sort by timestamp to ensure proper time series split
        merged_df = merged_df.sort_values('timestamp')
    
    # Identify numeric and categorical columns
    numeric_features = ['temperature', 'hour', 'day_of_week', 'is_weekend', 'month']
    categorical_features = ['season']
    
    # Filter out any columns that don't exist in the dataframe
    numeric_features = [col for col in numeric_features if col in merged_df.columns]
    categorical_features = [col for col in categorical_features if col in merged_df.columns]
    
    # Combine all available features
    feature_cols = numeric_features + categorical_features
    
    # Prepare data
    X = merged_df[feature_cols]
    y = merged_df['electricity_demand']
    
    # Split data (time series aware split for forecasting tasks)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False, random_state=42
    )
    
    # Create preprocessing steps for numeric and categorical data
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])
    
    # Add polynomial features for non-tree models
    if model_type in ['linear', 'ridge', 'lasso']:
        numeric_transformer = Pipeline(steps=[
            ('scaler', StandardScaler()),
            ('poly', PolynomialFeatures(degree=2, include_bias=False))
        ])
    
    # Categorical transformer with one-hot encoding
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'))
    ])
    
    # Combine transformers
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='drop'  # Drop any columns not specified
    )
    
    # Select model based on type
    if model_type == 'linear':
        model = LinearRegression()
        param_grid = {}  # Linear regression doesn't have hyperparameters to tune
        
    elif model_type == 'ridge':
        model = Ridge()
        param_grid = {'model__alpha': [0.01, 0.1, 1.0, 10.0, 100.0]}
        
    elif model_type == 'lasso':
        model = Lasso(max_iter=10000)
        param_grid = {'model__alpha': [0.001, 0.01, 0.1, 1.0, 10.0]}
        
    elif model_type == 'rf':
        model = RandomForestRegressor(random_state=42)
        param_grid = {
            'model__n_estimators': [50, 100, 200],
            'model__max_depth': [None, 10, 20, 30],
            'model__min_samples_split': [2, 5, 10]
        }
        
    elif model_type == 'gb':
        model = GradientBoostingRegressor(random_state=42)
        param_grid = {
            'model__n_estimators': [50, 100, 200],
            'model__learning_rate': [0.01, 0.1, 0.2],
            'model__max_depth': [3, 5, 7]
        }
    
    # Create final pipeline with preprocessing and model
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', model)
    ])
    
    # Hyperparameter tuning using time series cross-validation if requested
    if tune_hyperparameters and param_grid:
        tscv = TimeSeriesSplit(n_splits=5)
        grid_search = GridSearchCV(
            pipeline, param_grid, cv=tscv, scoring='neg_mean_squared_error',
            n_jobs=-1, verbose=1
        )
        grid_search.fit(X_train, y_train)
        model = grid_search.best_estimator_
        print(f"Best parameters: {grid_search.best_params_}")
    else:
        model = pipeline.fit(X_train, y_train)
    
    # Predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    
    # Calculate MAPE safely (avoiding division by zero)
    mask = y_test != 0
    mape = np.mean(np.abs((y_test[mask] - y_pred[mask]) / y_test[mask])) * 100
    
    # Print additional metrics for reference
    print(f"Model: {model_type}")
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R²: {r2:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"MAPE: {mape:.4f}%")
    
    # Get feature importance if available
    feature_importance = get_feature_importance(model, X)
    if feature_importance is not None:
        print("Feature Importance:")
        for feature, importance in feature_importance.items():
            print(f"  {feature}: {importance:.4f}")
    
    return model, y_test, y_pred, mse, rmse, r2, mae, mape, feature_importance

def get_feature_importance(model, X):
    """Extract feature importance from the model if available"""
    feature_importance = None
    
    # For pipeline models
    if hasattr(model, 'named_steps') and 'model' in model.named_steps:
        model_step = model.named_steps['model']
        
        # For tree-based models with feature_importances_
        if hasattr(model_step, 'feature_importances_'):
            # Get feature names after preprocessing
            if hasattr(model, 'feature_names_in_'):
                feature_names = model.feature_names_in_
            else:
                feature_names = [f'Feature {i}' for i in range(len(model_step.feature_importances_))]
                
            feature_importance = dict(zip(feature_names, model_step.feature_importances_))
            
        # For linear models with coefficients
        elif hasattr(model_step, 'coef_'):
            if hasattr(model, 'feature_names_in_'):
                feature_names = model.feature_names_in_
            else:
                feature_names = [f'Feature {i}' for i in range(len(model_step.coef_))]
                
            feature_importance = dict(zip(feature_names, np.abs(model_step.coef_)))
    
    return feature_importance

def save_model(model, filename='electricity_demand_model.pkl'):
    """Save the trained model to disk"""
    joblib.dump(model, filename)
    print(f"Model saved to {filename}")

def load_model(filename='electricity_demand_model.pkl'):
    """Load a trained model from disk"""
    if os.path.exists(filename):
        return joblib.load(filename)
    else:
        print(f"Model file {filename} not found")
        return None

def plot_feature_importance(feature_importance, title="Feature Importance"):
    """Plot feature importance as a bar chart"""
    if feature_importance is None:
        return None
        
    # Convert to DataFrame and sort
    fi_df = pd.DataFrame({
        'Feature': list(feature_importance.keys()),
        'Importance': list(feature_importance.values())
    })
    fi_df = fi_df.sort_values('Importance', ascending=False)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(fi_df['Feature'], fi_df['Importance'])
    ax.set_xlabel('Importance')
    ax.set_title(title)
    
    return fig

def compare_models(merged_df, models=['linear', 'ridge', 'rf']):
    """Compare multiple model types on the same dataset"""
    results = {}
    
    for model_type in models:
        print(f"\nTraining {model_type} model...")
        model, y_test, y_pred, mse, rmse, r2, mae, mape, _ = build_model(merged_df, model_type=model_type)
        results[model_type] = {
            'mse': mse,
            'rmse': rmse,
            'r2': r2,
            'mae': mae,
            'mape': mape
        }
    
    # Convert results to DataFrame for easy comparison
    results_df = pd.DataFrame(results).transpose()
    print("\nModel Comparison:")
    print(results_df)
    
    return results_df