import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import scipy.stats as stats

class ElectricityDataAnalyzer:
    """
    Class for performing detailed exploratory data analysis on electricity demand data.
    """
    
    def __init__(self, df):
        """Initialize with a dataframe containing electricity demand data."""
        self.df = df.copy()
        if 'timestamp' in self.df.columns and not pd.api.types.is_datetime64_any_dtype(self.df['timestamp']):
            self.df['timestamp'] = pd.to_datetime(self.df['timestamp'])
        
    def get_basic_stats(self):
        """Generate basic statistical summaries of the data."""
        numeric_cols = self.df.select_dtypes(include=['number'])
        
        # Calculate additional statistics
        stats_df = numeric_cols.describe().T
        
        # Add skewness and kurtosis
        stats_df['skewness'] = numeric_cols.skew()
        stats_df['kurtosis'] = numeric_cols.kurtosis()
        
        return stats_df
    
    def analyze_time_patterns(self):
        """Analyze electricity demand by different time components."""
        time_patterns = {}
        
        # Hourly patterns
        hourly = self.df.groupby('hour')['electricity_demand'].agg(['mean', 'std', 'min', 'max'])
        time_patterns['hourly'] = hourly
        
        # Daily patterns (day of week)
        if 'day_of_week' in self.df.columns:
            daily = self.df.groupby('day_of_week')['electricity_demand'].agg(['mean', 'std', 'min', 'max'])
            time_patterns['daily'] = daily
        
        # Monthly patterns
        if 'month' in self.df.columns:
            monthly = self.df.groupby('month')['electricity_demand'].agg(['mean', 'std', 'min', 'max'])
            time_patterns['monthly'] = monthly
        
        # Seasonal patterns
        if 'season' in self.df.columns:
            seasonal = self.df.groupby('season')['electricity_demand'].agg(['mean', 'std', 'min', 'max'])
            time_patterns['seasonal'] = seasonal
        
        # Weekend vs. Weekday
        if 'is_weekend' in self.df.columns:
            weekend_vs_weekday = self.df.groupby('is_weekend')['electricity_demand'].agg(['mean', 'std', 'min', 'max'])
            time_patterns['weekend_vs_weekday'] = weekend_vs_weekday
            
        return time_patterns
    
    def plot_time_series(self, resample_freq=None, figsize=(12, 6)):
        """
        Plot electricity demand as a time series.
        
        Parameters:
        -----------
        resample_freq : str, optional
            Frequency string for resampling (e.g., 'D' for daily, 'W' for weekly)
        figsize : tuple, optional
            Figure size (width, height)
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create a copy for plotting
        plot_df = self.df.copy()
        plot_df = plot_df.set_index('timestamp')
        
        if resample_freq:
            # Resample data to reduce plot noise
            plot_df = plot_df['electricity_demand'].resample(resample_freq).mean()
            title_suffix = f" (Resampled to {resample_freq})"
        else:
            plot_df = plot_df['electricity_demand']
            title_suffix = ""
        
        plot_df.plot(ax=ax)
        ax.set_title(f"Electricity Demand Over Time{title_suffix}")
        ax.set_ylabel("Demand")
        ax.set_xlabel("Time")
        
        return fig
    
    def plot_distribution(self, figsize=(12, 8)):
        """Plot the distribution of electricity demand."""
        fig, axs = plt.subplots(2, 1, figsize=figsize)
        
        # Histogram with KDE
        sns.histplot(self.df['electricity_demand'], kde=True, ax=axs[0])
        axs[0].set_title('Distribution of Electricity Demand')
        
        # Q-Q plot to check for normality
        stats.probplot(self.df['electricity_demand'], plot=axs[1])
        axs[1].set_title('Q-Q Plot (Check for Normality)')
        
        plt.tight_layout()
        return fig
    
    def plot_correlation_matrix(self, figsize=(10, 8)):
        """Plot correlation matrix for numeric columns."""
        numeric_df = self.df.select_dtypes(include=['number'])
        corr_matrix = numeric_df.corr()
        
        # Create mask for upper triangle
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        
        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(corr_matrix, mask=mask, annot=True, fmt=".2f", cmap='coolwarm', ax=ax)
        ax.set_title('Correlation Matrix')
        
        return fig, corr_matrix
    
    def perform_time_series_decomposition(self, period=24):
        """
        Perform time series decomposition to extract trend, seasonality, and residual components.
        
        Parameters:
        -----------
        period : int, optional
            Period for seasonality detection (default is 24 for hourly data)
        """
        # Prepare data for decomposition (needs regular time series)
        ts_data = self.df.set_index('timestamp')['electricity_demand'].resample('h').mean()
        ts_data = ts_data.dropna()
        
        # Perform decomposition
        decomposition = seasonal_decompose(ts_data, model='additive', period=period)
        
        # Create plot
        fig, axes = plt.subplots(4, 1, figsize=(12, 12))
        
        decomposition.observed.plot(ax=axes[0])
        axes[0].set_title('Observed')
        axes[0].set_ylabel('Demand')
        
        decomposition.trend.plot(ax=axes[1])
        axes[1].set_title('Trend')
        axes[1].set_ylabel('Trend')
        
        decomposition.seasonal.plot(ax=axes[2])
        axes[2].set_title('Seasonality')
        axes[2].set_ylabel('Seasonal')
        
        decomposition.resid.plot(ax=axes[3])
        axes[3].set_title('Residuals')
        axes[3].set_ylabel('Residual')
        
        plt.tight_layout()
        
        return fig, decomposition
    
    def check_stationarity(self):
        """Test for stationarity using the Augmented Dickey-Fuller test."""
        ts_data = self.df.set_index('timestamp')['electricity_demand'].resample('h').mean().dropna()
        
        # Perform ADF test
        adf_result = adfuller(ts_data)
        
        adf_output = {
            'Test Statistic': adf_result[0],
            'p-value': adf_result[1],
            'Critical Values': adf_result[4]
        }
        
        is_stationary = adf_result[1] < 0.05
        
        return adf_output, is_stationary
    
    def plot_autocorrelation(self, lags=48, figsize=(12, 8)):
        """
        Plot autocorrelation and partial autocorrelation functions.
        
        Parameters:
        -----------
        lags : int, optional
            Number of lags to include in the plots
        figsize : tuple, optional
            Figure size (width, height)
        """
        ts_data = self.df.set_index('timestamp')['electricity_demand'].resample('h').mean().dropna()
        
        fig, axes = plt.subplots(2, 1, figsize=figsize)
        
        # ACF plot
        plot_acf(ts_data, lags=lags, ax=axes[0])
        axes[0].set_title('Autocorrelation Function (ACF)')
        
        # PACF plot
        plot_pacf(ts_data, lags=lags, ax=axes[1])
        axes[1].set_title('Partial Autocorrelation Function (PACF)')
        
        plt.tight_layout()
        return fig
    
    def analyze_weather_impact(self):
        """Analyze the relationship between weather variables and electricity demand."""
        if 'temperature' not in self.df.columns:
            return None, "No weather data available for analysis"
            
        # Scatter plot of temperature vs demand
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        sns.scatterplot(x='temperature', y='electricity_demand', data=self.df, ax=axes[0])
        axes[0].set_title('Temperature vs Electricity Demand')
        axes[0].set_xlabel('Temperature')
        axes[0].set_ylabel('Electricity Demand')
        
        # Temperature impact by season if season is available
        if 'season' in self.df.columns:
            sns.scatterplot(x='temperature', y='electricity_demand', hue='season', data=self.df, ax=axes[1])
            axes[1].set_title('Temperature vs Electricity Demand by Season')
            axes[1].set_xlabel('Temperature')
            axes[1].set_ylabel('Electricity Demand')
        
        plt.tight_layout()
        
        # Calculate correlation coefficient
        corr = self.df['temperature'].corr(self.df['electricity_demand'])
        
        return fig, corr

def run_eda(df):
    """Run full EDA analysis and return results dictionary."""
    analyzer = ElectricityDataAnalyzer(df)
    
    results = {
        'basic_stats': analyzer.get_basic_stats(),
        'time_patterns': analyzer.analyze_time_patterns(),
        'correlation': analyzer.plot_correlation_matrix()[1],
        'stationarity': analyzer.check_stationarity()
    }
    
    return results, analyzer