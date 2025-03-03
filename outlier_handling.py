import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

class OutlierDetector:
    """
    Class for detecting and handling outliers in electricity demand data.
    """
    
    def __init__(self, df):
        """Initialize with a dataframe containing electricity demand data."""
        self.df = df.copy()
        self.outliers_info = {}
    
    def detect_iqr_outliers(self, column, factor=1.5):
        """
        Detect outliers using the Interquartile Range (IQR) method.
        
        Parameters:
        -----------
        column : str
            Column name to check for outliers
        factor : float, optional
            IQR multiplier to define outlier boundaries (default 1.5)
            
        Returns:
        --------
        outlier_indices : array-like
            Indices of detected outliers
        """
        Q1 = self.df[column].quantile(0.25)
        Q3 = self.df[column].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - factor * IQR
        upper_bound = Q3 + factor * IQR
        
        outlier_indices = self.df[(self.df[column] < lower_bound) | 
                                   (self.df[column] > upper_bound)].index
        
        self.outliers_info['iqr'] = {
            'method': 'IQR',
            'column': column,
            'factor': factor,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'count': len(outlier_indices),
            'percentage': len(outlier_indices) / len(self.df) * 100,
            'indices': outlier_indices
        }
        
        return outlier_indices
    
    def detect_zscore_outliers(self, column, threshold=3.0):
        """
        Detect outliers using Z-score method.
        
        Parameters:
        -----------
        column : str
            Column name to check for outliers
        threshold : float, optional
            Z-score threshold to identify outliers (default 3.0)
            
        Returns:
        --------
        outlier_indices : array-like
            Indices of detected outliers
        """
        z_scores = (self.df[column] - self.df[column].mean()) / self.df[column].std()
        outlier_indices = self.df[abs(z_scores) > threshold].index
        
        self.outliers_info['zscore'] = {
            'method': 'Z-Score',
            'column': column,
            'threshold': threshold,
            'count': len(outlier_indices),
            'percentage': len(outlier_indices) / len(self.df) * 100,
            'indices': outlier_indices
        }
        
        return outlier_indices
    
    def detect_isolation_forest(self, columns, contamination=0.05):
        """
        Detect outliers using Isolation Forest algorithm.
        
        Parameters:
        -----------
        columns : list
            List of column names to use for outlier detection
        contamination : float, optional
            Expected proportion of outliers in the dataset (default 0.05)
            
        Returns:
        --------
        outlier_indices : array-like
            Indices of detected outliers
        """
        X = self.df[columns].values
        
        # Train Isolation Forest
        iso = IsolationForest(contamination=contamination, random_state=42)
        iso.fit(X)
        
        # Predict outliers (-1 for outliers, 1 for inliers)
        outlier_pred = iso.predict(X)
        outlier_indices = self.df[outlier_pred == -1].index
        
        self.outliers_info['isolation_forest'] = {
            'method': 'Isolation Forest',
            'columns': columns,
            'contamination': contamination,
            'count': len(outlier_indices),
            'percentage': len(outlier_indices) / len(self.df) * 100,
            'indices': outlier_indices
        }
        
        return outlier_indices
    
    def detect_lof_outliers(self, columns, contamination=0.05):
        """
        Detect outliers using Local Outlier Factor algorithm.
        
        Parameters:
        -----------
        columns : list
            List of column names to use for outlier detection
        contamination : float, optional
            Expected proportion of outliers in the dataset (default 0.05)
            
        Returns:
        --------
        outlier_indices : array-like
            Indices of detected outliers
        """
        X = self.df[columns].values
        
        # Apply LOF
        lof = LocalOutlierFactor(n_neighbors=20, contamination=contamination)
        outlier_pred = lof.fit_predict(X)
        outlier_indices = self.df[outlier_pred == -1].index
        
        self.outliers_info['lof'] = {
            'method': 'Local Outlier Factor',
            'columns': columns,
            'contamination': contamination,
            'count': len(outlier_indices),
            'percentage': len(outlier_indices) / len(self.df) * 100,
            'indices': outlier_indices
        }
        
        return outlier_indices
    
    def plot_outliers(self, column, method='all'):
        """
        Plot detected outliers.
        
        Parameters:
        -----------
        column : str
            Column name to visualize outliers
        method : str, optional
            Method of outlier detection to visualize ('iqr', 'zscore', 'isolation_forest', 'lof', or 'all')
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot original data
        plt.scatter(range(len(self.df)), self.df[column], alpha=0.5, label='Data points')
        
        # Plot outliers based on method
        if method == 'all':
            methods = [m for m in self.outliers_info.keys()]
        else:
            methods = [method]
        
        colors = ['red', 'orange', 'green', 'purple']
        for i, m in enumerate(methods):
            if m in self.outliers_info:
                indices = self.outliers_info[m]['indices']
                plt.scatter(indices, self.df.loc[indices, column], 
                           color=colors[i % len(colors)], 
                           s=50, label=f'{self.outliers_info[m]["method"]} outliers')
        
        plt.title(f'Outlier Detection for {column}')
        plt.xlabel('Data point index')
        plt.ylabel(column)
        plt.legend()
        plt.tight_layout()
        
        return fig
    
    def handle_outliers(self, method, column=None, strategy='cap'):
        """
        Handle outliers in the data.
        
        Parameters:
        -----------
        method : str
            Method used to detect outliers ('iqr', 'zscore', 'isolation_forest', 'lof')
        column : str, optional
            Column to handle outliers in (required for 'cap' and 'mean' strategies)
        strategy : str, optional
            Strategy to handle outliers:
            - 'remove': Remove outlier rows
            - 'cap': Cap outliers at bounds (for IQR/zscore)
            - 'mean': Replace with mean
            - 'median': Replace with median
        
        Returns:
        --------
        cleaned_df : pandas DataFrame
            Dataframe with handled outliers
        """
        # Create a copy of the dataframe to avoid modifying the original
        cleaned_df = self.df.copy()
        
        # Get outlier indices from stored info
        if method not in self.outliers_info:
            raise ValueError(f"Method '{method}' not found. Run detection first.")
            
        outlier_indices = self.outliers_info[method]['indices']
        
        # Handle outliers based on strategy
        if strategy == 'remove':
            cleaned_df = cleaned_df.drop(outlier_indices)
            
        elif strategy == 'cap' and column:
            if method == 'iqr':
                lower_bound = self.outliers_info['iqr']['lower_bound']
                upper_bound = self.outliers_info['iqr']['upper_bound']
                cleaned_df.loc[cleaned_df[column] < lower_bound, column] = lower_bound
                cleaned_df.loc[cleaned_df[column] > upper_bound, column] = upper_bound
                
            elif method == 'zscore':
                threshold = self.outliers_info['zscore']['threshold']
                mean = cleaned_df[column].mean()
                std = cleaned_df[column].std()
                lower_bound = mean - threshold * std
                upper_bound = mean + threshold * std
                cleaned_df.loc[cleaned_df[column] < lower_bound, column] = lower_bound
                cleaned_df.loc[cleaned_df[column] > upper_bound, column] = upper_bound
                
        elif strategy == 'mean' and column:
            mean_value = cleaned_df[column].mean()
            cleaned_df.loc[outlier_indices, column] = mean_value
            
        elif strategy == 'median' and column:
            median_value = cleaned_df[column].median()
            cleaned_df.loc[outlier_indices, column] = median_value
            
        else:
            raise ValueError(f"Invalid strategy '{strategy}' or missing column parameter.")
        
        return cleaned_df
    
    def compare_before_after(self, original_df, cleaned_df, column):
        """
        Compare distributions before and after outlier handling.
        
        Parameters:
        -----------
        original_df : pandas DataFrame
            Original dataframe with outliers
        cleaned_df : pandas DataFrame
            Dataframe after outlier handling
        column : str
            Column to compare
        
        Returns:
        --------
        fig : matplotlib Figure
            Figure with comparison plots
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Boxplot comparison
        sns.boxplot(x=original_df[column], ax=axes[0, 0])
        axes[0, 0].set_title('Before Outlier Handling')
        
        sns.boxplot(x=cleaned_df[column], ax=axes[0, 1])
        axes[0, 1].set_title('After Outlier Handling')
        
        # Histogram comparison
        sns.histplot(original_df[column], kde=True, ax=axes[1, 0])
        axes[1, 0].set_title('Before Outlier Handling')
        
        sns.histplot(cleaned_df[column], kde=True, ax=axes[1, 1])
        axes[1, 1].set_title('After Outlier Handling')
        
        plt.tight_layout()
        return fig
    
    def get_summary(self):
        """Get a summary of detected outliers across all methods."""
        summary = pd.DataFrame({
            'Method': [info['method'] for info in self.outliers_info.values()],
            'Count': [info['count'] for info in self.outliers_info.values()],
            'Percentage (%)': [info['percentage'] for info in self.outliers_info.values()]
        })
        return summary

def detect_and_handle_outliers(df, column='electricity_demand'):
    """Main function to detect and handle outliers in a dataset."""
    detector = OutlierDetector(df)
    
    # Detect outliers using multiple methods
    iqr_outliers = detector.detect_iqr_outliers(column)
    zscore_outliers = detector.detect_zscore_outliers(column)
    
    # Use multiple features if available
    if 'temperature' in df.columns:
        features = ['electricity_demand', 'temperature']
        iso_outliers = detector.detect_isolation_forest(features)
        lof_outliers = detector.detect_lof_outliers(features)
    
    # Get summary of detected outliers
    summary = detector.get_summary()
    
    # Handle outliers (example: using IQR with capping)
    cleaned_df = detector.handle_outliers('iqr', column=column, strategy='cap')
    
    return detector, cleaned_df, summary

def handle_outliers(df, column='electricity_demand'):
    """
    Legacy function for backward compatibility with main.py
    Just calls detect_and_handle_outliers and returns the cleaned dataframe
    """
    _, cleaned_df, _ = detect_and_handle_outliers(df, column)
    return cleaned_df