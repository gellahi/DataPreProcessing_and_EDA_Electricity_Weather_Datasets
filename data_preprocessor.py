import numpy as np
import pandas as pd  # Add this import

def clean_data(merged_df):
    # Handle missing values (fixing deprecation warnings)
    merged_df = merged_df.ffill().bfill()  # Replace deprecated fillna(method=...)
    
    # Remove duplicates
    merged_df.drop_duplicates(subset='timestamp', keep='first', inplace=True)
    
    # Validate data types
    merged_df['electricity_demand'] = pd.to_numeric(merged_df['electricity_demand'], errors='coerce')
    merged_df['temperature'] = pd.to_numeric(merged_df['temperature'], errors='coerce')
    
    # Create new time-based features
    merged_df['hour'] = merged_df['timestamp'].dt.hour
    merged_df['day_of_week'] = merged_df['timestamp'].dt.dayofweek
    merged_df['is_weekend'] = merged_df['day_of_week'].isin([5,6]).astype(int)
    merged_df['month'] = merged_df['timestamp'].dt.month
    merged_df['season'] = pd.cut(merged_df['month'],
                            bins=[0,2,5,8,11,12],
                            labels=['Winter', 'Spring', 'Summer', 'Autumn', 'Winter'],
                            right=False,
                            include_lowest=True,
                            ordered=False)
    
    return merged_df