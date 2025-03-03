import pandas as pd
import os
from data_loader import load_data_from_folder, validate_and_merge_data
from data_preprocessor import clean_data
from outlier_handling import handle_outliers

# 1. Load data
print("Loading electricity data...")
electricity_data = load_data_from_folder("electricity_raw_data")
print("Loading weather data...")
weather_data = load_data_from_folder("weather_raw_data")

# 2. Merge datasets
print("Merging datasets...")
merged_df = validate_and_merge_data(electricity_data, weather_data)

# 3. Clean data
print("Cleaning data...")
merged_df = clean_data(merged_df)

# 4. Handle outliers
print("Handling outliers...")
merged_df = handle_outliers(merged_df)

# 5. Save processed data
print("Saving processed data...")
merged_df.to_csv('merged_data.csv', index=False)

print("Data processing complete. You can now run the Streamlit app.")