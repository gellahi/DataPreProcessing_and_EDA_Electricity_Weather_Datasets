import os
import glob
import pandas as pd
import json

def load_data_from_folder(folder_path):
    all_files = []
    
    all_files.extend(glob.glob(os.path.join(folder_path, "*.csv")))
    all_files.extend(glob.glob(os.path.join(folder_path, "*.json")))
    
    dataframes = []
    for file in all_files:
        try:
            if file.endswith('.csv'):
                df = pd.read_csv(file)
            else:  # JSON file
                # Read the raw JSON file
                with open(file, 'r') as f:
                    json_data = json.load(f)
                
                # Extract data from the nested structure
                if 'response' in json_data and isinstance(json_data['response'], dict) and 'data' in json_data['response']:
                    # The actual time series data is in response.data
                    data = json_data['response']['data']
                    df = pd.DataFrame(data)
                    print(f"Successfully extracted data from {file}, columns: {df.columns.tolist()}")
                else:
                    # Fallback to direct DataFrame conversion
                    df = pd.DataFrame([json_data])
                    
            dataframes.append(df)
            print(f"Successfully loaded {file}")
        except Exception as e:
            print(f"Error loading {file}: {str(e)}")
    
    if not dataframes:
        return pd.DataFrame()
        
    result = pd.concat(dataframes, ignore_index=True)
    print(f"Final columns for {folder_path}: {result.columns.tolist()}")
    return result

def validate_and_merge_data(electricity_df, weather_df):
    print("Electricity columns:", electricity_df.columns.tolist())
    print("Weather columns:", weather_df.columns.tolist())
    
    # Find appropriate columns in electricity data
    e_time_col = next((col for col in electricity_df.columns if col in ['period', 'datetime', 'timestamp', 'date']), None)
    demand_col = next((col for col in electricity_df.columns if col in ['value', 'demand', 'load', 'consumption']), None)
    
    # Find appropriate columns in weather data
    w_time_col = next((col for col in weather_df.columns if col in ['date', 'datetime', 'timestamp']), None)
    temp_col = next((col for col in weather_df.columns if 'temp' in col.lower()), None)
    
    if not e_time_col or not w_time_col or not demand_col or not temp_col:
        print("Missing required columns:")
        print(f"  Electricity time column: {e_time_col}")
        print(f"  Electricity demand column: {demand_col}")
        print(f"  Weather time column: {w_time_col}")
        print(f"  Weather temperature column: {temp_col}")
        raise ValueError("Cannot find required columns in the data")
    
    print(f"Found key columns - Time: {e_time_col}, Demand: {demand_col}, Weather time: {w_time_col}, Temperature: {temp_col}")
    
    # Create standardized dataset
    electricity_clean = electricity_df[[e_time_col, demand_col]].copy()
    electricity_clean.rename(columns={
        e_time_col: 'timestamp', 
        demand_col: 'electricity_demand'
    }, inplace=True)
    
    weather_clean = weather_df[[w_time_col, temp_col]].copy()
    weather_clean.rename(columns={
        w_time_col: 'timestamp',
        temp_col: 'temperature'
    }, inplace=True)
    
    # Sample values for debugging
    print(f"Sample electricity timestamp: {electricity_clean['timestamp'].iloc[0]}")
    print(f"Sample weather timestamp: {weather_clean['timestamp'].iloc[0]}")
    
    # Convert to datetime with flexible parsing
    electricity_clean['timestamp'] = pd.to_datetime(electricity_clean['timestamp'], errors='coerce')
    weather_clean['timestamp'] = pd.to_datetime(weather_clean['timestamp'], errors='coerce')
    
    # Make both timestamps timezone naive by removing timezone info
    if electricity_clean['timestamp'].dt.tz is not None:
        electricity_clean['timestamp'] = electricity_clean['timestamp'].dt.tz_localize(None)
    if weather_clean['timestamp'].dt.tz is not None:
        weather_clean['timestamp'] = weather_clean['timestamp'].dt.tz_localize(None)
    
    # Check for and remove any NaT values that resulted from parsing errors
    electricity_clean = electricity_clean.dropna(subset=['timestamp'])
    weather_clean = weather_clean.dropna(subset=['timestamp'])
    
    # Ensure the timestamps are in datetime64 format (fix deprecation warning too)
    electricity_clean['timestamp'] = electricity_clean['timestamp'].dt.floor('h')
    weather_clean['timestamp'] = weather_clean['timestamp'].dt.floor('h')
    
    print(f"Electricity data shape: {electricity_clean.shape}")
    print(f"Weather data shape: {weather_clean.shape}")
    
    # Merge datasets
    print("Merging datasets...")
    merged_df = pd.merge_asof(
        electricity_clean.sort_values('timestamp'), 
        weather_clean.sort_values('timestamp'),
        on='timestamp', 
        direction='nearest',
        tolerance=pd.Timedelta('1h')
    )
    
    print(f"Merged dataset has {len(merged_df)} rows and {len(merged_df.columns)} columns")
    return merged_df