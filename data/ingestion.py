import pandas as pd 
import os

def load_raw_data(file_name = "lcl_merged_data.csv"):

    # Get the path to the data folder environment-agnostic execution
    base_path = os.path.dirname(os.path.dirname(__file__))
    data_path = os.path.join(base_path, 'data', file_name)

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found at {data_path}")

    # Initialize dataframe from LCL
    df = pd.read_csv(data_path)

    # Cast temporal column to UTC-aware datetime objects
    df['DateTime'] = pd.to_datetime(df['DateTime'], utc=True)

    # Deduplicate and aggregate by timestamp to ensure a unique temporal index
    df = df.groupby('DateTime').mean()

    print(f"Successfully loaded {len(df)} rows.")

    return df

