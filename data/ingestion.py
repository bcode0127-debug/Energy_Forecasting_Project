import pandas as pd 
import os

def load_raw_data(file_name = "lcl_merged_data.csv"):

    # get the path to the data folder
    base_path = os.path.dirname(os.path.dirname(__file__))
    data_path = os.path.join(base_path, 'data', file_name)

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found at {data_path}")

    # load data
    df = pd.read_csv(data_path)

    # convert DateTime column
    df['DateTime'] = pd.to_datetime(df['DateTime'], utc=True)

    # set index and sort
    df = df.groupby('DateTime').mean()

    print(f"Successfully loaded {len(df)} rows.")

    return df

