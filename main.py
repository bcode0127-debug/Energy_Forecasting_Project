from data.ingestion import load_raw_data
from data.preprocessing import clean_and_interpolation
from features.lag_features import build_lag_features
from features.time_features import build_time_features
from features.weather_features import build_weather_features

def run_pipeline():
    print("--- Starting Energy Forecasting Pipeline ---")
    
    # 1. Ingestion & Preprocessing
    df = load_raw_data("lcl_merged_data.csv")
    df = clean_and_interpolation(df)
    
    # 2. Feature Engineering
    df = build_time_features(df)
    df = build_weather_features(df)
    
    # IMPORTANT: Re-assign the result of the function back to 'df'
    df = build_lag_features(df) 
    
    print("--- Feature Engineering Complete ---")
    print(f"Final Dataset Shape: {df.shape}")
    
    # Check if columns exist before printing to avoid the KeyError
    cols_to_show = ['mean_consumption', 'lag_30m', 'lag_24h']
    if all(c in df.columns for c in cols_to_show):
        print(df[cols_to_show].head())
    else:
        print("Error: Lag columns were not found in the final dataframe!")

if __name__ == "__main__":
    run_pipeline()

