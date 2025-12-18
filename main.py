from data.ingestion import load_raw_data
from data.preprocessing import clean_and_interpolation
from features.time_features import build_time_features
from features.weather_features import build_weather_features

def run_pipeline():
    print("-- starting energy forcasting pipeline --")

    # Ingestion
    raw_data = load_raw_data("lcl_merged_data.csv")

    # Preprocessing
    clean_data = clean_and_interpolation(raw_data)
    
    # feature engineering
    df = build_time_features(clean_data)
    df = build_weather_features(df) # <-- Add Weather Features

    # print's
    print("--- Feature Engineering Complete ---")
    print(f"Final Column Count: {len(df.columns)}")
    print(df[['temp', 'HDD', 'CDD']].head())
    

if __name__ == "__main__":
    run_pipeline()
