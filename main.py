from data.ingestion import load_raw_data
from data.preprocessing import clean_and_interpolation

def run_pipeline():
    print("-- starting energy forcasting pipeline --")

    # Ingestion
    raw_data = load_raw_data("lcl_merged_data.csv")

    # Preprocessing
    clean_data = clean_and_interpolation(raw_data)
    print(clean_data.head())
    print(f"\nMissing values after cleaning:  \n{clean_data.isnull().sum()}")

if __name__ == "__main__":
    run_pipeline()
