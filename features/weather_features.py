import pandas as pd

def build_weather_features(df, base_temp_h = 18, base_temp_c=22):

    # creating heat and cool degree day features
    df = df.copy()

    # headting feature
    df['HDD'] = (base_temp_h - df['temp'].clip(lower=0))

    # cooling feature
    df['CDD'] = (df['temp'] - base_temp_c).clip(lower=0)

    return df