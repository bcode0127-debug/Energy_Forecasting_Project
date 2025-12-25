import pandas as pd

def build_weather_features(df, base_temp_h = 18, base_temp_c=22):

    # Creating heat and cool degree day features
    df = df.copy()

    # Headting feature
    df['HDD'] = (base_temp_h - df['temp'].clip(lower=0))

    # Cooling feature
    df['CDD'] = (df['temp'] - base_temp_c).clip(lower=0)

    return df

def weather_interactions(df):

    # Capturing non-linear energy sensitivity by multiplying weather features with time features
    df = df.copy()

    # Interaction: Temp * hour
    df['temp_hour_sin'] = df['temp'] * df['hour_sin']
    df['temp_hour_cos'] = df['temp'] * df['hour_cos']

    # Interaction: Humidity * Season
    df['humidity_seasonal'] = df['humidity'] * df['month_sin']

    return df 