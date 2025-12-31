import pandas as pd

def clean_and_interpolation(df):

    # Enforce strict 30-minute periodicity to ensure temporal continuity
    df = df.asfreq('30min')

    # Synthesize missing intervals via linear interpolation 
    # This aligns hourly weather data with 30-minute energy pings
    df['mean_consumption'] = df['mean_consumption'].interpolate(method='linear')
    df['temp'] = df['temp'].interpolate(method='linear')
    df['humidity'] = df['humidity'].interpolate(method='linear')

    # Impute boundary values to ensure a zero-null feature matrix
    df = df.ffill().bfill()

    return df

