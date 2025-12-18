import pandas as pd

def clean_and_interpolation(df):

    # enfore 30-min frequency
    df = df.asfreq('30min')

    # Linear interpolation
    df['mean_consumption'] = df['mean_consumption'].interpolate(method='linear')
    df['temp'] = df['temp'].interpolate(method='linear')
    df['humidity'] = df['humidity'].interpolate(method='linear')

    # final cleanup
    df = df.ffill().bfill()

    return df

