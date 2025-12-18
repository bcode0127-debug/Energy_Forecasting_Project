import pandas as pd

def build_lag_features(df, target_col='mean_consumption'):

    # time delay embedding Lags
    df = df.copy()

    df['lag_30m'] = df[target_col].shift(1)
    df['lag_24h'] = df[target_col].shift(48)

    # rolling mean
    df['rolling_mean_4h'] = df[target_col].shift(1).rolling(window = 8).mean()

    print(f"DEBUG: Internal columns are: {df.columns.tolist()}")
    
    df = df.dropna()

    return df

