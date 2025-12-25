import pandas as pd

def build_lag_features(df, target_col='mean_consumption'):

    # time delay embedding Lags
    df = df.copy()

    df['lag_30m'] = df[target_col].shift(1)
    df['lag_24h'] = df[target_col].shift(48)
    df['rolling_mean_4h'] = df[target_col].shift(1).rolling(window = 8).mean()

    # Target (Day-ahead)
    df['target_day_ahead'] = df[target_col].shift(-48)
    df = df.dropna()

    print(f"DEBUG: Internal columns are: {df.columns.tolist()}")
    
    return df

