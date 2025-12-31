import pandas as pd

def build_lag_features(df, target_col='mean_consumption'):

    # time delay embedding Lags to eliminate residual autocorrelation.
    df = df.copy()

    #df['lag_30m'] = df[target_col].shift(1) # 1-hour lag (Short-term momentum)
    df['lag_48h'] = df[target_col].shift(48) # 24-hour lag (Crucial for fixing the ACF 'wave')
    df['lag_96'] = df[target_col].shift(96) # 48-hour lag (Captures multi-day trends)

    # Rolling 24-hour average (Smooths out noise)
    df['rolling_mean_4h'] = df[target_col].shift(48).rolling(window = 48).mean() 

    # Target (Day-ahead)
    df['target_day_ahead'] = df[target_col].shift(-48)
    df = df.dropna()

    print(f"DEBUG: Internal columns are: {df.columns.tolist()}")
    
    return df

