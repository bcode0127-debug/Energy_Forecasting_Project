import numpy as np
import pandas as pd

def build_time_features(df):

    # create cyclical and binary time features from the index
    df = df.copy()

    # cyclical encoding (sine/cosine)
    df['hour_sin'] = np.sin(2 * np.pi * df.index.hour / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df.index.hour / 24)

    # day of week
    df['day_sin'] = np.sin(2 * np.pi * df.index.dayofweek / 7)
    df['day_cos'] = np.cos(2 * np.pi * df.index.dayofweek / 7)

    # month
    df['month_sin'] = np.sin(2 * np.pi * (df.index.month - 1)/ 12)
    df['month_cos'] = np.cos(2 * np.pi * (df.index.month - 1)/ 12)

    # binary flags
    df['is_weekend'] = df.index.dayofweek.map(lambda x: 1 if x >= 5 else 0)

    return df