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

    # 1. Calculate the Means
    temp_mean = df['temp'].mean()
    h_sin_mean = df['hour_sin'].mean()
    
    # 2. Create the Interaction using centered values
    # Formula: (A - mean_A) * (B - mean_B)
    df['temp_hour_interaction'] = (df['temp'] - temp_mean) * (df['hour_sin'] - h_sin_mean)

    return df 

def centered_interactions(df):
    
    df = df.copy()
    
    # 1. Calculate means from a fixed training window (Reference: Workflow 7.1)
    # Note: Adjust the date range to match your training period
    temp_mean = 11.5  # Typical London mean, or use df['temp'].mean() from train set
    
    # 2. Center the temperature
    df['temp_centered'] = df['temp'] - temp_mean
    
    # 3. Create the orthogonal interaction terms
    # This specifically addresses the 'Fan Shape' by capturing time-of-day sensitivity
    df['temp_hour_interaction_sin'] = df['temp_centered'] * df['hour_sin']
    df['temp_hour_interaction_cos'] = df['temp_centered'] * df['hour_cos']
    
    return df