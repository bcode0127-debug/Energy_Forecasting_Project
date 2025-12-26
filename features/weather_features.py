import pandas as pd

def build_weather_features(df, base_temp_h = 18, base_temp_c=22):

    # Creating heat and cool degree day features
    df = df.copy()

    # Heating feature
    df['HDD'] = (base_temp_h - df['temp'].clip(lower=0))

    # Cooling feature
    df['CDD'] = (df['temp'] - base_temp_c).clip(lower=0)

    return df

def weather_interactions(df):

    # Capturing non-linear energy sensitivity by multiplying weather features with time features
    df = df.copy()

    # Calculate the Means
    temp_mean = df['temp'].mean()
    h_sin_mean = df['hour_sin'].mean()
    
    # Create the Interaction using centered values
    df['temp_hour_interaction'] = (df['temp'] - temp_mean) * (df['hour_sin'] - h_sin_mean)

    return df 

def centered_interactions(df):
    
    df = df.copy()
    
    # Calculate means from a fixed training window 
    # Note: Adjust the date range to match training period
    temp_mean = 11.5  
    
    # Center the temperature
    df['temp_centered'] = df['temp'] - temp_mean
    
    # Create the orthogonal interaction terms
    # This specifically addresses the 'Fan Shape' by capturing time-of-day sensitivity
    df['temp_hour_interaction_sin'] = df['temp_centered'] * df['hour_sin']
    df['temp_hour_interaction_cos'] = df['temp_centered'] * df['hour_cos']
    
    return df