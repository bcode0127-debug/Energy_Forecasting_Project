import numpy as np

def calculate_rmse(y_true, y_pred):
    
    # Computes the Root Mean Squared Error between actual and predicted vectors.
    return np.sqrt(np.mean((np.array(y_true) - np.array(y_pred))**2))

def calculate_improvement(baseline_rmse, model_rmse):
    
    # Calculates the percentage improvement of a model over a baseline.
    if baseline_rmse == 0:
        return 0.0
    return ((baseline_rmse - model_rmse) / baseline_rmse) * 100

def get_segmented_rmse(df, hour_start, hour_end):
    
    # Calculates the percentage improvement of a model over a baseline.
    subset = df[df['hour'].between(hour_start, hour_end)]
    if len(subset) == 0:
        return 0.0
    return np.sqrt(subset['error_sq'].mean())