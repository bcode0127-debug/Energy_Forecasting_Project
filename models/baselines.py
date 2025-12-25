import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.metrics import mean_squared_error
import json
import os

def prepare_model_data(df, target_col='target_day_ahead', test_size = 0.2):

    # Perform temporal splitting and responsible scaling

    # Temporal split(NO Shuffling)
    split_idx = int(len(df) * (1 - test_size))
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]

    # Feature/Target separation
    # dropping 'mean_consuption' because it's current values (leakage)
    X_train = train_df.drop(columns=[target_col, 'mean_consumption'])
    y_train = train_df[target_col]
    X_test = test_df.drop(columns=[target_col, 'mean_consumption'])
    y_test = test_df[target_col]
    
    feature_names = X_train.columns

    # Scaling (fit on train, transform on test)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, X_train.columns


def evaluate_trustworthiness(y_test, y_pred, df_test):
    # Calculate Model RMSE
    model_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    # Calculate Naive Persistence RMSE (Predicting T+48 using T)
    naive_preds = df_test['mean_consumption'] 
    naive_rmse = np.sqrt(mean_squared_error(y_test, naive_preds))
    
    # Calculate Improvement 
    improvement = ((naive_rmse - model_rmse) / naive_rmse) * 100
    
    return model_rmse, naive_rmse, improvement

def save_results_json(metrics, feature_importance, folder='results'):

    # Ensure the folder exists
    if not os.path.exists(folder):
        os.makedirs(folder)
        
    # Prepare the data structure
    results_data = {
        "metrics": metrics,
        "top_features": feature_importance.to_dict()
    }
    
    file_path = os.path.join(folder, 'baseline_results.json')
    
    with open(file_path, 'w') as f:
        json.dump(results_data, f, indent=4)
    
    print(f"Results successfully saved to {file_path}")