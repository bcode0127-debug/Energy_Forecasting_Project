import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance
from data.ingestion import load_raw_data
from data.preprocessing import clean_and_interpolation
from evaluation.diagnostics import plot_diagnostic_results, run_audit
from features.time_features import build_time_features, fourier_features
from features.weather_features import build_weather_features, weather_interactions, centered_interactions
from features.lag_features import build_lag_features
from models.baselines import prepare_model_data, evaluate_trustworthiness
from models.linear_models import train_elastic_net, train_ols, train_huber, train_wls
from models.ensemble_model import train_gbr_model
from evaluation.diagnostics import plot_diagnostic_results, run_audit
from evaluation.metrics import calculate_rmse, calculate_improvement 

def run_pipeline():
    # Data ingestion and feature engineering
    print("Initializing Data Pipeline...")
    df = load_raw_data("lcl_merged_data.csv")
    df = clean_and_interpolation(df)
    df = build_time_features(df)
    df = build_weather_features(df)
    df = weather_interactions(df)
    df = fourier_features(df, harmonics=3)
    df = centered_interactions(df)
    df = build_lag_features(df) 
    
    X_train, X_test, y_train, y_test, feat_names = prepare_model_data(df)
    
   # Calculate sample weights for WLS based on hour-of-day variance
    split_idx = int(len(df) * 0.8)
    df_train = df.iloc[:split_idx]  
    df_test = df.iloc[split_idx:]

    # Get the actual training indices after NaN removal
    # We need to align with X_train's actual size
    train_hours = df_train.index.hour[-len(X_train):] 

    # Calculate variance by hour for weighting
    hour_variance = df.groupby(df.index.hour)['mean_consumption'].var()
    test_hours = df_test.index.hour
    sample_weights = 1.0 / train_hours.map(hour_variance)
    sample_weights = sample_weights.values
    sample_weights = sample_weights / sample_weights.sum() * len(sample_weights)

    # Now prepare model data (this does the same split internally)
    X_train, X_test, y_train, y_test, feat_names = prepare_model_data(df)

    # Train linear models in logical order
    print("Training OLS...")
    ols_model = train_ols(X_train, y_train)
    ols_pred = ols_model.predict(X_test)

    print("Training Elastic Net...")
    en_model = train_elastic_net(X_train, y_train)
    en_pred = en_model.predict(X_test)

    print("Training Huber Regression...")
    huber_model = train_huber(X_train, y_train)
    huber_pred = huber_model.predict(X_test)

    print("Training Weighted Least Squares...")
    wls_model = train_wls(X_train, y_train, sample_weights)
    wls_pred = wls_model.predict(X_test)

    print("Training GBR Model...")
    gbr_model = train_gbr_model(X_train, y_train)
    gbr_pred = gbr_model.predict(X_test)
    
    # Diagnostic evaluation with temporal test set
    # Establish baseline metrics for comparison
    gbr_rmse, naive_rmse, gbr_improvement = evaluate_trustworthiness(y_test, gbr_pred, df_test)
    
    print("Generating Diagnostic Plots...")
    
    # Ensemble model
    gbr_diag_df = pd.DataFrame({'actual': y_test, 'predicted': gbr_pred})
    gbr_diag_df['error'] = gbr_diag_df['actual'] - gbr_diag_df['predicted']
    run_audit(y_test, gbr_pred, df_test)
    plot_diagnostic_results(gbr_diag_df, model_name="Ensemble")

    # linear model (WLS - addresses hetroscedaticity)
    wls_diag_df = pd.DataFrame({'actual': y_test, 'predicted': wls_pred})
    wls_diag_df['error'] = wls_diag_df['actual'] - wls_diag_df['predicted']
    plot_diagnostic_results(wls_diag_df, model_name="WLS")

    plt.show()

    # calculating all model metrics
    ols_rmse = calculate_rmse(y_test, ols_pred)
    ols_improvement = calculate_improvement(naive_rmse, ols_rmse)

    en_rmse = calculate_rmse(y_test, en_pred)
    en_improvement = calculate_improvement(naive_rmse, en_rmse)

    huber_rmse = calculate_rmse(y_test, huber_pred)
    huber_improvement = calculate_improvement(naive_rmse, huber_rmse)

    wls_rmse = calculate_rmse(y_test, wls_pred)
    wls_improvement = calculate_improvement(naive_rmse, wls_rmse)

    # Feature importance analysis via permutation
    print("\nCalculating Permutation Importance...")
    result = permutation_importance(gbr_model, X_test, y_test, n_repeats=5, random_state=42)
    importance = pd.Series(result.importances_mean, index=feat_names)

    summary_data = {
        "Model Type": [
            "Naive Baseline", 
            "OLS (Linear)", 
            "Elastic Net (Linear)", 
            "Huber (Robust)", 
            "WLS (Weighted)",  
            "GBR (Ensemble)"
        ],
        "RMSE": [naive_rmse, ols_rmse, en_rmse, huber_rmse, wls_rmse, gbr_rmse], 
        "Improvement (%)": [
            0.00, 
            ols_improvement,  
            en_improvement, 
            huber_improvement, 
            wls_improvement, 
            gbr_improvement
        ],
        "Trust Status": [
            "Low (Bias)", 
            "Medium (Unregularized)",
            "Medium (Regularized)", 
            "Medium (Outlier-Robust)",
            "Medium (Variance-Weighted)",
            "High (Accuracy)"
        ]
    }
    summary_df = pd.DataFrame(summary_data)
    
    print("\n" + "="*45)
    print("PROJECT 1: FINAL SYSTEM AUDIT SUMMARY")
    print("="*45)
    print(summary_df.to_string(index=False))

    # Export results and diagnostics
    if not os.path.exists('results'): 
        os.makedirs('results')
    summary_df.to_csv('results/final_model_comparison.csv', index=False)
    
    # Manual Save for Diagnostics Metadata
    diagnostics_dict = {
    "naive_rmse": float(naive_rmse),
        "ols_rmse": float(ols_rmse),
        "elastic_net_rmse": float(en_rmse),
        "huber_rmse": float(huber_rmse),
        "wls_rmse": float(wls_rmse),
        "gbr_rmse": float(gbr_rmse)
    } 

    with open('results/final_diagnostics.json', 'w') as f:
        json.dump(diagnostics_dict, f, indent=4)
    
    print(f"\nAudit complete. Artifacts saved to 'results/'")

if __name__ == "__main__":
    run_pipeline()