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
from models.linear_models import train_elastic_net
from models.ensemble_model import train_gbr_model
from evaluation.diagnostics import plot_diagnostic_results, run_audit
from evaluation.metrics import calculate_rmse, calculate_improvement 

def run_pipeline():
    # Data ingestion and feature engineering
    print("Initializing Data Pipeline.../n")
    df = load_raw_data("lcl_merged_data.csv")
    df = clean_and_interpolation(df)
    df = build_time_features(df)
    df = build_weather_features(df)
    df = weather_interactions(df)
    df = fourier_features(df, harmonics=3)
    df = centered_interactions(df)
    df = build_lag_features(df) 
    
    X_train, X_test, y_train, y_test, feat_names = prepare_model_data(df)
    
    # Train linear and ensemble models
    print("Training Elastic Net...")
    en_model = train_elastic_net(X_train, y_train)
    en_pred = en_model.predict(X_test)
    
    print("Training GBR Model...")
    gbr_model = train_gbr_model(X_train, y_train) 
    gbr_pred = gbr_model.predict(X_test) 

    # Diagnostic evaluation with temporal test set
    split_idx = int(len(df) * 0.8)
    df_test = df.iloc[split_idx:]
    
    # Establish baseline metrics for comparison
    gbr_rmse, naive_rmse, gbr_improvement = evaluate_trustworthiness(y_test, gbr_pred, df_test)
    
    print("Updating Ensemble Diagnostic Plots...")
    gbr_diag_df = pd.DataFrame({'actual': y_test, 'predicted': gbr_pred})
    gbr_diag_df['error'] = gbr_diag_df['actual'] - gbr_diag_df['predicted']
    run_audit(y_test, gbr_pred, df_test)
    plot_diagnostic_results(gbr_diag_df, model_name="Ensemble")

    print("Updating Linear Diagnostic Plots...")
    en_diag_df = pd.DataFrame({'actual': y_test, 'predicted': en_pred})
    en_diag_df['error'] = en_diag_df['actual'] - en_diag_df['predicted']
    plot_diagnostic_results(en_diag_df, model_name="linear")
    plt.show()

    # Calculate linear model performance against baseline
    en_rmse = calculate_rmse(y_test, en_pred)
    en_improvement = calculate_improvement(naive_rmse, en_rmse)
    
    # Feature importance analysis via permutation
    print("\nCalculating Permutation Importance...")
    result = permutation_importance(gbr_model, X_test, y_test, n_repeats=5, random_state=42)
    importance = pd.Series(result.importances_mean, index=feat_names)

    summary_data = {
        "Model Type": ["Naive Baseline", "Elastic Net (Linear)", "GBR (Ensemble)"],
        "RMSE": [naive_rmse, en_rmse, gbr_rmse], 
        "Improvement (%)": [0.00, en_improvement, gbr_improvement],
        "Trust Status": ["Low (Bias)", "Medium (Interpretability)", "High (Accuracy)"]
    }
    summary_df = pd.DataFrame(summary_data)
    
    print("\n" + "="*45)
    print("PROJECT 1: FINAL SYSTEM AUDIT SUMMARY")
    print("="*45)
    print(summary_df.to_string(index=False))

    # Export results and diagnostics
    if not os.path.exists('results'): os.makedirs('results')
    summary_df.to_csv('results/final_model_comparison.csv', index=False)
    
    # Manual Save for Diagnostics Metadata
    diagnostics_dict = {
    "final_metrics": {
        "naive_rmse": float(naive_rmse),
        "elastic_net_rmse": float(en_rmse),
        "gbr_rmse": float(gbr_rmse)
    },
    "present_change": {
        "en_present_change": float(en_improvement),
        "gbr_presenet_change": float(gbr_improvement)
    } 
    }

    with open('results/final_diagnostics.json', 'w') as f:
        json.dump(diagnostics_dict, f, indent=4)
    
    print(f"\nAudit complete. Artifacts saved to 'results/'")

if __name__ == "__main__":
    run_pipeline()