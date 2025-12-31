import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf
import numpy as np
import pandas as pd
import os
from evaluation.metrics import get_segmented_rmse

def run_audit(y_actual, y_pred, df_test, output_path="results/figures/"):
    
    # Performs a multi-dimensional diagnostic audit of model residuals.
    os.makedirs(output_path, exist_ok=True)
    residuals = y_actual - y_pred

    # Statistical Check: Verify if residuals behave as 'White Noise'
    # Spikes outside the confidence interval indicate uncaptured temporal patterns.
    plt.figure(figsize=(10, 5)) 
    ax = plt.gca() 
    plot_acf(residuals, lags=50, ax=ax)
    plt.title("Residual Autocorrelation (Audit)")
    plt.savefig("results/figures/acf_audit.png")
    #plt.close()

    # Data Alignment: Map residuals back to their respective temporal hours
    df_diag = pd.DataFrame({'actual': np.array(y_actual).flatten(), 
                            'pred': np.array(y_pred).flatten()})
    
    # Recover temporal metadata from the test index for segmented analysis
    if 'hour' in df_test.columns:
        df_diag['hour'] = df_test['hour'].values
    else:
        df_diag['hour'] = df_test.index.hour

    # Calculate squared errors for RMSE segmentation
    df_diag['error_sq'] = (df_diag['actual'] - df_diag['pred'])**2

    # Segmented Reliability: Compare stability during 'Off-Peak' vs 'Peak' demand
    # High variance during evening hours suggests heteroscedasticity.
    night_rmse = get_segmented_rmse(df_diag, 3, 6)
    evng_rmse = get_segmented_rmse(df_diag, 18, 21)

    print("\n" + "="*40)
    print(" ** DIAGNOSTIC REPORT ** ")
    print("="*40)
    print(f"3AM - 6AM RMSE (Night):   {night_rmse:.6f}")
    print(f"6PM - 9PM RMSE (Evening): {evng_rmse:.6f}")
    print("-" * 40)
    
    # Heuristic check for non-constant variance (Heteroscedasticity)
    if evng_rmse > night_rmse * 1.5:
        print("DIAGNOSIS: High Heteroscedasticity.")
        print("The model fails significantly during peak hours.")
    
    print(f"Audit plot saved: results/figures/acf_audit.png")
    print("="*40 + "\n")


def plot_diagnostic_results(diag_df, model_name = "Model"):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Residual Plot 
    ax1.scatter(diag_df['predicted'], diag_df['error'], alpha=0.1)
    ax1.axhline(0, color='red', linestyle='--')
    ax1.set_title(f"{model_name} Residuals: Check for Fan Shape")
    ax1.set_xlabel("Predicted Energy")
    ax1.set_ylabel("Error")
    
    # ACF Plot 
    plot_acf(diag_df['error'], lags=60, ax=ax2)
    ax2.set_title(f"{model_name} Autocorrelation: Check for Spike at 48")
    plt.tight_layout()

    file_path = f"results/figures/diagnostic_residuals_{model_name.lower()}.png"
    plt.savefig(file_path)
    print(f"updated diagnostic plot: {file_path}")
    plt.show()