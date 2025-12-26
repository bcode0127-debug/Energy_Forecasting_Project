import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf
import numpy as np
import pandas as pd
import os

def run_audit(y_actual, y_pred, df_test, output_path="results/figures/"):

    os.makedirs(output_path, exist_ok=True)
    residuals = y_actual - y_pred

    # ACF spike check
    plt.figure(figsize=(10, 5)) 
    ax = plt.gca() 
    plot_acf(residuals, lags=50, ax=ax)
    plt.title("Residual Autocorrelation (Audit)")
    plt.savefig("results/figures/acf_audit.png")
    plt.close()

    # Hourly error check
    df_diag = pd.DataFrame({'actual': np.array(y_actual).flatten(), 
                            'pred': np.array(y_pred).flatten()})
    
    # Recover the 'hour' from the df_test index or columns
    if 'hour' in df_test.columns:
        df_diag['hour'] = df_test['hour'].values
    else:
        # If hour is not a column, assume it's in the index
        df_diag['hour'] = df_test.index.hour

    df_diag['error_sq'] = (df_diag['actual'] - df_diag['pred'])**2

    night_rmse = np.sqrt(df_diag[df_diag['hour'].between(3, 6)] ['error'].mean())
    evng_rmse = np.sqrt(df_diag[df_diag['hour'].between(18, 21)] ['error'].mean())

    print("\n" + "="*40)
    print(" ** DIAGNOSTIC REPORT ** ")
    print("="*40)
    print(f"3AM - 6AM RMSE (Night):   {night_rmse:.6f}")
    print(f"6PM - 9PM RMSE (Evening): {evng_rmse:.6f}")
    print("-" * 40)
    
    if evng_rmse > night_rmse * 1.5:
        print("DIAGNOSIS: High Heteroscedasticity.")
        print("The model fails significantly during peak hours.")
    
    print(f"Audit plot saved: results/figures/acf_audit.png")
    print("="*40 + "\n")


def plot_diagnostic_results(diag_df):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Residual Plot 
    ax1.scatter(diag_df['predicted'], diag_df['error'], alpha=0.1)
    ax1.axhline(0, color='red', linestyle='--')
    ax1.set_title("Residuals vs Predicted")
    ax1.set_xlabel("Predicted Energy")
    ax1.set_ylabel("Error")
    
    # ACF Plot 
    plot_acf(diag_df['error'], lags=60, ax=ax2)
    ax2.set_title("Autocorrelation of Errors")
    
    plt.tight_layout()
    plt.show()