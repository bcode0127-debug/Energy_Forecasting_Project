import matplotlib.pyplot as plt
from data.ingestion import load_raw_data
from data.preprocessing import clean_and_interpolation
from features.time_features import build_time_features
from features.weather_features import build_weather_features
from features.lag_features import build_lag_features
from models.baselines import prepare_model_data
from models.linear_models import train_elastic_net
from models.baselines import prepare_model_data, evaluate_trustworthiness
from models.baselines import save_results_json
from features.weather_features import weather_interactions
from evaluation.diagnostics import plot_diagnostic_results
import pandas as pd

def run_pipeline():
    # 1. Data Preparation
    df = load_raw_data("lcl_merged_data.csv")
    df = clean_and_interpolation(df)
    
    # 2. Feature Engineering & Target Shifting
    df = build_time_features(df)
    df = build_weather_features(df)

    print("Adding Interaction Terms (Temp x Hour)...")
    df = weather_interactions(df)

    df = build_lag_features(df) # Handles the -48 target shift and dropna()
    
    # 3. Responsible Splitting & Scaling
    X_train, X_test, y_train, y_test, feat_names = prepare_model_data(df)
    
    # 4. Model Training (The "Discovery" Phase)
    print("Training Elastic Net with Cross-Validation...")
    model = train_elastic_net(X_train, y_train)
    
    print(f"Best Alpha (Penalty): {model.alpha_:.4f}")
    print(f"Best L1 Ratio (Sparsity): {model.l1_ratio_:.2f}")
    
    y_pred = model.predict(X_test) 

    # 5. Evaluation & Naive Comparison
    # Note: We need the unscaled test dataframe to get the naive baseline values
    split_idx = int(len(df) * 0.8)
    df_test = df.iloc[split_idx:]
    
    model_rmse, naive_rmse, improvement = evaluate_trustworthiness(y_test, y_pred, df_test)
    
    print("\n" + "="*30)
    print("FINAL PERFORMANCE AUDIT")
    print("="*30)
    print(f"Elastic Net RMSE:  {model_rmse:.6f}")
    print(f"Naive Baseline RMSE: {naive_rmse:.6f}")
    print(f"Total Improvement:   {improvement:.2f}%")
    print("="*30)

    # 6. Feature Reveal
    importance = pd.Series(model.coef_, index=feat_names)
    print("\nTop 5 Influential Features:")
    print(importance.abs().sort_values(ascending=False).head(5))

    # 7. SAVE NUMERICAL RESULTS
    metrics_dict = {
        "model_rmse": round(model_rmse, 6),
        "naive_rmse": round(naive_rmse, 6),
        "improvement_percent": round(improvement, 2)
    }
    
    # Save to the folder you created
    save_results_json(metrics_dict, importance.head(10))

    # Evaluation & Plotting
    y_pred = model.predict(X_test)

   # --- THE PROFESSOR'S FIX ---
    diagnostic_df = pd.DataFrame({
        'actual': y_test, 
        'predicted': y_pred
    })
    # This line solves the KeyError
    diagnostic_df['error'] = diagnostic_df['actual'] - diagnostic_df['predicted']
    # ---------------------------

    # Now run your audit/evaluation functions using this new df
    model_rmse, naive_rmse, improvement = evaluate_trustworthiness(y_test, y_pred, df_test)
    
    # NEW: Run the Diagnostic Plots (Workflow Section 9)
    # This fulfills the "ACF" and "Fan Shape" requirements
    plot_diagnostic_results(diagnostic_df)
    
    plt.figure(figsize=(12, 6))
    plt.plot(y_test.values[:100], label="Actual (T+48)", color="black", alpha=0.7)
    plt.plot(y_pred[:100], label="Elastic Net Forecast", color="blue", linestyle="--")
    plt.title("Day-Ahead Energy Forecast (First 100 30-min slots of Test Set)")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    run_pipeline()

