# Energy Consumption Forecasting: Breaking the Linear Barrier

## Overview

End-to-end machine learning pipeline for short-term energy consumption forecasting using 167 million smart meter records from the London Grid. Achieves 14.52% improvement over naive baseline through cloud-native data engineering on Google Cloud Platform and systematic feature engineering.

## Results

| Model | Configuration | RMSE | Improvement |
|-------|--------------|------|-------------|
| Naive Baseline | N/A | 0.0358 | - |
| Elastic Net (initial) | lag_30m, lag_24h | 0.0332 | +7.52% |
| Elastic Net (current) | lag_48h, lag_96h | 0.0365 | -1.72% |
| Gradient Boosting (initial) | lag_30m, lag_24h | 0.0306 | +14.52% |
| Gradient Boosting (current) | lag_48h, lag_96h | 0.0325 | +9.56% |

## Technical Stack

- Google Cloud Platform (BigQuery, Compute Engine)
- Python 3.9+ (scikit-learn, pandas, numpy)
- Dataset: London Smart Meter (167M records, 30-minute intervals)
- Weather API: Visual Crossing (temperature, humidity)

## Architecture

### Data Engineering Pipeline

**Infrastructure Orchestration:** Google Compute Engine VM instance manages data movement and executes processing scripts in the cloud.

**Distributed SQL Transformation:** High-efficiency joins and temporal aggregations performed directly in Google BigQuery, integrating 167M raw energy records with external weather data.

**Temporal Alignment:** Linear interpolation resolves frequency mismatch between hourly weather data and 30-minute energy readings, ensuring continuous feature matrix.

**Export:** Final dataset (lcl_merged_data.csv) transitions from cloud engineering to local modeling.

### Feature Engineering

**Fourier Series Harmonics:** Sine and cosine waves capture 24-hour cyclical patterns of energy demand.

**Temporal Lags:** Short-term (30m) and long-term (24h) dependencies ground model in historical data.

**Weather Interactions:** Temperature-hour coupling (temp × hour, temp × sin(hour)) accounts for time-varying weather effects.

**Cyclical Encoding:** Hour, day, month encoded as sine/cosine transformations to preserve temporal continuity.

**Rolling Statistics:** 4-hour moving averages and standard deviations capture short-term trends.

### Modeling

**Baselines:** Naive Persistence, Seasonal Naive

**Linear Models:** Elastic Net (primary), Ridge, Lasso

**Ensemble:** Gradient Boosting Regressor (HistGradientBoosting)

**Training Protocol:** 80/20 temporal split, T+48h forecast target, strict chronological ordering to prevent leakage.

## Key Findings

**Lag Structure Sensitivity:** Comparison between configurations reveals critical importance of short-term temporal dependencies. Initial configuration (lag_30m, lag_24h) achieved 7.52% linear improvement, while current configuration (lag_48h, lag_96h) degraded to -1.72%, demonstrating that 30-minute lags capture essential intra-day autocorrelation patterns.

**Ensemble Performance:** Gradient Boosting consistently outperformed linear models across both configurations (14.52% initial, 9.56% current), indicating non-linear methods better handle peak-hour demand variability.

**Feature Importance:** Top predictors are fourier_cos_1 (24-hour cycle), lag_30m (short-term autocorrelation), and temp_hour_interaction (weather-time coupling).

**Diagnostic Insights:** ACF analysis confirmed 24-hour cycle. Residual analysis revealed significant heteroscedasticity with peak-hour RMSE (0.0444) nearly 3x higher than night hours (0.0153).

### Research Finding

Current lag structure (lag_48h, lag_96h) resulted in linear model degradation compared to baseline. This demonstrates that energy forecasting requires short-term temporal dependencies (30-minute lags) to capture intra-day consumption dynamics. Future work will test optimal lag combinations through systematic ablation studies.

## Installation

```bash
git clone https://github.com/bcode0127-debug/Energy_Forecasting_Project.git
cd energy-forecasting
pip install -r requirements.txt
```

## Usage

```bash
python main.py
```

Outputs:
- results/final_model_comparison.csv
- results/figures/ (diagnostic plots)

## Project Structure

```
├── data_engineering/       # BigQuery SQL scripts
├── features/              # Temporal and Fourier features
├── models/                # Elastic Net, Gradient Boosting
├── evaluation/            # Diagnostics and metrics
├── results/               # Output tables and figures
└── main.py                # Pipeline
```

## Future Research Directions

**Ablation Studies:** Systematic removal of feature groups (weather, Fourier terms, lags) to quantify individual contributions.

**Heteroscedasticity Mitigation:** Weighted Least Squares or segment-specific models (night/day/peak hours).

**Uncertainty Quantification:** Prediction intervals via conformal prediction or quantile regression.

**Multi-Horizon Forecasting:** Extend to 7-day ahead predictions.

**Temporal Drift Detection:** Monitor model performance across seasons and trigger adaptive retraining.

## License

MIT License - see LICENSE file

