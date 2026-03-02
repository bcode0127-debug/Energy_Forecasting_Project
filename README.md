# Energy Consumption Forecasting: Breaking the Linear Barrier



## 1. Research Question

Can interpretable linear models achieve competitive performance in production energy forecasting systems, or non-linear ensemble methods necessary? 

This question is critical for grid operators who require explainable predictions for regulatory compliance while maintaining forecasting accuracy.

## 2. Abstract

This work presents a systematic evaluation om linear regression variants and ensemble methods for energy 
forecasting using 167M smart meter records from London Grid. Despite optimal feature engineering on GCP, linear models achieve only 8% improvement over baseline, while gradient boosting reaches 15%—a 7-point 
gap representing the cost of interpretability. Linear models experience 10% performance swings based on lag configuration and fail to address heteroscedasticity (2.9x variance during peak hours), indicating 
structural non-linearities. We propose a hybrid deployment strategy: linear models for off-peak (sufficient accuracy, regulatory compliance) and ensembles for peak hours (critical accuracy).

## 3. Key Findings

**Finding 1: Short-Term Lags Are Critical**

Linear models experience a 10% performance swing based on lag configuration. Models using 30-minute autocorrelation achieve 7-8% improvement over baseline, while 48-hour lags cause performance degradation.

**Finding 2: Linear Models Hit a Performance Ceiling**

Even with optimal feature engineering, linear models (OLS, ElasticNet, Huber) achieve approximately 8% improvement, compared to 15% for gradient boosting ensembles. This 7 percentage point gap represents the cost of interpretability.

**Finding 3: Heteroscedasticity Persists Across All Linear Approaches**

Peak-hour variance (6-9 PM) is 2.9x higher than night hours (3-6 AM). Weighted Least Squares fails to address this, indicating the variance pattern is driven by structural non-linearity rather than simple variance weighting issues.

**Finding 4: Ensemble Methods Are More Robust**

Gradient Boosting shows only 6% performance degradation with suboptimal lags, compared to 10% degradation for linear models, indicating lower sensitivity to feature engineering choices.

## 4. Results Overview

### Comparative Performance by Lag Configuration

| Model | lag_30m/24h | lag_48h/96h | Performance Swing |
|-------|-------------|-------------|-------------------|
| Naive Baseline | 0.0% | 0.0% | - |
| OLS | +7.81% | -2.05% | -9.86% |
| Elastic Net | +8.07% | -1.72% | -9.79% |
| Huber | +7.41% | -2.59% | -10.00% |
| WLS | +4.44% | -7.11% | -11.55% |
| GBR | +15.44% | +9.56% | -5.88% |

 **Interpretation:**

The optimal lag configuration (30 min/ 24 h) enables linear models to capture short-term autocorrelation and daily seasonality, delivering modest gains. 

The same models falter with longer lags because they miss immediate temporal dependencies. GBR remains positive under both configurations, confirming its ability to learn non linear relationships.


## 5. Technical Infrastructure

### Cloud-Native Data Engineering

**Platform:** Google Cloud Platform (BigQuery, Compute Engine)

**Data Processing:**
- Google Compute Engine VM for orchestration and script execution
- BigQuery for distributed SQL joins and temporal aggregations on 167M records
- Linear interpolation for aligning 30-minute energy readings with hourly weather data
- Export to local CSV for modeling pipeline

**Data Sources:**
- London Smart Meter Dataset: 167M records, 30-minute intervals (2012-2013)
- Visual Crossing Weather: Hourly temperature, humidity

## 6. Feature Engineering

**Temporal Features:**
- Fourier series harmonics (3 harmonics) for 24-hour cyclical patterns
- Sine/cosine encoding for hour, day, month (preserves cyclical continuity)
- Weekend/weekday indicators

**Lag Features (Tested Configurations):**
1. lag_30m/24h: 30-minute and 24-hour lags
2. lag_48h/96h: 48-hour and 96-hour lags

**Weather Features:**
- Temperature (centered), humidity
- Heating/Cooling Degree Days (HDD/CDD)
- Temperature-hour interaction terms

**Rolling Statistics:**
- 4-hour moving average and standard deviation

## 7. Modeling

**Baselines:** Naive Persistence

**Linear Models:** OLS, Elastic Net, Huber, Weighted Least Squares

**Ensemble:** Gradient Boosting Regressor (HistGradientBoosting)

**Training Protocol:**
- 80/20 temporal split (strict chronological ordering)
- T+48h forecast target (2-day ahead)
- Leakage prevention via temporal validation

## 8. Diagnostic Analysis

### Heteroscedasticity Detection

**Night hours (3-6 AM):** RMSE = 0.0148
**Peak hours (6-9 PM):** RMSE = 0.0431
**Variance Ratio:** 2.9x

**Interpretation:** Linear models exhibit fan-shaped residual patterns with systematically higher errors during peak demand periods. This heteroscedasticity persists across all linear variants, including WLS, indicating structural non-linearity in peak-hour consumption dynamics.

### Autocorrelation Function Analysis

ACF plots confirm strong 24-hour cyclical patterns (lag-48 spike) in residuals for linear models. Gradient Boosting successfully minimizes residual autocorrelation, demonstrating effective capture of temporal dependencies.

## 9. Practical Implications

### Hybrid Deployment Strategy

Based on these findings, grid operators should consider a hybrid approach:

**Off-Peak Hours (10 PM - 6 AM):**
- Deploy Elastic Net or Ridge regression
- Rationale: 7-8% improvement sufficient, interpretable coefficients for regulatory compliance
- Lower stakes: errors during low-demand periods are less costly

**Peak Hours (6 AM - 10 PM):**
- Deploy Gradient Boosting ensemble
- Rationale: 15% improvement critical during high-stakes periods
- Higher stakes: errors during peak demand impact grid stability and market costs

**Monitoring:**
- Track variance in real-time
- Auto-switch to ensemble when variance exceeds threshold
- Log model decisions for regulatory audit trails

## 10. Installation

```bash
git clone https://github.com/bcode0127-debug/Energy_Forecasting_Project.git
cd energy-forecasting
pip install -r requirements.txt
python main.py
```

## 11. Project Structure

```
├── data_engineering/      # BigQuery SQL scripts
├── features/              # Temporal and Fourier features
├── models/                # Linear models and ensemble
├── evaluation/            # Diagnostics and metrics
├── results/               # Output tables and figures
└── main.py                # Pipeline execution
```

## 12. Limitations

This study focuses exclusively on aggregated household consumption patterns. Individual household forecasting may exhibit different characteristics requiring alternative modeling approaches. The London Grid dataset represents European consumption patterns which may not generalize to other geographic regions with different climate and usage profiles.

## 13. License

MIT License - see LICENSE file

## 14. Acknowledgments

- Data Source: London Smart Meter Dataset (UK Power Networks)
- Weather Data: Visual Crossing 
- Cloud Infrastructure: Google Cloud Platform
