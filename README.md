# Energy Consumption Forecasting: Breaking the Linear Barrier

## Project Overview
This study illustrates a thorough, research-driven methodology for forecasting short-term energy consumption for the London Grid. By designing a cloud-native ETL pipeline utilizing Google BigQuery and Compute Engine, I effectively transformed and analyzed 167 million records. The research highlighted the importance of **Trustworthy AI** and **Explainability**, leading to a **14.52% improvement** in predictive accuracy by progressing from a **Naive Persistence Baseline** to an **Interpretable Linear Model**, and ultimately to a **Non-Linear Ensemble**.

## Key Results
* **Final Improvement:** 14.52% above the Naive Baseline.
* **Linear Frontier:** Achieved a 7.52% improvement with Elastic Net before encountering structural constraints.
* **Ensemble Breakthrough:** Successfully applied Gradient Boosting to capture non-linear household energy peaks.

---

## Technical Workflow
### 1. Cloud-Native Data Engineering (GCP) 
The basis of this project was the design of a scalable ETL pipeline on the **Google Cloud Platform (GCP)** to accommodate 167 million smart-meter records.  

* **Infrastructure Orchestration:** Set up a **Google Compute Engine (VM)** instance to act as the primary node for managing data movement and executing processing scripts in the cloud. * **Distributed SQL Transformation:** Performed high-efficiency joins and temporal aggregations directly within **Google BigQuery**. This facilitated the effective integration of raw LCL energy pings with external weather data sourced from **Visual Crossing**. 
* **Temporal Alignment:** Addressed the frequency discrepancy between hourly weather data and 30-minute energy pings by utilizing **Linear Interpolation** in the cloud environment, ensuring a continuous and high-fidelity feature matrix. 
* **Cloud Export Pipeline:** The final refined dataset was exported as `lcl_merged_data.csv`, marking the transition of the project from an intensive cloud-engineering phase to a local, high-performance modeling pipeline.

### 2. Data Integrity & Preprocessing
Following the cloud export, rigorous protocols were implemented to guarantee the validity of the model:

* **Leakage Prevention:** The implementation of rigorously enforced temporal splitting (80/20) and **T+48** target shifting guarantees that the model does not have access to future information during the training process. 
* **Anomalies:** Sensor errors and absent values were addressed using time-sensitive interpolation.

### 3. Feature Engineering (The "Physics" of the Grid) 
Instead of depending exclusively on unprocessed data, I developed features that reflect the physical patterns of a city:

* **Fourier Series Harmonics:** Illustrated the 24-hour cyclical "pulse" of energy demand as a combination of sine and cosine waves. 
* **Centered Interactions:** Analyzed the correlation between Temperature and Hour-of-Day to account for fluctuating weather effects. 
* **Temporal Lags:** Employed short-term (30m) and long-term (24h) dependencies to ground the model in historical data.

---

###  The Modeling Journey

| Phase | Model | Improvement | Audit Note |
| :--- | :--- | :--- | :--- |
| **Baseline** | Naive Persistence | 0.00% | Positioned at $RMSE = 0.0358$ |
| **Phase 1** | Elastic Net (Linear) | 7.52% | Reach the "Linear Ceiling"; faced challenges with peaks |
| **Phase 2** | GBR (Ensemble) | **14.52%** | Captured non-linear partitions and high-variance states |

---

## Trustworthy AI & Diagnostics 
To guarantee the dependability of the "Black Box" ensemble, I performed a systematic diagnostic audit:

* **Residual Analysis:** Detected heteroscedasticity (referred to as the "Fan Shape") during the linear phase, which served as the scientific basis for transitioning to non-linear modeling. 
* **ACF Audit:** Confirmed that the **Lag-48 (24-hour) spike** in error autocorrelation was significantly diminished, demonstrating that the model had "exhausted" the available temporal information. 
* **Permutation Importance:** Investigated the GBR to validate that significant features such as `fourier_cos_1` and `lag_30m` were influencing predictions, rather than random noise.

---

## Repository Structure
```text
├── data_engineering/  # BigQuery SQL and Dataset Merging scripts
├── features/       # Temporal and Fourier engineering modules
├── models/         # Elastic Net and HistGradientBoosting implementations
├── results/        # Final audit tables (final_model_comparison.csv) and diagnostics
└── main.py         # The unified research pipeline