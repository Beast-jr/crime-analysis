# Multi-City Crime Analysis: Hotspot Prediction & Crime Forecasting

Comparative crime analytics across **Chicago, Los Angeles and New York City** using public police-complaint data (~20M raw records). The notebook covers EDA, grid-based **hotspot classification**, **time-series forecasting** (deep learning vs. ARIMA), **crime-category forecasting**, and an **ablation study** of hybrid architectures.

Originally built and run on Kaggle.

## Highlights

- Harmonised three differently-structured city datasets into one schema (date, crime type, lat/lon, area, city).
- Hotspot classification on a ~1 km grid with 8 classifiers plus a soft-voting ensemble, evaluated on a strict time-based split.
- SHAP interpretability for the best tree model in each city.
- Daily crime-count forecasting with LSTM, GRU, CNN and Transformer models, benchmarked against an ARIMA baseline.
- Per-category forecasting for the top 5 crime categories in each city.
- Ablation study on hybrid architectures (CNN+GRU, CNN+LSTM, GRU+Transformer, CNN+GRU+Transformer).

## Data

| City | Source file | Cleaned rows | Date range |
|---|---|---|---|
| Chicago | `Crimes_-_2001_to_Present` | ~8.42M | 2001-01-01 to 2026-03-09 |
| Los Angeles | `Crime_Data_from_2010_to_2019` | ~2.13M | 2010-01-01 to 2019-12-31 |
| New York | `NYPD_Complaint_Data_Historic` | ~9.49M | 2000-01-01 to 2024-12-31 |

Datasets are not included in this repo (size). They were loaded from Kaggle datasets by `shristibose`:
`crime-dataset` (Chicago + LA) and `new-york-dataset` (NYC), originally published by the City of Chicago, LAPD and NYPD open-data portals.

## Pipeline

1. **Preprocessing:** column standardisation across cities, city-wise date parsing (with a strict fix for NYC dates), cleaning, and saved dataset summaries.
2. **EDA (per city):** yearly/monthly/weekday/hourly trends, top crime types, area breakdowns, correlation heatmaps and spatial plots.
3. **Hotspot modelling**
   - Locations are binned into a 0.01° (~1 km) grid and aggregated to daily counts per cell.
   - **Label:** a cell-day is a hotspot if its crime count is above the median.
   - **Features:** `lag_1`, `lag_7`, `rolling_mean_7`, `rolling_std_7`, `day_of_week`, `month`.
   - **Split:** chronological 80/20 (no leakage from the future).
   - **Models:** KNN, Logistic Regression, Decision Tree, Naive Bayes, XGBoost, MLP, Linear SVM (calibrated), Random Forest, plus a soft-voting ensemble of the top 5.
   - **Metrics:** Accuracy, Balanced Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC, MCC, Cohen's Kappa, with confusion matrices and ROC curves.
   - **Explainability:** SHAP summary plots for the best tree model per city.
4. **Total-crime forecasting**
   - Daily series, 30-day look-back.
   - Deep learning: LSTM, GRU, CNN, Transformer (early stopping, LR scheduling, train/val loss health checks).
   - Baseline: ARIMA(3,1,2), with ADF stationarity tests.
   - Metrics: RMSE, MAE, MedAE, R², NRMSE, MAPE, sMAPE, WAPE, MASE. Residual and actual-vs-predicted plots are included.
5. **Category forecasting:** DL and ARIMA/SARIMA on the top 5 crime categories per city.
6. **Ablation study:** hybrid architectures compared to isolate the contribution of the CNN, recurrent and attention components.

## Key Results

**Hotspot classification (best model: XGBoost in all three cities)**

| City | Accuracy | F1 | ROC-AUC | PR-AUC | MCC |
|---|---|---|---|---|---|
| Chicago | 0.745 | 0.714 | 0.825 | 0.776 | 0.485 |
| Los Angeles | 0.784 | 0.525 | 0.804 | 0.637 | 0.411 |
| New York | 0.803 | 0.801 | 0.887 | 0.896 | 0.606 |

LA has a notably lower recall (~0.43) than the other cities, which pulls its F1 down despite similar accuracy.

**Daily total-crime forecasting (best DL model vs. ARIMA baseline, RMSE)**

| City | Best DL model | RMSE | R² | ARIMA RMSE |
|---|---|---|---|---|
| Chicago | LSTM | 61.1 | 0.54 | 93.9 |
| Los Angeles | GRU | 57.5 | 0.16 | 68.8 |
| New York | GRU | 102.2 | 0.49 | 143.4 |

Recurrent models (GRU/LSTM) beat CNN and Transformer on every city, and all of them beat ARIMA, which has R² ≈ 0 or below. Forecasting is hardest for Los Angeles (R² only 0.16).

## Project Structure

```
.
├── crime-analysis.ipynb   # full pipeline (EDA → hotspots → forecasting → ablation)
├── README.md
└── requirements.txt       # see Setup
```

The notebook also writes figures, tables and reports to output folders created in Cell 2.

## Setup

```bash
git clone https://github.com/Beast-jr/crime-analysis.git
cd crime-analysis
pip install numpy pandas matplotlib seaborn scikit-learn xgboost shap statsmodels pmdarima tensorflow joblib
jupyter notebook crime-analysis.ipynb
```

**Before running locally:** the notebook reads from `/kaggle/input/...`. Download the three CSVs from the Kaggle datasets above and update the paths in **Cell 3 (Dataset Paths)**. A GPU is recommended for the deep learning sections. The full notebook is heavy (millions of rows, many models), so expect a long run and high memory use.

## Tech Stack

Python, pandas, NumPy, scikit-learn, XGBoost, TensorFlow/Keras, statsmodels, pmdarima, SHAP, Matplotlib, Seaborn.

## Author

**Jahnavi**: [GitHub](https://github.com/Beast-jr) · [LinkedIn](https://linkedin.com/in/jahnavi-raja-0103bb314)

## Acknowledgements

Data from the City of Chicago, Los Angeles Police Department and NYPD open-data portals, accessed via Kaggle.
