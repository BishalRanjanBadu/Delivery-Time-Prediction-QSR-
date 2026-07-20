# 🍔 QSR Delivery Time Prediction

**Predicting food delivery ETAs with machine learning to improve operational planning and customer experience.**

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python)
![Model](https://img.shields.io/badge/Model-XGBoost-orange)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## 📌 Table of Contents
- [Overview](#overview)
- [Business Problem](#business-problem)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Feature Engineering](#feature-engineering)
- [Models & Results](#models--results)
- [Key Insights](#key-insights)
- [Tech Stack](#tech-stack)
- [Repository Structure](#repository-structure)
- [How to Run](#how-to-run)
- [Author](#author)

---

## Overview
This project builds a regression model that predicts **expected food delivery time (in minutes)** using logistics, rider, and environmental data. Accurate ETA prediction directly impacts customer satisfaction, delivery partner allocation, and QSR (Quick Service Restaurant) operational efficiency.

## Business Problem
Quick-commerce and food delivery platforms lose customer trust when quoted delivery times are inaccurate. This project frames delivery time estimation as a **supervised regression problem**, using historical order, rider, weather, and traffic data to generate reliable ETAs before an order is dispatched.

## Dataset
- **Size:** ~39,000 cleaned delivery records
- **Features:** rider age & rating, vehicle type & condition, weather conditions, road traffic density, order type, city type, festival indicator, order and pickup timestamps, delivery distance
- **Target:** `Time_taken (min)` — actual delivery duration

## Methodology
1. **Data Cleaning** — removed identifier columns (`ID`, `Delivery_person_ID`) and any post-outcome fields to prevent target leakage
2. **Datetime Feature Extraction** — parsed order and pickup timestamps into usable numeric features
3. **Missing Value Treatment** — median imputation for numerical fields (age, rating, prep time)
4. **Categorical Encoding** — one-hot encoding for weather, order type, vehicle type, and city; label encoding for festival flag
5. **Model Benchmarking** — compared Linear Regression, Random Forest, and XGBoost
6. **Hyperparameter Tuning** — `RandomizedSearchCV` on the XGBoost model
7. **Validation** — 10-fold cross-validation to confirm stability of generalization error

## Feature Engineering
| Feature | Description |
|---|---|
| `order_prep_time` | Time between order placement and rider pickup — strongest predictor of delay |
| `is_peak_hour` | Binary flag for lunch (11–14h) and dinner (18–22h) rush windows |
| `traffic_level` | Ordinal encoding of road traffic density (Low → Jam) |
| `distance_traffic` | Interaction term: delivery distance × traffic level |

Feature engineering improved model performance by **~23%** over the raw baseline feature set.

## Models & Results

| Model | Notes |
|---|---|
| Linear Regression | Baseline |
| Random Forest | Mid-complexity benchmark |
| **XGBoost (Final)** | Tuned via RandomizedSearchCV |

**Final Model Performance (XGBoost):**

| Metric | Score |
|---|---|
| R² | **0.82** |
| MAE | **3.09 minutes** |
| MAPE | **13.3%** |
| 10-Fold CV MAE | 3.11 (low variance — stable across folds) |

## Key Insights
- **Order preparation time**, **traffic-adjusted distance**, and **rider rating** are the top drivers of delivery duration.
- Vehicle condition and weather conditions have a measurable secondary effect.
- The model generalizes consistently (train/CV MAE gap is minimal), indicating low overfitting risk despite ensemble complexity.

## Tech Stack
`Python` · `Pandas` · `NumPy` · `Scikit-learn` · `XGBoost` · `Matplotlib` · `Joblib`

## Repository Structure
```
├── EXPECTED_DELIVERY_TIME_PREDICTION.ipynb   # Full analysis: EDA → feature engineering → modeling → tuning
├── predictions.csv                            # Actual vs. predicted delivery times (test set)
└── README.md
```

## How to Run
```bash
git clone https://github.com/BishalRanjanBadu/Delivery-Time-Prediction-QSR-.git
cd Delivery-Time-Prediction-QSR-
pip install pandas numpy scikit-learn xgboost matplotlib joblib
jupyter notebook "EXPECTED_DELIVERY_TIME_PREDICTION .ipynb"
```

## Author
**Bishal Ranjan Badu**
Data Science | Machine Learning | Predictive Analytics
