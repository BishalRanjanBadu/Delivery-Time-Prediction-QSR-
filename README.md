#  Zomato Delivery Time Prediction

## 📌 Overview

This project builds an end-to-end Machine Learning model to predict **food delivery time** using real-world logistics data. The objective is to help optimize delivery operations by understanding key factors affecting delivery performance.

---

## 🎯 Problem Statement

Accurately predicting delivery time is critical for:

* Enhancing customer experience
* Optimizing delivery partner allocation
* Reducing operational delays

---

## 📊 Dataset

* ~39,000 delivery records
* Features include:

  * Delivery partner details
  * Location coordinates
  * Traffic & weather conditions
  * Order type and vehicle type

---

## 🛠️ Approach

### 🔹 Data Preprocessing

* Handled missing values using median/mode imputation
* Converted time columns to extract meaningful features
* Removed data leakage (`delivery_speed`)

### 🔹 Feature Engineering

* Created high-impact features:

  * `distance_traffic` (distance × traffic intensity)
  * `order_prep_time`
  * `is_peak_hour`
* Encoded categorical variables using One-Hot Encoding

### 🔹 Modeling

Trained and compared multiple models:

* Linear Regression (Baseline)
* Random Forest Regressor
* XGBoost Regressor (Final Model)

### 🔹 Hyperparameter Tuning

* Used RandomizedSearchCV for optimizing XGBoost
* Improved model generalization and performance

### 🔹 Model Validation

* Performed 5-fold Cross Validation
* Ensured consistency and avoided overfitting

---

## 📈 Results

| Model             | MAE      | RMSE     | R²       | MAPE       |
| ----------------- | -------- | -------- | -------- | ---------- |
| Linear Regression | 4.75     | 5.94     | 0.58     | 20.8%      |
| Random Forest     | 3.35     | 4.21     | 0.79     | 14.46%     |
| XGBoost (Tuned)   | **3.09** | **3.86** | **0.82** | **13.32%** |

✅ Final Model: **XGBoost Regressor**

---

## 🔍 Key Insights

* Delivery time is strongly influenced by:

  * Delivery partner rating
  * Traffic-adjusted distance
  * Vehicle condition
  * Weather conditions
* Feature engineering significantly improved model performance

---

## 📦 Tech Stack

* Python
* Pandas, NumPy
* Scikit-learn
* XGBoost

---


---

## 💡 Future Improvements

* Deploy model using Streamlit
* Add real-time prediction API
* Incorporate geospatial clustering
* Use deep learning models for further improvement

---

## 👨‍💻 Author

**Bishal Ranjan Badu**

---

## ⭐ If you found this useful

Give this repo a ⭐ and connect with me!
