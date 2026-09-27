# 🚀 Spaceship Titanic — End-to-End Machine Learning Pipeline

> An end-to-end Machine Learning pipeline built to predict which passengers were transported to an alternate dimension during the *Spaceship Titanic* collision with a spacetime anomaly.

---

## 📌 Overview

This project implements a complete Machine Learning workflow from raw data ingestion, exploratory data analysis (EDA), and feature engineering, to model training, hyperparameter tuning, and deployment-ready pipeline architecture. 

Based on the popular [Kaggle Spaceship Titanic Competition](https://www.kaggle.com/competitions/spaceship-titanic), the objective is to predict the binary target variable `Transported` using personal, itinerary, and spending features of the passengers.

---

## 🛠️ Tech Stack & Key Tools

* **Language**: Python 3.10+
* **Data Processing & Analysis**: Pandas, NumPy
* **Data Visualization**: Seaborn, Matplotlib
* **Machine Learning**: Scikit-Learn, XGBoost, LightGBM
* **Model Experimentation & Pipeline**: Joblib / Pipeline architecture

---

## 📂 Dataset Summary

The dataset consists of approximately `8,700` training records containing personal, geographical, and monetary features:

* **Passenger Identification**: `PassengerId` (extracted into `Group` and `GroupSize`).
* **Demographics & Travel Info**: `HomePlanet`, `CryoSleep`, `Cabin` (parsed into `Deck`, `Num`, `Side`), `Destination`, `Age`, `VIP`.
* **Amenities Spending**: `RoomService`, `FoodCourt`, `ShoppingMall`, `Spa`, `VRDeck`.
* **Target Variable**: `Transported` (Boolean: `True` / `False`).

---

## ⚙️ End-to-End Project Workflow
1. **Exploratory Data Analysis (EDA)**:
   - Evaluated missing data patterns across passenger groups and cabins.
   - Analyzed spending behavior correlation with transport status.

2. **Feature Engineering**:
   - **Cabin Extraction**: Split `Cabin` into `Deck`, `Cabin_Num`, and `Side` (Port/Starboard).
   - **Group Mapping**: Extracted group IDs from `PassengerId` to compute group size and solo traveler indicators.
   - **Total Spending Aggregation**: Created `TotalSpending` and zero-spending indicators (`NoSpending`) from amenity features.

3. **Preprocessing Pipeline**:
   - Imputed missing values using domain-specific rules and median/mode strategies.
   - Scaled numerical features using `StandardScaler` / `RobustScaler`.
   - One-Hot Encoded categorical variables for tree-based and linear models.

4. **Model Training & Evaluation**:
   - Experimented with multiple classifiers: Logistic Regression, Random Forest, XGBoost, and LightGBM.
   - Performed hyperparameter optimization via Cross-Validation (`GridSearchCV` / `RandomizedSearchCV`).
