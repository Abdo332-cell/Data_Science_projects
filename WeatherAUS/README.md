# 🌤️ End-to-End Machine Learning Pipeline: Predicting Rain in Australia

## 📌 Project Overview
This project implements a complete Machine Learning pipeline to predict whether it will rain tomorrow in Australia (`RainTomorrow`) using daily meteorological observations from the `weatherAUS.csv` dataset[cite: 2].

The main stages of the project include:
1. **Data Preprocessing & Cleaning**: Handling missing values, dropping `RISK_MM` to prevent data leakage, and encoding categorical variables[cite: 2].
2. **Exploratory Data Analysis (EDA)**: Analyzing feature relationships, distributions, and target class balance[cite: 2].
3. **Feature Engineering & Scaling**: Extracting date features (Year, Month, Day) and scaling numerical features via `StandardScaler`[cite: 2].
4. **Model Training & Comparison**: Benchmarking Logistic Regression, K-Nearest Neighbors (KNN), and Random Forest Classifier[cite: 2].
5. **Model Evaluation**: Comparing model performance using Accuracy, Precision, Recall, and F1-Score[cite: 2].

---

## 📁 Dataset Features
The dataset contains historical daily weather metrics from multiple locations across Australia[cite: 2]:

- **Date & Location**: Observation date and station name[cite: 2].
- **Temperature**: `MinTemp`, `MaxTemp`, `Temp9am`, `Temp3pm` (°C)[cite: 2].
- **Precipitation & Sunshine**: `Rainfall` (mm), `Evaporation` (mm), `Sunshine` (hours)[cite: 2].
- **Wind Parameters**: Gust direction/speed, morning/afternoon wind speed and direction[cite: 2].
- **Atmospheric Metrics**: `Humidity9am`, `Humidity3pm` (%), `Pressure9am`, `Pressure3pm` (hPa), `Cloud9am`, `Cloud3pm` (oktas)[cite: 2].
- **Target Variable**: `RainTomorrow` (`Yes` / `No`)[cite: 2].

> **Note**: `RISK_MM` was removed during preprocessing because it directly records the rain amount on the following day, which causes data leakage[cite: 2].

---

## ⚙️ Data Preprocessing & Pipeline Steps
- **Missing Value Imputation**:
  - **Numerical**: Imputed using group-wise median by `Location` to retain local weather characteristics, followed by overall feature medians[cite: 2].
  - **Categorical**: Imputed using column modes[cite: 2].
- **Outlier Handling**: Evaluated using the Interquartile Range (IQR); retained to preserve extreme, real-world meteorological events[cite: 2].
- **Feature Scaling**: Numerical variables standardized using `StandardScaler`[cite: 2].

---

## 📊 Evaluation & Machine Learning Models
The model pipeline tests three distinct classification approaches:
- **Logistic Regression**: Linear baseline model[cite: 2].
- **K-Nearest Neighbors (KNN)**: Non-parametric distance-based classifier[cite: 2].
- **Random Forest Classifier**: Ensemble decision tree model for non-linear decision boundaries[cite: 2].

---

## 🛠️ Requirements & Quick Start

### 1. Requirements
```bash
pip install numpy pandas matplotlib seaborn scikit-learn
