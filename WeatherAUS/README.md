# 🌤️ WeatherAUS – Rain Prediction

## 📌 Project Overview

This project uses the **Australia Weather dataset** to build a Machine Learning classification model that predicts whether it will rain tomorrow (`RainTomorrow`) based on historical weather conditions.

The project follows a complete Data Science and Machine Learning workflow, starting from data exploration and cleaning to preprocessing, visualization, model training, and evaluation.

## 📊 Dataset

The dataset contains **142,193 records and 24 original columns** covering different weather measurements and observations from locations across Australia.

Some of the main features include:

- Minimum and maximum temperature
- Rainfall
- Evaporation
- Sunshine
- Wind speed and direction
- Humidity
- Atmospheric pressure
- Cloud coverage
- Rain today
- Location
- Date

The target variable is:

- `RainTomorrow` → Whether it will rain the following day (`Yes` / `No`)

## 🧹 Data Preprocessing

Several preprocessing steps were applied to prepare the dataset for Machine Learning:

- Checked the dataset structure and data types.
- Checked for missing values and duplicated records.
- Filled missing numerical values using the **median based on location**, with an overall median as a fallback.
- Filled missing categorical values using the **mode**.
- Removed the `RISK_MM` feature because it could cause **data leakage** when predicting `RainTomorrow`.
- Converted the `Date` column into `Year`, `Month`, and `Day`.
- Removed the original `Date` column.
- Kept detected outliers because extreme weather values can represent real weather conditions and may contain useful information.

## 🔎 Exploratory Data Analysis

EDA was performed to understand the dataset and identify patterns in the weather variables.

The analysis included:

- Feature distributions
- Class distribution
- Correlations between numerical variables
- Relationships between weather conditions and rainfall
- Differences between locations

The target variable was also found to be **imbalanced**, with non-rainy days representing the majority of observations.

## ⚙️ Feature Engineering

The target variable was converted into numerical values:

- `No` → `0`
- `Yes` → `1`

Categorical features were converted using **One-Hot Encoding** with `drop_first=True`.

The dataset was then split into:

- **80% Training data**
- **20% Testing data**

Stratification was used to maintain the class distribution between training and testing sets.

## 📏 Feature Scaling

`StandardScaler` was applied to the training features.

The scaler was fitted only on the training data and then used to transform the test data to avoid data leakage.

## 🤖 Machine Learning Models

Three classification models were trained and evaluated:

### 1. Logistic Regression

Used as a baseline classification model.

- Accuracy: **84.99%**
- Precision: **73.68%**
- Recall: **51.42%**
- F1-Score: **60.57%**

### 2. K-Nearest Neighbors (KNN)

A distance-based classification algorithm trained using the scaled features.

- Accuracy: **80.68%**
- Precision: **62.51%**
- Recall: **34.53%**
- F1-Score: **44.48%**

### 3. Random Forest

An ensemble-based model capable of capturing non-linear relationships in the data.

- Accuracy: **85.71%**
- Precision: **78.38%**
- Recall: **50.09%**
- F1-Score: **61.12%**

## 📈 Model Comparison

| Model | Accuracy | Precision | Recall | F1-Score |
|---|---:|---:|---:|---:|
| Logistic Regression | 84.99% | 73.68% | **51.42%** | 60.57% |
| KNN | 80.68% | 62.51% | 34.53% | 44.48% |
| Random Forest | **85.71%** | **78.38%** | 50.09% | **61.12%** |

Based on the evaluated metrics, **Random Forest achieved the best overall performance**, with the highest Accuracy, Precision, and F1-Score.

However, **Logistic Regression achieved a slightly higher Recall**, meaning it detected a slightly larger proportion of rainy days.

## 🛠️ Tools & Libraries

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Jupyter Notebook

## 🎯 Conclusion

This project demonstrates an end-to-end Machine Learning workflow for predicting rainfall in Australia.

The project covers:

**Data Understanding → Data Cleaning → EDA → Feature Engineering → Encoding → Scaling → Train/Test Split → Model Training → Model Evaluation**

The results show that the models can predict rainfall with reasonable accuracy. However, the relatively low recall for the `RainTomorrow = Yes` class means that the models still miss a significant number of rainy days.
