# 🚢 Titanic – Survival Prediction

## 📌 Project Overview

This project analyzes the Titanic dataset and builds Machine Learning classification models to predict whether a passenger survived the disaster.

The project follows a complete Data Science workflow, including data exploration, cleaning, visualization, feature engineering, model training, and evaluation.

## 📊 Dataset

The dataset contains **891 records and 12 original columns** describing Titanic passengers.

Some of the main features include:

- Passenger Class
- Gender
- Age
- Number of Siblings/Spouses
- Number of Parents/Children
- Passenger Fare
- Port of Embarkation
- Survival Status

The target variable is:

- `Survived` – Whether the passenger survived (`1`) or did not survive (`0`)

## 🧹 Data Preprocessing

Several preprocessing steps were applied:

- Checked the dataset structure and data types.
- Checked for missing values and duplicated records.
- Dropped the `Cabin` column because of its high percentage of missing values.
- Filled missing `Age` values using the median.
- Filled missing `Embarked` values using the mode.
- Created a new `FamilySize` feature using `SibSp + Parch + 1`.
- Removed unnecessary columns such as `PassengerId`, `Name`, and `Ticket`.

## 🔎 Exploratory Data Analysis

EDA was performed to understand the factors affecting passenger survival.

The analysis included:

- Survival distribution
- Survival by gender
- Survival by passenger class
- Survival by gender and class
- Age distribution
- Fare vs. survival
- Family size vs. survival
- Correlation analysis

### Key Insights

- Female passengers had a much higher survival rate than male passengers.
- 1st-class passengers had the highest survival rate.
- 3rd-class passengers had the lowest survival rate.
- Passengers who paid higher fares generally had better survival rates.
- Smaller family groups showed better survival rates than larger groups.

## ⚙️ Feature Engineering

The target variable was already numerical:

- `No → 0`
- `Yes → 1`

Categorical features were converted using **One-Hot Encoding**.

The dataset was split into:

- **80% Training Data**
- **20% Testing Data**

Stratification was used to maintain the class distribution.

## 📏 Feature Scaling

`StandardScaler` was applied to the numerical features for models that require scaled data.

The scaler was fitted only on the training data and then used to transform the test data to avoid data leakage.

## 🤖 Machine Learning Models

Three classification models were trained and evaluated:

### 1. Logistic Regression

- Accuracy: **81.56%**
- Precision: **80.00%**
- Recall: **69.57%**
- F1-Score: **74.42%**

### 2. K-Nearest Neighbors (KNN)

- Accuracy: **82.12%**
- Precision: **80.33%**
- Recall: **71.01%**
- F1-Score: **75.39%**

### 3. Random Forest

- Accuracy: **82.12%**
- Precision: **78.46%**
- Recall: **73.91%**
- F1-Score: **76.12%**

## 🏆 Conclusion

Among the tested models, **Random Forest** achieved the best overall performance based on the **highest Recall and F1-Score**, with an accuracy of **82.12%**.

The project demonstrates how data preprocessing, exploratory analysis, feature engineering, and Machine Learning can be combined to analyze and predict passenger survival.
