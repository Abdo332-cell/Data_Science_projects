# 🌤️ End-to-End Machine Learning Pipeline: Predicting Rain in Australia

## 📌 Project Overview & Workflow
This project implements a comprehensive, end-to-end Machine Learning pipeline designed to predict whether it will rain tomorrow in Australia (`RainTomorrow`) using historical meteorological data from the `weatherAUS.csv` dataset.

The project follows a structured data science workflow:
1. **Data Preprocessing & Cleaning**: Handling missing values using domain-specific strategies, dropping `RISK_MM` to prevent data leakage, and encoding binary/categorical variables.
2. **Exploratory Data Analysis (EDA)**: Visualizing feature distributions, understanding correlations, and identifying class imbalances.
3. **Feature Engineering & Scaling**: Extracting date features (Year, Month, Day) and normalizing numerical features using `StandardScaler`.
4. **Model Training & Comparison**: Training and benchmarking multiple ML algorithms including Logistic Regression, K-Nearest Neighbors (KNN), Support Vector Machines (SVM), and Random Forest.
5. **Evaluation Metrics**: Assessing models using Accuracy, Precision, Recall, and F1-Score.

---

## 📁 Dataset Description
The dataset used in this project is `weatherAUS.csv`, which contains daily weather observations from various locations across Australia.

### Key Features:
- **Date & Location**: Date of observation and Australian weather station location.
- **Temperature**: `MinTemp`, `MaxTemp`, `Temp9am`, `Temp3pm` (°C).
- **Rainfall & Evaporation**: `Rainfall` (mm), `Evaporation` (mm), and `Sunshine` (hours).
- **Wind**: `WindGustDir`, `WindGustSpeed` (km/h), `WindDir9am`, `WindDir3pm`, `WindSpeed9am`, `WindSpeed3pm`.
- **Atmospheric Conditions**: `Humidity9am`, `Humidity3pm` (%), `Pressure9am`, `Pressure3pm` (hPa), `Cloud9am`, `Cloud3pm` (oktas).
- **Targets & Flags**:
  - `RainToday`: Binary categorical (`Yes` / `No`) indicating if rainfall exceeded 1mm today.
  - `RainTomorrow` (**Target Variable**): Binary categorical (`Yes` / `No`) indicating if it rained the next day.
  - `RISK_MM`: Amount of rain on the next day (Dropped during preprocessing to prevent data leakage).

---

## ⚙️ Data Preprocessing & Cleaning
- **Handling Missing Values**:
  - **Numerical Features**: Imputed using group-wise median based on `Location` to maintain regional weather patterns. Any remaining nulls were filled using global column medians.
  - **Categorical Features**: Imputed using the mode (most frequent value) of each respective column.
- **Data Leakage Prevention**: Dropped `RISK_MM` as it directly reveals the target `RainTomorrow`.
- **Outlier Handling**: Identified using the Interquartile Range (IQR) method. Outliers were preserved because extreme values represent genuine severe weather events critical for rain prediction.
- **Feature Engineering**:
  - Extracted `Year`, `Month`, and `Day` from the `Date` column.
  - Normalized numerical variables using `StandardScaler`.
  - Encoded binary/categorical columns (`RainToday`, `Location`, wind directions).

---

## 📊 Models Evaluated
The following algorithms were trained and evaluated on the preprocessed data:
- **Logistic Regression** (Baseline linear classifier)
- **K-Nearest Neighbors (KNN)**
- **Support Vector Machine (SVM)**
- **Random Forest Classifier** (Ensemble tree model)

---

## 🚀 How to Run the Project

### 1. Prerequisites
Ensure you have Python installed along with the required dependencies:
```bash
pip install numpy pandas matplotlib seaborn scikit-learn
