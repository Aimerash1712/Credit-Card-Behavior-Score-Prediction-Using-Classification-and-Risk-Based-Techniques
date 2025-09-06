# **Credit Card Behavior Score Prediction Using Classification and Risk-Based Techniques**

## 📌 Project Overview
This project predicts **next-month credit card default risk** using machine learning classification models and risk-based features. It helps financial institutions **identify high-risk customers early**, reducing credit losses and improving decision-making.

---

## 🔍 Problem Statement
Credit card issuers need to predict whether a customer will **default next month** using historical data (demographics, payment history, financial ratios). Early detection of defaulters ensures **risk mitigation and better credit policy management**.

---

## 🎯 Objectives
1. Predict `next_month_default` (0 = No Default, 1 = Default).  
2. Compare multiple **classification models** for performance.  
3. Handle **class imbalance** using advanced techniques.  
4. Provide **EDA insights** for customer risk behavior.

---

## 📂 Dataset
- **Train File:** `train_dataset_final1.csv`  
- **Validation File:** `validate_dataset_final.csv`  
- **Rows:** ~30,000  

**Columns:**  
- **Demographic:** `age`, `sex`, `marriage`, `education`  
- **Credit Limit:** `LIMIT_BAL`  
- **Payment History:** `pay_0` to `pay_6`  
- **Billing:** `Bill_amt1` to `Bill_amt6`  
- **Payments:** `pay_amt1` to `pay_amt6`  
- **Derived Features:** `AVG_Bill_amt`, `PAY_TO_BILL_ratio`  
- **Target:** `next_month_default`

---

## ⚙️ Tech Stack
- **Language:** Python  
- **Libraries:**  
  - Data Handling: `pandas`, `numpy`  
  - Visualization: `matplotlib`, `seaborn`  
  - ML Models: `scikit-learn`, `xgboost`, `lightgbm`, `imblearn`  
- **Techniques:**  
  - Feature Scaling (`StandardScaler`)  
  - Class Balancing (`SMOTE`)  
  - Model Evaluation (`F2-score`, `Recall`, `Accuracy`, `F1`)  

---

## 📊 Exploratory Data Analysis (EDA)
- **Target Imbalance:** Found ~80:20 ratio of No Default vs Default.  
- **Insights:**  
  - Higher payment delays → Higher default probability.  
  - Lower Pay-to-Bill ratio → Higher risk.  
  - Younger customers → More prone to default.  

✅ Example Plots:  
- Class Distribution  
- Pay-to-Bill Ratio by Default  
- Age vs Default Status  
- Payment Delays Trend

---

## 🧠 Model Development

### **1. Preprocessing**
- Removed `Customer_ID`  
- Filled missing values (`age` → median)  
- Applied **StandardScaler**  

### **2. Feature Engineering**
Added risk-based features using `feature_engineering(df)`:  
- `num_past_due` → Number of past due months  
- `max_delay` → Maximum payment delay  
- `recent_trend` → Recent payment trend  
- `utilization_ratio` → Credit utilization  
- `payment_trend` → Change in payment amount over time  
- `payment_std` → Payment variability  
- `bill_std` → Bill variability  

### **3. Handling Imbalance**
- Applied **SMOTE** to balance the dataset.  

### **4. Models Implemented**
- Logistic Regression  
- Random Forest  
- Gradient Boosting  
- XGBoost  
- LightGBM  

### **5. Evaluation Metrics**
- **Accuracy**  
- **Recall** (Most important for catching defaulters)  
- **F1-Score**  
- **F2-Score** (Recall weighted)

---

## 🏆 Model Performance

| **Model**             | **Accuracy** | **Recall** | **F1**  | **F2**  |
|------------------------|-------------|-----------|---------|---------|
| Logistic Regression    | 0.6820     | 0.6466    | 0.4364  | 0.5421  |
| Random Forest          | 0.8298     | 0.6178    | 0.5802  | 0.6022  |
| Gradient Boosting      | 0.8505     | 0.4641    | 0.5417  | 0.4923  |
| **XGBoost**            | **0.7498** | **0.8866**| 0.5744  | **0.7282**|
| LightGBM              | 0.8476     | 0.4581    | 0.5337  | 0.4856  |

✅ **Best Model:** XGBoost (Highest F2 and Recall)

---

## 📦 Output
Final predictions saved in:  
- `submission_22123043_Amey_Som.csv`  

Columns: `Customer_ID`, `next_month_default`

---

