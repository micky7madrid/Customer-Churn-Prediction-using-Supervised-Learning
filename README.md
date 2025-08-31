# 📊 Customer Churn Prediction (CPS844 Project)

This project applies machine learning techniques to predict **customer churn** using the **Telco Customer Churn dataset** (7,043 records) from Kaggle. The goal is to analyze customer behavior and identify which clients are most likely to leave a telecommunications service provider.

---

## 📂 Dataset
- **Source:** [Kaggle - Telco Customer Churn](https://www.kaggle.com/blastchar/telco-customer-churn)  
- **Size:** 7,043 rows, 21 features + churn label  
- **Features:** Demographic info (gender, age), account details (contract type, tenure), billing details (monthly charges, total charges), and target variable `Churn`.

---

## 🔍 Preprocessing & EDA
- Converted categorical values into numeric (binary encoding, one-hot encoding).  
- Cleaned missing values in **TotalCharges** column and standardized numeric features (`tenure`, `MonthlyCharges`, `TotalCharges`).  
- Created visualizations (bar charts, histograms, correlation heatmaps) to analyze churn distribution and feature relationships.  

---

## ⚙️ Methodology
We implemented and compared **five supervised learning models**:
1. Logistic Regression (L1 Regularization)  
2. Decision Tree  
3. Random Forest  
4. Support Vector Machine (Linear Kernel)  
5. Gradient Boosting  

### Feature Selection
- Applied **Recursive Feature Elimination (RFE)** to identify the top 10 most important features (e.g., `Contract`, `Tenure`, `MonthlyCharges`, `PaperlessBilling`).

### Evaluation Metrics
- **Accuracy**  
- **F1 Score** (due to class imbalance)  
- **ROC AUC**  

---

## 📈 Results
- **Best Models:** Gradient Boosting & Logistic Regression (highest ROC AUC and F1 Score).  
- **Gradient Boosting:** Achieved **ROC AUC of 0.845** and strong F1 score.  
- **Effect of Feature Selection:** Performance remained stable with RFE (top 10 features), while models became simpler and more efficient.  
- **Decision Tree:** Weakest performer due to overfitting.  

---

## 🛠️ Tech Stack
- **Language:** Python  
- **Libraries:** pandas, NumPy, scikit-learn, matplotlib, seaborn  
- **Environment:** Jupyter Notebook / Python 3.x  
