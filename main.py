import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import f1_score, roc_auc_score, make_scorer
from sklearn.feature_selection import RFE

# ============================
# Step 1: Load and Explore Data
# ============================
# Load the dataset (adjust the filename/path as needed)
df = pd.read_csv('Telco-Customer-Churn.csv')

# 1.1 Display basic dataset information
print("Dataset shape:", df.shape)
print("\nDataset info:")
print(df.info())
print("\nSummary statistics:")
print(df.describe(include='all'))

# 1.2 Check for missing values
print("\nMissing values in each column:")
print(df.isnull().sum())

# 1.3 Visualize the distribution of the target variable "Churn"
churn_counts = df['Churn'].value_counts()
plt.figure(figsize=(6, 4))
churn_counts.plot(kind='bar', color=['skyblue', 'salmon'])
plt.title("Distribution of Customer Churn")
plt.xlabel("Churn")
plt.ylabel("Count")
plt.xticks(rotation=0)
plt.show()

# 1.4 Plot histograms for numeric features
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
df[numeric_cols].hist(figsize=(12, 10))
plt.tight_layout()
plt.show()

# 1.5 Visualize a correlation matrix for numeric features
corr = df[numeric_cols].corr()
plt.figure(figsize=(12, 10))
plt.imshow(corr, cmap='coolwarm', interpolation='none')
plt.colorbar()
plt.xticks(range(len(numeric_cols)), numeric_cols, rotation=90)
plt.yticks(range(len(numeric_cols)), numeric_cols)
plt.title("Correlation Matrix of Numeric Features")
plt.show()

# ============================
# Step 2: Pre-processing
# ============================
# 2.1 Convert TotalCharges to numeric
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
print("\nMissing values in TotalCharges after conversion:", df['TotalCharges'].isnull().sum())
df['TotalCharges'].fillna(0, inplace=True)

# 2.2 Drop the customerID column (not useful for prediction)
if 'customerID' in df.columns:
    df.drop('customerID', axis=1, inplace=True)

# 2.3 Convert binary categorical columns (Yes/No) to 1/0
binary_cols = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling', 'Churn']
for col in binary_cols:
    if col in df.columns:
        df[col] = df[col].map({'Yes': 1, 'No': 0})

# 2.4 One-hot encode non-binary categorical columns
categorical_cols = ['gender', 'MultipleLines', 'InternetService', 'OnlineSecurity', 
                    'OnlineBackup', 'DeviceProtection', 'TechSupport', 
                    'StreamingTV', 'StreamingMovies', 'Contract', 'PaymentMethod']
existing_cat_cols = [col for col in categorical_cols if col in df.columns]
df = pd.get_dummies(df, columns=existing_cat_cols, drop_first=True)

# 2.5 Feature Scaling for numerical features
numeric_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
scaler = StandardScaler()
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

print("\nFinal dataset shape after pre-processing:", df.shape)
print(df.head())

# ============================
# Step 3: Model Evaluation
# ============================
# Separate features and target
X = df.drop('Churn', axis=1)
y = df['Churn']

# Define 5 classifiers
models = {
    'Logistic Regression (L1)': LogisticRegression(penalty='l1', solver='liblinear', max_iter=1000, random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42),
    'SVM (Linear Kernel)': SVC(kernel='linear', probability=True, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42)
}

# Define a custom ROC AUC scorer that avoids unwanted keyword arguments.
def custom_roc_auc_scorer(estimator, X, y):
    try:
        y_score = estimator.predict_proba(X)[:, 1]
    except AttributeError:
        y_score = estimator.decision_function(X)
    return roc_auc_score(y, y_score)

# For F1 score, use the built-in scorer.
f1_scorer = make_scorer(f1_score)

# --- Evaluation Without Feature Selection ---
print("\n=== Evaluation WITHOUT Feature Selection (Full Feature Set) ===\n")
results_full = []
for name, model in models.items():
    acc_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
    f1_scores = cross_val_score(model, X, y, cv=5, scoring=f1_scorer)
    auc_scores = cross_val_score(model, X, y, cv=5, scoring=custom_roc_auc_scorer)
    
    results_full.append({
        "Model": name,
        "Accuracy": acc_scores.mean(),
        "F1 Score": f1_scores.mean(),
        "ROC AUC": auc_scores.mean()
    })
    
    print(f"{name}")
    print(f"  Accuracy : {acc_scores.mean():.4f} (+/- {acc_scores.std():.4f})")
    print(f"  F1 Score : {f1_scores.mean():.4f} (+/- {f1_scores.std():.4f})")
    print(f"  ROC AUC  : {auc_scores.mean():.4f} (+/- {auc_scores.std():.4f})")
    print("-" * 50)

# ============================
# Step 4: Model Evaluation with 5 Algorithms using feature selection
# ============================
# Separate features and target
X = df.drop('Churn', axis=1)
y = df['Churn']
models = {
    'Logistic Regression (L1)': LogisticRegression(penalty='l1', solver='liblinear', max_iter=1000, random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42),
    'SVM (Linear Kernel)': SVC(kernel='linear', probability=True, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42)
}
# Define a custom ROC AUC scorer that does not pass unwanted keyword arguments.
def custom_roc_auc_scorer(estimator, X, y):
    try:
        # Try to get probability estimates for the positive class.
        y_score = estimator.predict_proba(X)[:, 1]
    except AttributeError:
        # Fall back to decision_function if predict_proba is not available.
        y_score = estimator.decision_function(X)
    return roc_auc_score(y, y_score)

# For F1 score, we can use the built-in scorer.
f1_scorer = make_scorer(f1_score)
rfe_results = []
print("\nModel Evaluation using 5-Fold Cross-Validation:\n")

for name, model in models.items():
    print(f"Evaluating: {name}")
    try:
        # Apply RFE to select top 10 features
        rfe = RFE(estimator=model, n_features_to_select=10)
        rfe.fit(X, y)
        selected_features = X.columns[rfe.support_]
        X_rfe = X[selected_features]
        # Cross-validation on selected features
        acc_scores = cross_val_score(model, X_rfe, y, cv=5, scoring='accuracy')
        f1_scores = cross_val_score(model, X_rfe, y, cv=5, scoring=f1_scorer)
        auc_scores = cross_val_score(model, X_rfe, y, cv=5, scoring=custom_roc_auc_scorer)

        rfe_results.append({
            "Model": name,
            "Accuracy": acc_scores.mean(),
            "F1 Score": f1_scores.mean(),
            "ROC AUC": auc_scores.mean()
        })

        print(f"  ✅ Accuracy : {acc_scores.mean():.4f}")
        print(f"  ✅ F1 Score : {f1_scores.mean():.4f}")
        print(f"  ✅ ROC AUC  : {auc_scores.mean():.4f}")

    except Exception as e:
        print(f"  ❌ RFE Failed: {e}")
    
    print("-" * 50)

results_df = pd.DataFrame(rfe_results)
results_full_df = pd.DataFrame(results_full)
results_rfe_df = pd.DataFrame(rfe_results)

print("\nFinal Results WITHOUT Feature Selection:")
print(results_full_df.to_string(index=False))
print("\nFinal Results WITH Feature Selection (RFE):")
print(results_rfe_df.to_string(index=False))

