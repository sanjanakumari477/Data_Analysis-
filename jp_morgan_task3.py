import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score

# --- STEP 1: Load data ---
df = pd.read_csv(r"C:\Users\sksan\Downloads\Task 3 and 4_Loan_Data.csv")

# Separate features and target
X = df.drop(columns=["customer_id", "default"])
y = df["default"]

# Save the column order
feature_cols = X.columns.tolist()

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- STEP 2: Train models ---
log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train_scaled, y_train)

tree_model = DecisionTreeClassifier(max_depth=5, random_state=42)
tree_model.fit(X_train, y_train)

# --- STEP 3: Evaluate using AUC (higher = better) ---
log_auc = roc_auc_score(y_test, log_model.predict_proba(X_test_scaled)[:, 1])
tree_auc = roc_auc_score(y_test, tree_model.predict_proba(X_test)[:, 1])

print(" Model Comparison (AUC Scores)")
print("----------------------------------")
print(f"Logistic Regression AUC : {log_auc:.4f}")
print(f"Decision Tree AUC       : {tree_auc:.4f}")
print()

# --- STEP 4: Function to compare Expected Loss ---
def compare_expected_loss(income, loan_amt_outstanding, years_employed,
                          fico_score, credit_lines_outstanding, total_debt_outstanding):
    recovery_rate = 0.10
    LGD = 1 - recovery_rate  # 0.9

    # Build input dataframe in EXACT same column order as training
    input_data = pd.DataFrame([{
        "income": income,
        "loan_amt_outstanding": loan_amt_outstanding,
        "years_employed": years_employed,
        "fico_score": fico_score,
        "credit_lines_outstanding": credit_lines_outstanding,
        "total_debt_outstanding": total_debt_outstanding
    }])[feature_cols]  # ensure exact same column order

    # Logistic Regression
    input_scaled = scaler.transform(input_data)
    log_pd = log_model.predict_proba(input_scaled)[:, 1][0]
    log_el = log_pd * LGD * loan_amt_outstanding

    # Decision Tree
    tree_pd = tree_model.predict_proba(input_data)[:, 1][0]
    tree_el = tree_pd * LGD * loan_amt_outstanding

    print(" Comparison for Given Borrower")
    print("----------------------------------")
    print(f"Logistic Regression → PD: {log_pd:.4f} | Expected Loss: ₹{log_el:.2f}")
    print(f"Decision Tree       → PD: {tree_pd:.4f} | Expected Loss: ₹{tree_el:.2f}")

    return {"Logistic_PD": log_pd, "Logistic_EL": log_el,
            "Tree_PD": tree_pd, "Tree_EL": tree_el}

# --- STEP 5: Example test case ---
compare_expected_loss(
    income=55000,
    loan_amt_outstanding=2500,
    years_employed=5,
    fico_score=680,
    credit_lines_outstanding=3,
    total_debt_outstanding=12000
)

