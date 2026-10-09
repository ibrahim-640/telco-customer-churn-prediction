import pandas as pd
from sklearn.base import clone
import joblib
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.ensemble import RandomForestClassifier

# ---- Load and split (same as Step 3) ----
df = pd.read_csv("data/clean.csv")
X = df.drop(columns=["Churn"])
y = df["Churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# ---- Preprocessing (same as Step 3) ----
num_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
cat_cols = [c for c in X.columns if c not in num_cols and c != "SeniorCitizen"]

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
    ],
    remainder="passthrough",
)

# ---- Helper: train a model and print how it did ----
def evaluate(name, model):
    pipe = Pipeline([("prep", clone(preprocessor)), ("model", model)])
    pipe.fit(X_train, y_train)

    preds = pipe.predict(X_test)
    proba = pipe.predict_proba(X_test)[:, 1]

    print(f"\n===== {name} =====")
    print(classification_report(y_test, preds, target_names=["Stay", "Churn"],
                                zero_division=0))
    print("Confusion matrix (rows = actual, columns = predicted):")
    print(confusion_matrix(y_test, preds))
    print("ROC AUC:", round(roc_auc_score(y_test, proba), 3))
    return pipe

# ---- 1. Baseline: always predicts "Stay" ----
evaluate("Baseline (always Stay)", DummyClassifier(strategy="most_frequent"))

# ---- 2. Logistic Regression ----
lr_pipe = evaluate(
    "Logistic Regression",
    LogisticRegression(max_iter=1000, class_weight="balanced"),
)
# ---- 3. Random Forest ----
rf_pipe = evaluate(
    "Random Forest",
    RandomForestClassifier(
        n_estimators=300,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    ),
)
# ---- 4. What drives churn? (Logistic Regression weights) ----
names = lr_pipe.named_steps["prep"].get_feature_names_out()
coefs = lr_pipe.named_steps["model"].coef_[0]
drivers = pd.Series(coefs, index=names).sort_values()

print("\nTop factors REDUCING churn risk:")
print(drivers.head(8).round(2))
print("\nTop factors INCREASING churn risk:")
print(drivers.tail(8).round(2))

# ---- 5. Save the model ----
joblib.dump(lr_pipe, "model.joblib")
print("\nSaved model.joblib")