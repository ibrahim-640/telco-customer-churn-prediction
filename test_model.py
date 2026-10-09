import pandas as pd
import joblib
from sklearn.model_selection import train_test_split

# Load the saved model and the cleaned data
model = joblib.load("model.joblib")

df = pd.read_csv("data/clean.csv")
X = df.drop(columns=["Churn"])
y = df["Churn"]

# Same split settings as train.py, so these customers were never seen in training
_, X_test, _, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# ---- Part 1: spot check on 15 random unseen customers ----
sample = X_test.sample(15, random_state=1)
probs = model.predict_proba(sample)[:, 1]

out = sample[["tenure", "Contract", "InternetService", "MonthlyCharges"]].copy()
out["churn_prob"] = probs.round(2)
out["predicted"] = (probs >= 0.5).astype(int)
out["actual"] = y_test.loc[sample.index]
out["correct"] = out["predicted"] == out["actual"]

print(print(out.to_string()))
print("Correct:", out["correct"].sum(), "of", len(out))

# ---- Part 2: score ALL unseen customers and save them ----
all_probs = model.predict_proba(X_test)[:, 1]

results = X_test.copy()
results["churn_prob"] = all_probs.round(3)
results["predicted"] = (all_probs >= 0.5).astype(int)
results["actual"] = y_test
results = results.sort_values("churn_prob", ascending=False)

results.to_csv("data/test_predictions.csv")
print("\nSaved", len(results), "predictions to data/test_predictions.csv")

# ---- Part 3: how good is the top of the list? ----
top100 = results.head(100)
print("Churn rate in the 100 riskiest customers:", round(top100["actual"].mean(), 2))
print("Churn rate across all test customers:   ", round(results["actual"].mean(), 2))