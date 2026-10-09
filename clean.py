import pandas as pd

df = pd.read_csv("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")

# 1. Drop the ID: it's a label, not a signal
df = df.drop(columns=["customerID"])

# 2. Fix TotalCharges: text -> numbers (blanks become NaN)
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

# 3. Fill those NaNs with 0 (these customers have not been billed yet)
df["TotalCharges"] = df["TotalCharges"].fillna(0)

# 4. Turn the target into numbers: Yes -> 1, No -> 0
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

# 5. Verify the result
print("Shape:", df.shape)
print("Missing values left:", df.isnull().sum().sum())
print(df.dtypes)
print(df["Churn"].value_counts())

# 6. Save the cleaned data
df.to_csv("data/clean.csv", index=False)
print("Saved data/clean.csv")