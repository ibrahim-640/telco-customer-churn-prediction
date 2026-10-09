import pandas as pd
df = pd.read_csv("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")
print(df.head())
print(df.shape)
df.info()
print(df.isnull().sum())
print("Duplicates:", df.duplicated().sum())

total = pd.to_numeric(df["TotalCharges"], errors="coerce")
print("Bad TotalCharges entries:", total.isnull().sum())
print(df[total.isnull()][["tenure", "TotalCharges"]])
print(df["Churn"].value_counts())
print(df["Churn"].value_counts(normalize=True))

df["ChurnFlag"] = df["Churn"].map({"Yes": 1, "No": 0})
df["tenure_group"] = pd.cut(df["tenure"], bins=[-1, 12, 24, 48, 72],
                            labels=["0-12", "13-24", "25-48", "49-72"])
print(df.groupby("tenure_group", observed=True)["ChurnFlag"].mean())
print(df.groupby("Contract")["ChurnFlag"].mean().round(3))
import matplotlib.pyplot as plt
import seaborn as sns
sns.countplot(data=df, x="Churn")
plt.title("Churn distribution")
plt.show()

sns.countplot(data=df, x="Contract", hue="Churn")
plt.title("Churn by contract type")
plt.show()

sns.boxplot(data=df, x="Churn", y="tenure")
plt.title("Tenure by churn")
plt.show()

