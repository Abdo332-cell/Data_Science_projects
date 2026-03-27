import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("calc/train.csv")

print(df.head())
print(df.shape)
print(df.isnull().sum())

df["Age"] = df["Age"].fillna(df["Age"].median())
df = df.drop(columns=["Cabin"])
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])
print("after cleaning") 
print(df.isnull().sum())

survival_by_sex = df.groupby("Sex")["Survived"].mean()

plt.figure(figsize=(9,6))
plt.bar(survival_by_sex.index, survival_by_sex.values)
plt.title("Survival Rate by Sex (Titanic)")
plt.xlabel("Sex")
plt.ylabel("Survival Rate")
plt.grid(True, axis="y")
plt.show()


plt.figure(figsize=(8,5))
plt.hist(df["Age"])
plt.xlabel("Age")
plt.ylabel("count")
plt.title("Age")
plt.grid(True)
plt.show()



survival_by_class = df.groupby("Pclass")["Survived"].mean()
plt.figure(figsize=(9,6))
plt.bar(survival_by_class.index, survival_by_class.values)
plt.title("Survival Rate by Pclass Titanic")
plt.xlabel("Pclass")
plt.ylabel("Survival Rate")
plt.grid(True, axis="y")
plt.show()


plt.figure(figsize=(9,6))
plt.boxplot(df["Fare"])
plt.title("Fare")
plt.ylabel("Fare")
plt.grid(True)
plt.show()

