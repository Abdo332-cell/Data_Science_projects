import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("calc/netflix_titles.csv")

print(df.head(7))
print(df.shape)
print(df.info())

print("Missing Values:")
print(df.isnull().sum())

df["director"] = df["director"].fillna("Unknown")
df["cast"] = df["cast"].fillna("Unknown")
df["country"] = df["country"].fillna("Unknown")
df["date_added"] = df["date_added"].fillna("Unknown")
df["rating"] = df["rating"].fillna("Unknown")

print(" Values After Cleaning:")
print(df.isnull().sum())

type_count = df["type"].value_counts()
print("Type Count:", type_count)

top10_country = df["country"].value_counts().head(10)
print("Top 10 Countries:", top10_country)

genres = df["listed_in"].str.split(", ").explode()
top10_genres = genres.value_counts().head(10)
print("Top 10 Genres:", top10_genres)


df["date_added"] = df["date_added"].str.strip()
df["date_added"] = pd.to_datetime(df["date_added"], errors="coerce")

df["year_added"] = df["date_added"].dt.year

year_count = df["year_added"].value_counts().sort_index()
print("Titles Added Per Year:", year_count)

plt.figure(figsize=(8,6))
sns.countplot(x="type", data=df)
plt.title("Movies vs TV Shows on Netflix")
plt.xlabel("Type")
plt.ylabel("Count")
plt.show()

plt.figure(figsize=(10,6))
sns.barplot(x=top10_country.values, y=top10_country.index)
plt.title("Top 10 Countries Producing Content")
plt.xlabel("Number of Titles")
plt.ylabel("Country")
plt.show()

plt.figure(figsize=(10,6))
sns.barplot(x=top10_genres.values, y=top10_genres.index)
plt.title("Top 10 Genres on Netflix")
plt.xlabel("Count")
plt.ylabel("Genre")
plt.show()

plt.figure(figsize=(10,6))
plt.plot(year_count.index, year_count.values, marker='o')
plt.title("Number of Titles Added per Year")
plt.xlabel("Year")
plt.ylabel("Count")
plt.grid(True)
plt.show()
