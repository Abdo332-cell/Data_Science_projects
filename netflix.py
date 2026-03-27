import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv("calc/netflix_titles.csv")
print(df.head(7))
print(df.shape)
print(df.info())
print(df.isnull().sum())

df["director"]=df["director"].fillna("unknown")
df["cast"]=df["cast"].fillna("unknown")
df["country"]=df["country"].fillna("unknown")
df["date_added"]=df["date_added"].fillna("unknown")
df["rating"]=df["rating"].fillna("unknown")
print(df.isnull().sum())
 
print(df["type"].value_counts())

count=df["type"].value_counts()

plt.figure(figsize=(8,6))
plt.bar(count.index , count.values)
plt.title("TV show vs movie")
plt.xlabel("type")
plt.ylabel("count")
plt.grid(True)
plt.show()

top10_country=df["country"].value_counts().head(10)
print(top10_country)

plt.figure(figsize=(8,6))
plt.bar(top10_country.index , top10_country.values)
plt.title("top10")
plt.xlabel("country")
plt.ylabel("count")
plt.grid(True)
plt.show()

genres = df["listed_in"].str.split(", ").explode()
top10_listed_in = genres.value_counts().head(10)
print(top10_listed_in)

plt.figure(figsize=(19,17))
plt.bar(top10_listed_in.index , top10_listed_in.values)
plt.title("top10")
plt.xlabel("listed_in")
plt.ylabel("count")
plt.xticks(rotation=45) 
plt.grid(True)
plt.show()

df["date_added"] = df["date_added"].str.strip()
df["date_added"] = pd.to_datetime(df["date_added"], errors="coerce")
df["year_added"] = df["date_added"].dt.year
year_count = df["year_added"].value_counts().sort_index()
print(year_count)

plt.figure(figsize=(10,6))
plt.plot(year_count.index, year_count.values)
plt.title("Number of Titles Added per Year")
plt.xlabel("Year")
plt.ylabel("Count")
plt.grid(True)
plt.show()