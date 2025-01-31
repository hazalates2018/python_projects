
import pandas as pd
df = pd.read_csv("C:\\Users\\hazal\\Downloads\\persona.csv")
print(df.head(10))

# Unique values in COUNTRY
print(df["COUNTRY"].unique())
# Unique values in SEX
print(df["SEX"].unique())

# Unique values in SOURCE
print(df["SOURCE"].unique())

# Age Distribution (Min, Max, Mean)
print(df["AGE"].min())
print(df["AGE"].max())
print(df["AGE"].mean())

# Filtering Users from "TUR" who use "android"
df_filtered=df[(df["COUNTRY"]=="tur")&(df["SOURCE"]=="android")]
print(df_filtered)


# Average PRICE for Each Country
print(df.groupby("COUNTRY")["PRICE"].mean())

# Top 5 Highest-Paying Users
top_5=df.nlargest(5,"PRICE")
print(top_5)

# Creating an Age Group Column
bin =[0,18,30,50]
my_labels=["0-18","19-30","31-50"]
df["AGE_CAT"]=pd.cut(df["AGE"],bins=bin,labels=my_labels)
print(df["AGE_CAT"].head())

# Compare Total Revenue from android vs ios
revenue_by_source = df.groupby("SOURCE")["PRICE"].sum()
print(revenue_by_source)

# Finding Outliers in PRICE using IQR
import numpy as np
Q1 = np.percentile(df["PRICE"],25) # 25th percentile
Q3 = np.percentile(df["PRICE"],75) # 75th percentile
IQR =Q3-Q1
print(IQR)
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
df_no_outliers = df[(df["PRICE"]>=lower_bound) & (df["PRICE"]<= upper_bound)]
print(df_no_outliers)

# Pivot Table Analysis: Average PRICE by COUNTRY and SOURCE
print(df.pivot_table(values="PRICE",index="COUNTRY",columns="SOURCE",aggfunc="mean"))

# Counting Users by Country
user_count_by_country = df["COUNTRY"].value_counts()
print(user_count_by_country)

# Finding the Most Common Age
most_common_age= df["AGE"].mode()[0]
print("Most Common Age:", most_common_age)

# Analyzing Gender Distribution by Country
gender_dist = df.groupby(["COUNTRY","SEX"]).size().unstack()
print(gender_dist)

df["AGE"].max()
bins=[0,16,30,45,df["AGE"].max()]
mylabels = ["0-16","17-30","31-45","46-66"]
df["AGE_CAT"]=pd.cut(df["AGE"],bins=bins,labels=mylabels)
print(df["AGE_CAT"])

# Finding the Highest-Paying Age Group
highest_paying_age_group = df.groupby("AGE_CAT")["PRICE"].mean().idxmax()
print("Highest Paying Age Group:", highest_paying_age_group)

# Finding the Most Popular Platform (iOS vs Android)
most_popular_platform=df["SOURCE"].value_counts().idxmax()
print(most_popular_platform)

# Creating a New Column for Age Categories
df["AGE_CATEGORY"]=df["AGE"].apply(lambda x: "Teen" if x<18 else "Senior")
print(df.head())

#Finding the Country with the Highest Revenue
highest_revenue_country = df.groupby("COUNTRY")["PRICE"].sum().idxmax()
print(highest_revenue_country)

#Identifying the Most Expensive Purchase
Identifying_the_Most_Expensive_Purchase= df.nlargest(1,"PRICE")
print(Identifying_the_Most_Expensive_Purchase)
#most_expensive_purchase = df.loc[df["PRICE"].idxmax()]
#print(most_expensive_purchase)

# Checking the Correlation Between Age and Price
correlation = df["AGE"].corr(df["PRICE"])
print(correlation)


#Correlation visualation Between Age and Price
import seaborn as sns
import matplotlib.pyplot as plt
numerical_df=df.select_dtypes(include=["number"])
correlation_matrix=numerical_df.corr()
sns.heatmap(correlation_matrix,annot=True,cmap="coolwarm",fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()


# Find the Country with the Highest Average Price
highest_avg_price_country = df.groupby("COUNTRY")["PRICE"].mean().idxmax()
print(highest_avg_price_country)

# Find the Gender That Spends the Most
highest_spending_gender = df.groupby("SEX")["PRICE"].sum().idxmax()
print("Gender That Spends the Most:",highest_spending_gender)

# How Many Users are from Each Platform (Android vs iOS)?
print(df["SOURCE"].value_counts())

# Find the Top 3 Age Groups that Spend the Most
print(df.groupby('AGE_CAT')["PRICE"].sum().nlargest(3))

# Find the Most Expensive Transaction
print(df.loc[df["PRICE"].idxmax()])

# Find the Youngest and Oldest Customers
youngest_customer = df.loc[df["AGE"].idxmin()]
oldest_customer = df.loc[df["AGE"].idxmax()]
print(youngest_customer)
print(oldest_customer)

# Visualize the Age Distribution of Customers
import matplotlib.pyplot as plt
plt.hist(df["AGE"],bins=10,edgecolor="black")
plt.xlabel("AGE")
plt.ylabel("Number of Customers")
plt.title("Age Distribution of Customers")

# Find the Top 5 Countries with the Highest Total Revenue
print(df.groupby("COUNTRY")["PRICE"].sum().nlargest(5))

# Create a New Column That Combines Country, Platform, and Gender
df["COUNTRY_SOURCE_GENDER"]=df["COUNTRY"]+"_"+df["SOURCE"]+"_"+df["SEX"]
print(df.head())

# Find the Top 5 Countries with the Highest Total Revenue
print(df.groupby("COUNTRY")["PRICE"].sum().nlargest(5))

# Create a New Column That Combines Country, Platform, and Gender
df["COUNTRY_SOURCE_GENDER"]=df["COUNTRY"]+"_"+df["SOURCE"]+"_"+df["SEX"]
print(df.head())

# Find the Most Common Customer Profile
print(df["COUNTRY_SOURCE_GENDER"].mode()[0])

# Count How Many Users Are in Each Age Group
print(df.groupby("AGE_CAT")["AGE_CAT"].count())
#df["AGE_CAT"].value_counts()


# Create a Pivot Table to See Average Price by Country and Platform
print(df.pivot_table(values="PRICE",index="COUNTRY",columns="SOURCE",aggfunc="mean"))

# Identify VIP Customers (Top 5% of Spenders)
df["PRICE"].quantile(0.95)
vip_customers = df[df["PRICE"] >= df["PRICE"].quantile(0.95)]
print(vip_customers)

# Calculate Customer Lifetime Value (CLV)
customer_ltv = df.groupby("COUNTRY_SOURCE_GENDER")["PRICE"].sum().sort_values(ascending=False)
print(customer_ltv.head(10))

# Predict Customer Spending Using Linear Regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
X =df[["AGE"]]
y = df[["PRICE"]]

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
print(predictions[:5])

# Find The Most Common Spending Range (Histogram)
import matplotlib.pyplot as plt
plt.hist(df["PRICE"],bins=10,edgecolor="red")
plt.xlabel("PRICE")
plt.ylabel("Spending Range")
plt.title("Spending Distribution of Customers")
plt.show()

# Cluster Customers Based on Spending Patterns (K-Means Clustering)
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
clustering_data = df[["AGE","PRICE"]]
clustering_data = clustering_data.dropna()
kmeans = KMeans(n_clusters=3,random_state=42)
df["CLUSTERS"] = kmeans.fit_predict(clustering_data)
print(df.groupby("CLUSTERS")[["AGE","PRICE"]].mean())


# Identify Anomalies in Customer Spending (Isolation Forest)
from sklearn.ensemble import IsolationForest
model = IsolationForest(contamination=0.05,random_state=42)
df["Anomaly"]= model.fit_predict(df[["PRICE"]])
anomalies = df[df["Anomaly"] == -1]
print(anomalies)

# Calculate the Rolling Average Price per Country
# df.groupby("COUNTRY")["PRICE"].mean()
df["Rolling_Avg"] = df.groupby("COUNTRY")["PRICE"].transform(lambda x:x.rolling(window=3,min_periods=1).mean())
print(df.head(20))

# Find Seasonal Trends in Spending
import seaborn as sns
import matplotlib.pyplot as plt
df_sorted = df.sort_values("AGE")
sns.lineplot(x=df_sorted["AGE"],y=df_sorted["PRICE"])
plt.xlabel("AGE")
plt.ylabel("PRICE")
plt.title("Spending Trend by Age")
plt.show()




