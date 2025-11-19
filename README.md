# Implementation-of-K-Means-Clustering-for-Customer-Segmentation

## AIM:
To write a program to implement the K Means Clustering for Customer Segmentation.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Load Data: Read the dataset using Pandas and explore it with .head() and .info() to understand
the structure.

2. Find Optimal Clusters (Elbow Method): Use the Elbow method by fitting KMeans for k values
from 1 to 10 and plot the Within-Cluster Sum of Squares (WCSS) to find the optimal number of
clusters.

3. Train KMeans Model: Initialize and fit the KMeans model using the chosen number of clusters
(e.g., 5) on the selected features (Annual Income and Spending Score).

4. Predict Clusters: Use the trained model to predict cluster labels and add them as a new column
to the dataset.

5. Segment Data: Split the data into separate DataFrames based on predicted cluster labels for
visualization.

6. Visualize Clusters: Plot the clusters using different colors to visualize how customers are
grouped based on income and spending score.

## Program:
```
Program to implement the K Means Clustering for Customer Segmentation.
Developed by: KIRAN MP
RegisterNumber: 212224230123
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df=pd.read_csv("Mall_Customers.csv")
df:
df.head()
df.tail()
df.info()
df.isnull().sum()
from sklearn.cluster import KMeans
wcss=[]
for i in range(1,11):
    kmeans=KMeans(n_clusters=i,init= "k-means++")
    kmeans.fit(df.iloc[:,3:])
    wcss.append(kmeans.inertia_)
plt.plot(range(1,11),wcss)
plt.xlabel("No. of Clusters")
plt.ylabel("wcss")
plt.title("Elbow Method")
kn=KMeans(n_clusters=5)
kn.fit(df.iloc[:,3:])
y_pred= kn.predict(df.iloc[:,3:])
y_pred
df["cluster"]=y_pred
df0=df[df["cluster"]==0]
df1=df[df["cluster"]==1]
df2=df[df["cluster"]==2]
df3=df[df["cluster"]==3]
df4=df[df["cluster"]==4]
plt.scatter(df0["Annual Income (k$)"],df0["Spending Score (1-
100)"],c="red",label="cluster0")
plt.scatter(df1["Annual Income (k$)"],df1["Spending Score (1-
100)"],c="black",label="cluster1")
plt.scatter(df2["Annual Income (k$)"],df2["Spending Score (1-
100)"],c="blue",label="cluster2")
plt.scatter(df3["Annual Income (k$)"],df3["Spending Score (1-
100)"],c="green",label="cluster3")
plt.scatter(df4["Annual Income (k$)"],df4["Spending Score (1-
100)"],c="magenta",label="cluster4")
plt.legend()
plt.title("Customer Segments")
```

## Output:
df:
<img width="817" height="580" alt="Screenshot 2025-11-19 172647" src="https://github.com/user-attachments/assets/85df2555-9539-41d2-98b2-d0eb64132531" />

df.head():

<img width="802" height="279" alt="Screenshot 2025-11-19 172701" src="https://github.com/user-attachments/assets/ca1e9143-5b10-4225-9dbe-4cee51513095" />

df.tail():

<img width="807" height="279" alt="Screenshot 2025-11-19 172709" src="https://github.com/user-attachments/assets/f0c9eb95-bdba-431d-9c80-9ebf4943d990" />

df.info():

<img width="720" height="357" alt="Screenshot 2025-11-19 172724" src="https://github.com/user-attachments/assets/80ee1d42-2776-4f77-aea3-19d0af91f637" />

df.isnull().sum():

<img width="409" height="199" alt="Screenshot 2025-11-19 172730" src="https://github.com/user-attachments/assets/882c1404-5672-403e-a78c-0ec020bd75f7" />

graph:

<img width="1018" height="767" alt="Screenshot 2025-11-19 172742" src="https://github.com/user-attachments/assets/15c8327d-249a-4ad3-b9ab-9cb89cc01442" />

kn.fit(df.iloc[:,3:]):

<img width="1079" height="98" alt="Screenshot 2025-11-19 172756" src="https://github.com/user-attachments/assets/2b8a847f-8e67-40d2-b6d5-db9a84eef589" />

y_pred:

<img width="977" height="302" alt="Screenshot 2025-11-19 172806" src="https://github.com/user-attachments/assets/8d7965bc-8cd8-452a-83ba-1be0328e49b5" />

Customer Segments:

<img width="998" height="805" alt="Screenshot 2025-11-19 172816" src="https://github.com/user-attachments/assets/fe71ad26-7ad3-4ad8-afe8-9adcbb05abad" />


## Result:
Thus the program to implement the K Means Clustering for Customer Segmentation is written and verified using python programming.
