# BLENDED LEARNING
# Implementation of Principal Component Analysis (PCA) for Dimensionality Reduction on Energy Data

## AIM:
To implement Principal Component Analysis (PCA) to reduce the dimensionality of the energy data.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import required libraries such as pandas, StandardScaler, PCA, matplotlib, and seaborn.
2.Load the dataset HeightsWeights.csv and select Height and Weight as input features.
3.Visualize the original data distribution using a scatter plot.
4.Standardize the features using StandardScaler and apply Principal Component Analysis (PCA).
5.Display the explained variance ratio and visualize the transformed data using a PCA scatter plot. 

## Program:
```
/*
Program to implement Principal Component Analysis (PCA) for dimensionality reduction on the energy data.
Developed by: ARTHI S
RegisterNumber:  212225220011
*/
```
```
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv("HeightsWeights.csv")

print("First 5 rows of the dataset:")
print(data.head())

X = data[['Height(Inches)', 'Weight(Pounds)']]

plt.figure(figsize=(8, 5))
sns.scatterplot(x='Height(Inches)', y="Weight(Pounds)", data=data)
plt.title("Original Data Distribution")
plt.show()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

print("Explained Variance Ratio:", pca.explained_variance_ratio_)

pca_df = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])

plt.figure(figsize=(6,5))
sns.scatterplot(x='PC1', y='PC2', data=pca_df)
plt.title("PCA Projection of Height and Weight")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.show()

```
## Output:
```
Screenshot 2026-03-15 154541.png
Screenshot 2026-03-15 154554.png
```

## Result:
Thus, Principal Component Analysis (PCA) was successfully implemented to reduce the dimensionality of the energy dataset.
