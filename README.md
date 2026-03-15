# BLENDED LEARNING
# Implementation of Principal Component Analysis (PCA) for Dimensionality Reduction on Energy Data

## AIM:
To implement Principal Component Analysis (PCA) to reduce the dimensionality of the energy data.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
```
1.Import required libraries such as pandas, StandardScaler, PCA, matplotlib, and seaborn.
2.Load the dataset HeightsWeights.csv and select Height and Weight as input features.
3.Visualize the original data distribution using a scatter plot.
4.Standardize the features using StandardScaler and apply Principal Component Analysis (PCA).
5.Display the explained variance ratio and visualize the transformed data using a PCA scatter plot.
```
## Program:
```
/*
Program to implement Principal Component Analysis (PCA) for dimensionality reduction on the energy data.
Developed by: ARTHI S
RegisterNumber:  212225220011
*/
```
```
# Import necessary libraries
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Load the dataset
data = pd.read_csv("HeightsWeights.csv")

# Step 2: Display first few rows
print("First 5 rows of the dataset:")
print(data.head())

# Step 3: Select features
X = data[['Height(Inches)', 'Weight(Pounds)']]

# Step 4: Visualize original data
plt.figure(figsize=(8, 5))
sns.scatterplot(x='Height(Inches)', y="Weight(Pounds)", data=data)
plt.title("Original Data Distribution")
plt.show()

# Step 5: Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 6: Apply PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Step 7: Explained variance
print("Explained Variance Ratio:", pca.explained_variance_ratio_)

# Step 8: Create PCA Dataframe
pca_df = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])

# Step 9: Plot PCA result
plt.figure(figsize=(6,5))
sns.scatterplot(x='PC1', y='PC2', data=pca_df)
plt.title("PCA Projection of Height and Weight")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.show()
```
## Output:

## Result:
Thus, Principal Component Analysis (PCA) was successfully implemented to reduce the dimensionality of the energy dataset.
