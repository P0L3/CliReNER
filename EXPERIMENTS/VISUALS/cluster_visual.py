import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import io


# Note: In your local environment, use: 
df = pd.read_csv("clirener_aggregated_mean_std_strict.csv")
# For this script to run immediately, I will assume 'df' is loaded.

# 2. Pivot the data to create a Matrix (Rows=Models, Cols=Tags)
# We fill NaNs with 0 (assuming missing means the model failed to predict that tag)
matrix = df.pivot(index='model_display_name', columns='tag', values='strict_f1_mean').fillna(0)

# 3. Visualization 1: Hierarchical Clustering (Clustermap)
# This reorders the rows and columns to group similar models and similar tags together.
plt.figure(figsize=(20, 15))
clustermap = sns.clustermap(
    matrix, 
    metric="euclidean", 
    method="ward", 
    cmap="viridis", 
    standard_scale=None, # Set to 1 if you want to compare relative patterns rather than absolute performance
    figsize=(18, 14),
    dendrogram_ratio=0.15
)
plt.title("Hierarchical Clustering of Models based on F1 Scores")
plt.show()

# 4. Visualization 2: PCA (Dimensionality Reduction)
# Projects the 28-dimensional tag space into 2D to see groupings.
pca = PCA(n_components=2)
coords = pca.fit_transform(matrix)

pca_df = pd.DataFrame(coords, columns=['PC1', 'PC2'], index=matrix.index)

plt.figure(figsize=(14, 10))
sns.scatterplot(x='PC1', y='PC2', data=pca_df, s=100)

# Add labels to points
for i, txt in enumerate(pca_df.index):
    plt.annotate(txt, (pca_df.iloc[i].PC1, pca_df.iloc[i].PC2), fontsize=9, alpha=0.7)

plt.title("PCA of Models (Grouping by Overall Similarity)")
plt.xlabel(f"Principal Component 1 ({pca.explained_variance_ratio_[0]:.2%} Variance)")
plt.ylabel(f"Principal Component 2 ({pca.explained_variance_ratio_[1]:.2%} Variance)")
plt.grid(True, alpha=0.3)
plt.show()