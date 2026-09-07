import json
import numpy as np
import matplotlib.pyplot as plt

# Write a python script to perform dimensionality reduction based on the principal components of the dataset in
# “DimensionReduction.json”. This 10-dimensional dataset contains a "secret" 2D message, that you will find difficult,
# if not impossible, to decode with standard scatter plots of combinations of different dimensions.
# The 2D subspace where the message is legible corresponds to the two directions of greatest variance in the dataset.
# Thus, principal components analysis can be used to decode the message.

# Scatter plot the coordinates of the samples in the PCA linear vector space in the first two PCA dimensions.
# This represents the data in a 2-D reduced space of the 10-D dataset. To ensure it is displayed in the proper aspect
# ratio, use ax.set_aspect(1) on this axis object.  What is the secret message?

from PCA import *


# Load the dataset
f = open('DimensionalityReduction.json', 'rt')
dataset = json.load(f)
f.close()

# Convert dataset to a NumPy array
D = np.array(dataset)

# Perform PCA
p = pca(D)

# Print the eigenvalues
print("Eigenvalues:")
print(p.evals)

# Calculate the percentage of variance explained
percent_variance = 100 * p.evals / np.sum(p.evals)

print("\nPercent variance explained:")
print(percent_variance)

# Project the data into PCA space
D_pca = p.project(D)

# Plot the first two principal components
plt.figure()
plt.scatter(D_pca[:, 0], D_pca[:, 1], s=5)

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("Dimension Reduction using PCA")

plt.axis("equal")
plt.show()