# Script for Project 1 part 1
# ECE 4370: Engineering for Surgery
# Fall 2026
# Author:   Ryan Michael Nuera; ryan.michael.a.nuera@vanderbilt.edu
#           Julia Ke; julia.ke@vanderbilt.edu
#           Moyo Fasola; moyosoreoluwa.o.fasola@vanderbilt.edu
#           Sabra Winston; sabra.m.winston@vanderbilt.edu

import json
import matplotlib.pyplot as plt
from PCA import *

# Write a python script to perform dimensionality reduction based on the principal components of the dataset in
# “DimensionReduction.json”. This 10-dimensional dataset contains a "secret" 2D message, that you will find difficult,
# if not impossible, to decode with standard scatter plots of combinations of different dimensions.
# The 2D subspace where the message is legible corresponds to the two directions of greatest variance in the dataset.
# Thus, principal components analysis can be used to decode the message.

# Scatter plot the coordinates of the samples in the PCA linear vector space in the first two PCA dimensions.
# This represents the data in a 2-D reduced space of the 10-D dataset. To ensure it is displayed in the proper aspect
# ratio, use ax.set_aspect(1) on this axis object.  What is the secret message?

# Load the dataset
f = open('DimensionalityReduction.json', 'rt')
dataset = json.load(f)
f.close()

# Convert dataset to a NumPy array
D = np.array(dataset)

# Perform PCA
p = pca(D)

# Project the data into PCA space
D_pca = p.project(D)

# Plot the first two principal components
fig, ax = plt.subplots()
ax.scatter(D_pca[:, 0], D_pca[:, 1], s=5)

ax.set_xlabel("Principal Component 1")
ax.set_ylabel("Principal Component 2")
ax.set_title("Dimension Reduction using PCA")

ax.set_aspect(1)
plt.show()

# SECRET MESSAGE: exp (j * pi) + 1 = 0