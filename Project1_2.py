# Script for Project 1 part 2
# ECE 4370: Engineering for Surgery
# Fall 2026
# Author:   Sabra Winston; sabra.m.winston@vanderbilt.edu\
#           Ryan Michael Nuera; ryan.michael.a.nuera@vanderbilt.edu
#           Julia Ke; julia.ke@vanderbilt.edu
#           Moyo Fasola; moyosoreoluwa.fasola@vanderbilt.edu

import json
import numpy as np
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
from PCA import *

# Write a python script to perform dimensionality reduction based on the principal components of the
# "human activity" dataset (smartphone accelerometer/gyroscope recordings during sitting, standing,
# walking, running, and dancing).

# 1) Load the data dictionary
f = open('humact.json', 'rt')
d = json.load(f)
f.close()

feat = np.array(d['feat'])          # NxM feature matrix
actid = np.array(d['actid'])        # N-length vector of activity codes (1-5)
actnames = d['actnames']            # names corresponding to codes 1-5

# 2) Perform PCA on the feature matrix
p = pca(feat)
feat_pca = p.project(feat)

# 3) Scatter plot of the first two PCA dimensions, one marker/color per activity, with a legend
fig, ax = plt.subplots()
markers = ['o', 's', '^', 'D', 'v']
for code in range(1, 6):
    idx = actid == code
    ax.scatter(feat_pca[idx, 0], feat_pca[idx, 1],
               s=5, marker=markers[code - 1], label=actnames[code - 1])

ax.set_xlabel("Principal Component 1")
ax.set_ylabel("Principal Component 2")
ax.set_title("Human Activity Data in PCA Space")
ax.legend()
plt.show()

# 3b) Zoomed-in view of the low-motion activities (Sitting/Standing/Walking), which are only
# separable from each other on a much smaller PC1/PC2 range than Running/Dancing
fig2, ax2 = plt.subplots()
for code in range(1, 6):
    idx = actid == code
    ax2.scatter(feat_pca[idx, 0], feat_pca[idx, 1],
                s=5, marker=markers[code - 1], label=actnames[code - 1])

ax2.set_xlim(0, 200)
ax2.set_ylim(-60, 20)
ax2.set_xlabel("Principal Component 1")
ax2.set_ylabel("Principal Component 2")
ax2.set_title("Human Activity Data in PCA Space (zoomed: low-motion activities)")
ax2.legend()
plt.show()

# 4) Pearson's correlation coefficient between raw feature 0 and 1, and between PCA feature 0 and 1
rho_raw, _ = pearsonr(feat[:, 0], feat[:, 1])
rho_pca, _ = pearsonr(feat_pca[:, 0], feat_pca[:, 1])

print(f"Pearson's rho, raw features 0 and 1:  {rho_raw:.4f}")
print(f"Pearson's rho, PCA features 0 and 1:  {rho_pca:.4f}")

# Explanation:
# The raw features can be correlated with each other in arbitrary ways, since they were chosen for
# physical/engineering meaning, not statistical independence -- so rho_raw can be far from 0.
# The PCA feature axes, by construction, are the eigenvectors of the (symmetric) covariance matrix,
# which are mutually orthogonal. Projecting the de-meaned data onto an orthogonal basis decorrelates
# the resulting coordinates, so the covariance (and therefore Pearson's rho) between any two distinct
# PCA dimensions is ~0 (up to numerical roundoff), regardless of what correlations existed in the
# original feature space.

# 5) Number of dimensions needed to capture 99.9% of the variance
N = p.num_effective_dims(99.9)
print(f"Number of PCA dimensions needed for >=99.9% of variance: {N}")
