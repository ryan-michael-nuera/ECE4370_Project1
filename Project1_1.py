import json
import numpy as np
import matplotlib.pyplot as plt

with open('DimensionalityReduction.json', 'r') as f:
    dataset = json.load(f)

print(dataset.keys())

data = np.array(dataset['data'])
print(data.shape)

