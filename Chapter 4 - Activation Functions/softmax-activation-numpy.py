import numpy as np

# Output from neural network
layerOutputs = [4.8, 1.21, 2.385]

expValues = np.exp(layerOutputs)

print('Exponentiated values:')
print(expValues)

# Normalize values
normValues = expValues / np.sum(expValues)

print('Normalized exponentiated values:')
print(normValues)

print('Sum: ', np.sum(normValues))
