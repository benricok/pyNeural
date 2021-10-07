import math

# Output from neural network
layerOutputs = [4.8, 1.21, 2.385]

# Mathematical constant
e = math.e

expValues = []
for output in layerOutputs:
    expValues.append(e ** output)

print('exponent values')
print(expValues)

# Normalize values
normBase = sum(expValues)
normValues = []
for value in expValues:
    normValues.append(value / normBase)

print('Normalized exponentiated values')
print(normValues)

print('Sum: ', sum(normValues))
