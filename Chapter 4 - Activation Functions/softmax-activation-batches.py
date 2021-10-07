import numpy as np


class ActivationSoftMAX:
    def __init__(self, inputs):
        # Get unnormalized propabilities
        expValues = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        # Normalize
        propabilities = expValues / np.sum(expValues, axis=1, keepdims=True)
        self.output = propabilities
