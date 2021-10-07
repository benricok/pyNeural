import numpy as np


class ActivationSoftMAX:
    def forward(self, inputs):
        # Get unnormalized propabilities
        expValues = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        # Normalize
        propabilities = expValues / np.sum(expValues, axis=1, keepdims=True)
        self.output = propabilities


outputs = [1, 2, 3]

softmax = ActivationSoftMAX()

softmax.forward([outputs])
print(softmax.output)

outputs = [[-2, -1, 0]]

softmax.forward(outputs)
print(softmax.output)

# Equal
