from nnfs.datasets import spiral_data
import numpy as np
import nnfs

nnfs.init()

# Create dataset
X, y = spiral_data(samples=100, classes=3)


class LayerDense:
    def __init__(self, n_inputs, n_neurons):
        # Initialize weight and biases
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases


dense1 = LayerDense(2, 3)

dense1.forward(X)

print(dense1.output[:5])

# https://nnfs.io/ch3
