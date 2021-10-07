import numpy as np
import nnfs

nnfs.init()

n_inputs = 2
n_neurons = 4


class LayerDense:
    def __init__(self, n_inputs, n_neurons):
        # Initialize weight and biases
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        pass  # placeholder


layer1 = LayerDense(n_inputs, n_neurons)

print(layer1.weights)
print(layer1.biases)
