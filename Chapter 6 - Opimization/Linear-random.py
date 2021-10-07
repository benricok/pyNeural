import numpy as np
from nnfs.datasets import vertical_data
import nnfs

nnfs.init()


class LayerDense:

    # Layer initialization
    def __init__(self, n_inputs, n_neurons):
        # Initialize weight and biases
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    # Forward pass
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases


class ActivationSoftMAX:

    # Forward pass
    def forward(self, inputs):
        # Get unnormalized propabilities
        expValues = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        # Normalize
        propabilities = expValues / np.sum(expValues, axis=1, keepdims=True)
        self.output = propabilities


class ActivationReLU:

    # Forward pass
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)


class Loss:

    # Calculates the data and regularization losses
    # given model output and ground truth values
    def calculate(self, output, y):
        # Calculate sample losses
        sampleLosses = self.forward(output, y)
        # Calculate mean loss
        dataLoss = np.mean(sampleLosses)
        return dataLoss


class Loss_CategorialCrossentropy(Loss):

    def forward(self, y_pred, y_true):
        # Number of samples in a batch
        samples = len(y_pred)

        # Clip data to prevent division by 0
        # Clip both sides to not drag mean towards any value
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        # probabilities for target values
        # only if categorial labels
        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[
                range(samples),
                y_true
            ]

        # Mask values -  only for one-hot encoded labels
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(
                y_pred_clipped * y_true,
                axis=1
            )

        # Losses
        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods


# Create dataset
X, y = vertical_data(samples=100, classes=3)

# Network model creation
dense1 = LayerDense(2, 3)
activation1 = ActivationReLU()
dense2 = LayerDense(3, 3)
activation2 = ActivationSoftMAX()

# Create loss function
lossFunction = Loss_CategorialCrossentropy()

# Varables
lowestLoss = 9999999  # initial value
best_dense1_weights = dense1.weights.copy()
best_dense1_biases = dense1.biases.copy()
best_dense2_weights = dense2.weights.copy()
best_dense2_biases = dense2.biases.copy()

for iteration in range(10000):
    # Update weights with some small random values
    dense1.weights += 0.05 * np.random.randn(2, 3)
    dense1.biases += 0.05 * np.random.randn(1, 3)
    dense1.weights += 0.05 * np.random.randn(2, 3)
    dense1.biases += 0.05 * np.random.randn(1, 3)

    # Preform pass of our training data through this layer
    dense1.forward(X)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    activation2.forward(dense2.output)

    # Perform a forward pass through activation function
    # it takes the output of second dense layer here and returns loss
    loss = lossFunction.calculate(activation2.output, y)

    # Calculate accuracy from output of activation2 and targets
    # calculate values along fist axis
    predictions = np.argmax(activation2.output, axis=1)
    accuracy = np.mean(predictions == y)

    # If loss is smaller - print and save weights and biases aside
    if loss < lowestLoss:
        print('New set of weights found, iteration:', iteration,
              'loss:', loss, 'acc:', accuracy)
        best_dense1_weights = dense1.weights.copy()
        best_dense1_biases = dense1.biases.copy()
        best_dense2_weights = dense2.weights.copy()
        best_dense2_biases = dense2.biases.copy()
        lowestLoss = loss
    # Revert changes
    else:
        dense1.weights = best_dense1_weights.copy()
        dense1.biases = best_dense1_biases.copy()
        dense2.weights = best_dense2_weights.copy()
        dense2.biases = best_dense2_biases.copy()

# https://nnfs.io/ch6
