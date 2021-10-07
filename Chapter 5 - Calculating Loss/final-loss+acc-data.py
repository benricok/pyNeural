import numpy as np
from nnfs.datasets import spiral_data
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
X, y = spiral_data(samples=100, classes=3)

dense1 = LayerDense(2, 3)
activation1 = ActivationReLU()

dense2 = LayerDense(3, 3)
activation2 = ActivationSoftMAX()

lossFunction = Loss_CategorialCrossentropy()

dense1.forward(X)
activation1.forward(dense1.output)

dense2.forward(activation1.output)
activation2.forward(dense2.output)

print(activation2.output[:5])

loss = lossFunction.calculate(activation2.output, y)

print('Loss: ', loss)

# Calculate accuracy from output of activation and targets
# Calculate values along first axis
predictions = np.argmax(activation2.output, axis=1)

# If targets are one-hot encoded - convert them
if len(y.shape) == 2:
    y = np.argmax(y, axis=1)

# True evaluates to 1; False to 0
accuracy = np.mean(predictions == y)

print('acc: ', accuracy)

# https://nnfs.io/ch5
