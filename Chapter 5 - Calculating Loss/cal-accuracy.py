import numpy as np

# Probabilities of 3 samples
softmaxOutputs = np.array([[0.7, 0.2, 0.1],
                           [0.5, 0.1, 0.4],
                           [0.02, 0.9, 0.08]])
# Target (ground-truth) labels for 3 samples
classTargets = np.array([0, 1, 1])

# Calculate values along second axis (axis of index 1)
predictions = np.argmax(softmaxOutputs, axis=1)

# If targets are one-hot encoded - convert them
if len(classTargets.shape) == 2:
    classTargets = np.argmax(classTargets, axis=1)

# True evaluates to 1; False to 0
accuracy = np.mean(predictions == classTargets)

print('acc: ', accuracy)
