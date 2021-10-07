import numpy as np

'''
softmaxOutputs = [[0.7, 0.1, 0.2],
                  [0.1, 0.5, 0.4],
                  [0.02, 0.9, 0.08]]

classTargets = [0, 1, 1]


for targ_idx, distribution in zip(classTargets, softmaxOutputs):
    print(distribution[targ_idx])
'''

softmaxOutputs = np.array([[0.7, 0.1, 0.2],
                           [0.1, 0.5, 0.4],
                           [0.002, 0.9, 0.08]])
classTargets = [0, 1, 1]

# print(softmaxOutputs[[0, 1, 2], classTargets])
print(softmaxOutputs[range(len(softmaxOutputs)), classTargets])

# Loss of distribution
Loss = -np.log(softmaxOutputs[
    range(len(softmaxOutputs)), classTargets
])

print('Loss: ', Loss)

averageLoss = np.mean(Loss)
print('Average loss: ', averageLoss)
