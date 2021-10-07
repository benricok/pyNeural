import math

# Output of neural network
softmaxOutput = [0.7, 0.1, 0.2]

'''
# Ground truth
targetOutput = [1, 0, 0]

loss = -(math.log(softmaxOutput[0])*targetOutput[0] +
         math.log(softmaxOutput[1])*targetOutput[1] +
         math.log(softmaxOutput[2])*targetOutput[2])
'''

# Thus
loss = -math.log(softmaxOutput[0])

print(loss)
