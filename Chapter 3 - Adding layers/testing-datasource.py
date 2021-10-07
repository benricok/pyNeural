import nnfs
import numpy as np
import matplotlib.pyplot as p
from nnfs.datasets import spiral_data

nnfs.init()

X, y = spiral_data(samples=100, classes=3)

p.scatter(X[:, 0], X[:, 1], c=y, cmap='brg')
p.show()
