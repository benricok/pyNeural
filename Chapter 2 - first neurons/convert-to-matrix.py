import numpy as np

a = [1, 2, 3]
b = np.expand_dims(np.array(a), axis=0)
print(b)  # 1D matrix
