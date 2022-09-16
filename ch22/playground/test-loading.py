import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

MODELS = "C:\\Users\\bkadmin\\Documents\\Datasets"

labels = os.listdir(MODELS + '\\fashion_mnist_images\\train')
print(labels)

files = os.listdir(MODELS + '\\fashion_mnist_images\\train\\4')
print(files[:10])
print(len(files))

image_data = cv2.imread(MODELS + '\\fashion_mnist_images\\train\\4\\0011.png',
cv2.IMREAD_UNCHANGED)

np.set_printoptions(linewidth=200)
print(image_data)

plt.imshow(image_data, cmap='gray')
plt.show()