import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

DATASETS = "C:\\DATASETS\\"
DATASET = "C:\\DATASETS\\" + "fashion_mnist_images\\"

labels = os.listdir(DATASET + 'train')
print(labels)

files = os.listdir(DATASET + 'train\\4')
print(files[:10])
print(len(files))

image_data = cv2.imread(DATASET + 'train\\4\\0011.png', cv2.IMREAD_UNCHANGED)

np.set_printoptions(linewidth=200)
print(image_data)

plt.imshow(image_data, cmap='gray')
plt.show()

## Create lists for samples and labels
#X = []
#y = []
## For each label folder
#for label in labels:
#    # And for each image in given folder
#    for file in os.listdir(os.path.join(
#            DATASETS + 'fashion_mnist_images', 'train', label
#    )):
#        # Read the image
#        image = cv2.imread(os.path.join(
#                DATASETS + 'fashion_mnist_images\\train', label, file
#            ), cv2.IMREAD_UNCHANGED)
#        # And append it and a label to the lists
#        X.append(image)
#        y.append(label)