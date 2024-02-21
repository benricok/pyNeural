import os, cv2, platform
import numpy as np 
import matplotlib.pyplot as plt

# Windows
DATASET_W = "C:\\DATASETS\\fashion_mnist_images\\"

# Linux
DATASET_L = "/datasets/fashion_mnist_images/"

if platform.system() == 'Windows':
    DATASET = DATASET_W
elif platform.system() == 'Linux':
    HOME = os.path.expanduser("~")
    DATASET = HOME + DATASET_L

    labels = os.listdir(DATASET + 'train')
    print(labels)

    files = os.listdir(DATASET + 'train/4')
    print(files[:10])
    print(len(files))

    image_data_1 = cv2.imread(DATASET + 'train/4/0011.png', cv2.IMREAD_UNCHANGED)
    image_data_2 = cv2.imread(DATASET + 'train/4/0012.png', cv2.IMREAD_UNCHANGED)

    np.set_printoptions(linewidth=200)
    print(image_data_1)
    print(image_data_2)

    plt.imshow(image_data_1, cmap='gray')
    plt.show()

else:
    raise Exception("OS not supported")

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
