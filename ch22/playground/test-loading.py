import os

MODELS = "C:\\Users\\bkadmin\\Documents\\Datasets"

labels = os.listdir(MODELS + '\\fashion_mnist_images\\train')
print(labels)

files = os.listdir(MODELS + '\\fashion_mnist_images\\train\\0')
print(files[:10])
print(len(files))