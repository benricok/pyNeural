import zipfile, os, urllib.request, platform

# Windows
DATASET_W = "C:\\DATASETS\\fashion_mnist_images\\"

# Linux
DATASET_L = "/datasets/fashion_mnist_images"

URL = 'https://nnfs.io/datasets/fashion_mnist_images.zip'
FILE = './fashion_mnist_images.zip'
FOLDER = './fashion_mnist_images'

if platform.system() == 'Windows':
    DATASET = DATASET_W
elif platform.system() == 'Linux':
    HOME = os.path.expanduser("~")
    DATASET = HOME + DATASET_L
else:
    raise Exception("OS not supported")

if not os.path.isfile(FILE):
    print(f'Downloading {URL} and saving as {FILE}...')
    urllib.request.urlretrieve(URL, FILE)
  
if not os.path.exists(DATASET):
    print('Creating dataset folder...') # 'fashion_mnist_images/
    os.makedirs(DATASET)

print('Unzipping into ' + DATASET)
print('Unzipping images...')

try:
    with zipfile.ZipFile(FILE) as zip_images:
        print(zip_images)
        zip_images.extractall(DATASET)
except zipfile.BadZipFile:
    raise Exception('Not a zip file or a corrupted zip file')
    
print('Done!')
