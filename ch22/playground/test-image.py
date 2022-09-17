import cv2
import matplotlib.pyplot as plt

image_data = cv2.imread('.\\ch22\\playground\\tshirt.png', cv2.IMREAD_GRAYSCALE)
image_data = cv2.resize(image_data, (28, 28))

plt.imshow(image_data, cmap='gray')
plt.show()
