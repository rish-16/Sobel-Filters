import math
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

x_operator = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
y_operator = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])

def rgb2gray(rgb):
	r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
	gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

	return gray

img = Image.open("../img/skyline.jpg")
img = np.array(img)
img = img / 255 # normalise image
gray_img = rgb2gray(img)

G_x = []
G_y = []

for i in range(0, len(gray_img), len(x_operator)):
	for j in range(0, len(gray_img), len(x_operator)):
		receptive_field = gray_img[i:i+len(x_operator), j:j+len(x_operator)]
		template = np.zeros(x_operator.shape)
		template[:receptive_field.shape[0],:receptive_field.shape[1]] = receptive_field
		
		x_result = np.sum(np.dot(x_operator, template))
		y_result = np.sum(np.dot(y_operator, template))
		
		G_x.append(x_result)
		G_y.append(y_result)

G_x = np.array(G_x)
G_y = np.array(G_y)

G = np.sqrt(np.square(G_x) + np.square(G_y))
print (G.shape)
G = G.reshape([int(math.sqrt(len(G))), int(math.sqrt(len(G)))])

plt.figure(1)
plt.subplot(131)
plt.imshow(img, cmap="gray")
plt.axis("off")

plt.subplot(132)
plt.imshow(gray_img, cmap="gray")
plt.axis("off")

plt.subplot(133)
plt.imshow(G, cmap="gray")
plt.axis("off")
plt.show()