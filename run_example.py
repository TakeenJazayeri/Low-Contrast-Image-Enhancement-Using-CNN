import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

row, col = 512, 512 

image = cv2.imread('Example/image.jpg', cv2.IMREAD_COLOR)
image_shape = image.shape
resized_image = cv2.resize(image, (col, row))
model_input = resized_image.reshape(1, row, col, 3)

model = keras.saving.load_model('cnn_model')
output = model(model_input)[0]

result = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
for i in range(row):
    for j in range(col):
        result[i][j] = output[i][j][0]
    if i%25 == 0:
        print(i)
result = cv2.resize(result, (image_shape[1], image_shape[0]))
cv2.imwrite('Example/probability_map.jpg', result)


prob_map = np.zeros((image_shape[0], image_shape[1]), dtype=float)

for i in range(image_shape[0]):
    for j in range(image_shape[1]):
        prob_map[i][j] = result[i][j] / 255


eps = 0.0001

image_enhanced = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
for i in range(image_shape[0]):
    for j in range(image_shape[1]):
        for k in range(3):
            r = int(image[i][j][k] / (1 - prob_map[i][j] + eps))
            if r < 255:
                image_enhanced[i][j][k] = r
            else:
                image_enhanced[i][j][k] = 255


cv2.imwrite('Example/image_enhanced.jpg', image_enhanced)


plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.show()
plt.imshow(cv2.cvtColor(image_enhanced, cv2.COLOR_BGR2RGB))
plt.show()