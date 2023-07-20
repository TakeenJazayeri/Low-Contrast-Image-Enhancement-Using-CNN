import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras
from guided_filter import guided_filter_gray

from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

row, col = 512, 512 

image = cv2.imread('Example/image.JPEG', cv2.IMREAD_COLOR)
image_shape = image.shape
resized_image = cv2.resize(image, (col, row))
model_input = resized_image.reshape(1, row, col, 3)

model = keras.saving.load_model('cnn_model')

output = model(model_input)[0]
result = output[:, :, 0].numpy().astype(np.uint8)
result = cv2.resize(result, (image_shape[1], image_shape[0]))
cv2.imwrite('Example/probability_map.JPEG', result)


prob_map = np.zeros((image_shape[0], image_shape[1]), dtype=float)

for i in range(image_shape[0]):
    for j in range(image_shape[1]):
        prob_map[i][j] = result[i][j] / 255


e = 0.0001

image_enhanced = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
for i in range(image_shape[0]):
    for j in range(image_shape[1]):
        for k in range(3):
            res = int(image[i][j][k] / (1 - prob_map[i][j] + e))
            image_enhanced[i][j][k] = res if res < 255 else 255


cv2.imwrite('Example/image_enhanced_norefining.JPEG', image_enhanced)

# Refining
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

r = 150
eps = 0.001
guided = guided_filter_gray(result, gray_image, r=r, eps=eps, s=None)
cv2.imwrite(f'Example/gd-{r}-{eps}.jpg', guided)



ref_prob_map = np.zeros((image_shape[0], image_shape[1]), dtype=float)

for i in range(image_shape[0]):
    for j in range(image_shape[1]):
        ref_prob_map[i][j] = guided[i][j] / 255


e = 0.0001

image_enhanced_ref = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
for i in range(image_shape[0]):
    for j in range(image_shape[1]):
        for k in range(3):
            res = int(image[i][j][k] / (1 - ref_prob_map[i][j] + e))
            image_enhanced_ref[i][j][k] = res if res < 255 else 255


cv2.imwrite(f'Example/image_enhanced_refined({r},{eps}).JPEG', image_enhanced_ref)



plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.show()

plt.imshow(cv2.cvtColor(image_enhanced, cv2.COLOR_BGR2RGB))
plt.show()

plt.imshow(cv2.cvtColor(image_enhanced_ref, cv2.COLOR_BGR2RGB))
plt.show()