import numpy as np
import os, cv2
from tensorflow import keras
from transformation import rotate_10_right, rotate_10_left, flip
from random import sample
from unet_model import build_Unet_mdel

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

X, Y = [], []
row, col = 512, 512

with os.scandir('Dataset\\Result') as imagefiles:
    for imagefile in imagefiles:
        image = cv2.imread(f'Dataset\\Main\\{imagefile.name}', cv2.IMREAD_COLOR)
        resized_image = cv2.resize(image, (col, row))
        X.append(resized_image.reshape(row, col, 3))
        X.append(rotate_10_right(resized_image).reshape(row, col, 3))
        X.append(rotate_10_left(resized_image).reshape(row, col, 3))
        X.append(flip(resized_image).reshape(row, col, 3))

        prob_map = cv2.imread(f'Dataset\\Result\\{imagefile.name}', 0)
        resized_prob_map = cv2.resize(prob_map, (col, row))
        Y.append(resized_prob_map.reshape(row, col, 1))
        Y.append(rotate_10_right(resized_prob_map).reshape(row, col, 1))
        Y.append(rotate_10_left(resized_prob_map).reshape(row, col, 1))
        Y.append(flip(resized_prob_map).reshape(row, col, 1))

train_perc = 0.8
selected_index = sample(list(range(1, len(X))), int(len(X)*train_perc))

X_train, X_val, Y_train, Y_val = [], [], [], []
for i in range(len(X)):
    if i in selected_index:
        X_train.append(X[i])
        Y_train.append(Y[i])
    else:
        X_val.append(X[i])
        Y_val.append(Y[i])

X_train = np.array(X_train)
Y_train = np.array(Y_train)
X_val = np.array(X_val)
Y_val = np.array(Y_val)


model = build_Unet_mdel(row, col)

model.compile(optimizer = 'adam', loss = keras.losses.MeanSquaredError())

history = model.fit(
    X_train, Y_train, validation_data=(X_val, Y_val),
    batch_size=5, epochs=1, verbose=1
)

model.save('cnn_model')


# RESULT OF RUNNING:
# Epoch 1/5
# 256/256 [==============================] - 1873s 7s/step - loss: 30829.3438 - val_loss: 23312.3945
# Epoch 2/5
# 256/256 [==============================] - 1856s 7s/step - loss: 10147.9492 - val_loss: 3533.6685
# Epoch 3/5
# 256/256 [==============================] - 1849s 7s/step - loss: 3605.4712 - val_loss: 3158.5308
# Epoch 4/5
# 256/256 [==============================] - 1830s 7s/step - loss: 3569.5601 - val_loss: 4154.7910
# Epoch 5/5
# 256/256 [==============================] - 1733s 7s/step - loss: 3450.5513 - val_loss: 3157.0552
