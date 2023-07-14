import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def _rotate (image, deg):
    row, col = 512, 512 
    row_, col_ = 600, 600

    resized_image = cv2.resize(image, (col_, row_))

    M1 = cv2.getRotationMatrix2D(((col_-1)/2.0, (row_-1)/2.0), deg, 1)
    dst = cv2.warpAffine(resized_image, M1, (col_,row_))

    M2 = np.float32([[1,0,-45], [0,1,-45]])
    return cv2.warpAffine(dst, M2, (col, row))

def rotate_10_right (image):
    return _rotate(image, -10)

def rotate_10_left (image):
    return _rotate(image, 10)


def flip (image):
    row, col = 512, 512 
    pts1 = np.float32([[0,0],[0,511],[511,0]])
    pts2 = np.float32([[511,0],[511,511],[0,0]])

    M = cv2.getAffineTransform(pts1,pts2)
    return cv2.warpAffine(image, M, (col,row))
