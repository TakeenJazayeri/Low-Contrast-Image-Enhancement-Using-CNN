import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras
from guided_filter import guided_filter_gray
from GcsDecolor import GcsDecolor2

from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

 
class Project:
    def __init__ (self, image_address, image_file, model_name):
        self.model = keras.saving.load_model(model_name)
        self.image = cv2.imread(f'{image_address}/{image_file}', cv2.IMREAD_COLOR)
        self.image_shape = self.image.shape

        self.result = self.run_model()
        cv2.imwrite(f'{image_address}/result_map.jpg', self.result)
        
        self.prob_map = self.prob_map_extract()
        
        self.enhanced_image = self.enhance()
        cv2.imwrite(f'{image_address}/image_enhanced_norefining.jpg', self.enhanced_image)

        ## REFINING
        # self.ref_prob_map = refine(result, cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
        self.guided, self.ref_prob_map = self.refine(gray_image=GcsDecolor2(self.image))
        cv2.imwrite(f'{image_address}/guided.jpg', self.guided)
        
        self.ref_enhanced_image = self.refined_enhance()
        cv2.imwrite(f'{image_address}/image_enhanced_refined.jpg', self.ref_enhanced_image)


    def run_model (self, row=512, col=512):
        resized_image = cv2.resize(self.image, (col, row))
        model_input = resized_image.reshape(1, row, col, 3)

        output = self.model(model_input)[0]
        output = output[:, :, 0].numpy().astype(np.uint8)
        return cv2.resize(output, (self.image_shape[1], self.image_shape[0]))


    def prob_map_extract (self):
        prob_map = np.zeros((self.image_shape[0], self.image_shape[1]), dtype=float)
        for i in range(self.image_shape[0]):
            for j in range(self.image_shape[1]):
                prob_map[i][j] = self.result[i][j] / 255
        
        return prob_map


    def enhance (self, e=0.0001):
        enhanced_image = cv2.cvtColor(self.result, cv2.COLOR_GRAY2BGR)
        for i in range(self.image_shape[0]):
            for j in range(self.image_shape[1]):
                for k in range(3):
                    res = int(self.image[i][j][k] / (1 - self.prob_map[i][j] + e)) * 1.25
                    enhanced_image[i][j][k] = res if res < 255 else 255
        
        return enhanced_image


    def refine (self, gray_image, r=150, eps=0.001):
        guided = guided_filter_gray(self.result, gray_image, r=r, eps=eps, s=None)

        ref_prob_map = np.zeros((self.image_shape[0], self.image_shape[1]), dtype=float)
        for i in range(self.image_shape[0]):
            for j in range(self.image_shape[1]):
                ref_prob_map[i][j] = guided[i][j] / 255

        return guided, ref_prob_map


    def refined_enhance (self, e=0.0001, alpha=1.25):
        ref_enhanced_image = cv2.cvtColor(self.result, cv2.COLOR_GRAY2BGR)
        for i in range(self.image_shape[0]):
            for j in range(self.image_shape[1]):
                for k in range(3):
                    res = int(self.image[i][j][k] / (1 - self.ref_prob_map[i][j] + e)) * alpha
                    ref_enhanced_image[i][j][k] = res if res < 255 else 255
        
        return ref_enhanced_image
