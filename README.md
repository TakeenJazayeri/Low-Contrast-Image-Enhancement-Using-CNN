# Low Contrast Image Enhancement Using CNN

This repository contains an implementation of a Convolutional Neural Network (CNN) for enhancing low-contrast images. The model is designed to improve the visibility and overall quality of images with low contrast levels, making it particularly useful for various computer vision and image processing applications.

## Article

The code is generally based on (Moon et al., 2019). Although different in some parts, the project tries to get the same results as the article. In addition, some pictures from [Exclusively Dark Image Dataset](https://github.com/cs-chan/Exclusively-Dark-Image-Dataset) are used as the dataset of this project. The database, labeled by human user, can be found on the Dataset directory.

## Architecture

The most significant feature of this method is using U-net architecture of convolutional neural networks to predict the probability of contrast issue in each pixel of the input image. Next, the predicted probability map is refined using guided filter, which could enhance the final result in details. Based on the filter's output, the local contrast issues are redressed in the first image.
