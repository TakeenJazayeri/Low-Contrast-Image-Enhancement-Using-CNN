from run_code import Project
import cv2
import matplotlib.pyplot as plt


proj = Project(image_address='Example', image_file='image.JPEG', model_name='cnn_model')

show_list = [proj.image, proj.enhanced_image, proj.ref_enhanced_image]
for im in show_list:
    if im.shape[2] == 3:
        plt.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
    elif im.shape[2] == 1:
        plt.imshow(cv2.cvtColor(im, cv2.COLOR_GRAY2RGB))
    plt.show()