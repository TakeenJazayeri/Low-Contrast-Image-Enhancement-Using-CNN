from run_code import Project
import cv2, os
import matplotlib.pyplot as plt


for i in os.listdir('Example'):
    if os.path.isdir(f'Example/{i}'):
        image_file = None
        for file in os.listdir(f'Example/{i}'):
            if file.startswith("image"):
                image_file = file
        
        if image_file == None:
            print('NO IMAGE FILE FOUND')

        proj = Project(image_address=f'Example/{i}', image_file=image_file, model_name='cnn_model')

        show_list = [proj.image, proj.enhanced_image, proj.ref_enhanced_image]
        for im in show_list:
            if im.shape[2] == 3:
                plt.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
            elif im.shape[2] == 1:
                plt.imshow(cv2.cvtColor(im, cv2.COLOR_GRAY2RGB))
            plt.show()