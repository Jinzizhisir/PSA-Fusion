import cv2 
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pylab

#matplotlib.use('TkAgg')

def cv_imread(filePath):
    # 核心就是下面这句，一般直接用这句就行，直接把图片转为mat数据
    cv_img=cv2.imdecode(np.fromfile(filePath,dtype=np.uint8),-1)
    # imdecode读取的是rgb，如果后续需要opencv处理的话，需要转换成bgr，转换后图片颜色会变化
    #cv_img=cv2.cvtColor(cv_img,cv2.COLOR_RGB2BGR)
    return cv_img

def visualize_image(image):
    # Visualize the image
    plt.imshow(image)
    plt.axis("off")
    plt.show()
    pylab.show()

os.chdir('/home/usslab/SensorFusion/Dataset')

kernel = np.ones((1, 5), np.uint8)
img = cv_imread('./KITTI/object/training/image_2/000000.png')  
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)  
visualize_image(img)
