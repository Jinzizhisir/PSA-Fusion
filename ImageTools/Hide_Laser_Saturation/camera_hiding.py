#scripts for camera_hiding

import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from tqdm import tqdm

#sys.path.append()

## 读取图像，解决imread不能读取中文路径的问题
def cv_imread(filePath):
    # 核心就是下面这句，一般直接用这句就行，直接把图片转为mat数据
    cv_img=cv2.imdecode(np.fromfile(filePath,dtype=np.uint8),-1)
    # imdecode读取的是rgb，如果后续需要opencv处理的话，需要转换成bgr，转换后图片颜色会变化
    #cv_img=cv2.cvtColor(cv_img,cv2.COLOR_RGB2BGR)
    return cv_img

def adjust_exposure(image_path, exposure_value):
    # Load the image
    image = cv_imread(image_path)
    plt.imshow(image)
    if image is None:
        print("Failed to read image: {}".format(image_path))
        return None
    # Convert the image to the float type
    image = image.astype(np.float32)

    # Adjust the exposure value
    image = exposure_value * image

    # Clip the pixel values that exceed the maximum value of 255
    image = np.clip(image, 0, 255)

    # Convert the image back to the uint8 type
    image = image.astype(np.uint8)

    return image

def visualize_image(image):
    # Visualize the image
    plt.imshow(image)
    plt.axis("off")
    plt.show()

# Example usage
exposure_value = 20
# current_dir = os.getcwd()
# print(current_dir)
os.chdir('/home/usslab/SensorFusion/Dataset')
for i in tqdm(range(1000,7481)):
    img_path = './KITTI/object/training/image_2/'+str(i).zfill(6)+'.png'
    image = adjust_exposure(img_path, exposure_value)
    visualize_image(image)
    cv2.imwrite("./kitti_attack/camera_hiding_totally/"+str(i).zfill(6)+'.png',image)