### writen by qinhong
### 2021/07/22

from PIL import Image
import numpy as np
from glob import glob
from tqdm import tqdm
import os
# import cv2

def em_truncation(img):
    # img = cv2.imread(img_input_path)
    start_line_num = int(np.shape(img)[0]/3)
    line_num = int(np.shape(img)[0]/3)
    end_line_num  = start_line_num + line_num

    img_array = np.array(img)
    img1 = np.delete(img_array, slice(start_line_num,end_line_num), axis=0)
    img2 = np.append(img1, img_array[-line_num:], axis=0)
    # cv2.imwrite(img_output_path,img2)
    img_corrupted = Image.fromarray(img2)
    return img_corrupted

if __name__ == '__main__':
    image_paths = glob('')
    for image_path in tqdm(image_paths):
        file_path, file_name = os.path.split(image_path)
        image_output_path = ""+file_name
        line_discard(image_path, image_output_path)

