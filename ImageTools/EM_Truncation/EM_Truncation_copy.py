### writen by qinhong
### 2021/07/22

import cv2
import numpy as np
import os 

def line_discard(img,start_line_num,line_num):
    end_line_num  = start_line_num + line_num
    img = cv2.imread(img_input_path)
    img1 = np.delete(img, slice(start_line_num,end_line_num), axis=0)
    img2 = np.append(img1, img[-line_num:], axis=0)
    # cv2.imwrite(img_output_path,img2)
    return img2

def em_truncation(img):
    up = 170
    down = 320
    img_input_path = '000928.png'
    img_outputfolder_path = '000928'
    for i in range(up,down):
        for j in range(1,down-i+1,5):
            img_output_path = img_outputfolder_path+'/'+'000928_'+str(i)+'_'+str(j)+'.jpg'
            print(i)
            line_discard(img_input_path, img_output_path, i, j)
    return 
if __name__ == '__main__':
    up = 170
    down = 320
    img_input_path = '/home/usslab/Documents2/qinhong/object/7.30/car_7.30/origin/000928.png'
    img_outputfolder_path = '/home/usslab/Documents2/qinhong/object/7.30/car_7.30/attack/000928'
    for i in range(up,down):
        for j in range(1,down-i+1,5):
            img_output_path = img_outputfolder_path+'/'+'000928_'+str(i)+'_'+str(j)+'.jpg'
            print(i)
            line_discard(img_input_path, img_output_path, i, j)