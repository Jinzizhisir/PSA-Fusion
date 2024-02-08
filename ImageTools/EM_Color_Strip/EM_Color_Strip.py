import PIL.Image as Image
import numpy as np
import os 
from glob import glob
from tqdm import tqdm

def em_color_strip(img):
    # img = Image.open(img_input_path)
    img_array = img.load()
    width, height = img.size
    step = int(height/12)
    img_array_new = [[0]*3 for i in range(height)]
    for x in range(0,width):
        for down in range (0,height-step,2*step):
            up = down + step
            for y in range(down,up):
                img_array_new[y][0] = int(img_array[x,y][1]*2.5)
                img_array_new[y][2] = int(img_array[x,y][1]*2.5)
                img_array_new[y][1] = int((img_array[x,y][0]+img_array[x,y][2])/2)-50
                img_array[x,y]= tuple(img_array_new[y])
    # img.save(img_output_path)
    return img

if __name__ == '__main__':
    image_paths = glob('/home/usslab/SensorFusion/kitti/training/image_3/*.png')
    for image_path in tqdm(image_paths):
        file_path, file_name = os.path.split(image_path)
        image_output_path = "/home/usslab/SensorFusion/kitti_attack/image_3_attack/camera_emi_strip_loss/"+file_name
        img2 = Image.open(image_path)
        bayer_swap(image_path,image_output_path)