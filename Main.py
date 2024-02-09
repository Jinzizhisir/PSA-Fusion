"""
Author: Zizhi Jin
Contact: zizhi@zju.edu.cn
Copyright (C) 2023-2024, USSLAB 
Version: 2.0
Acknowledge: Ack Chen Yan for the Laser_Color_Strip_Injection; Ack Yushi Cheng for the Ultrasound_Blur; Ack Qinhong Jiang for the EM_Truncation and EM_Color_Strip
"""

# import cv2
import numpy as np
from PIL import Image
import sys
from tqdm import tqdm
import os
sys.path.insert(0, '.\ImageTools')
sys.path.insert(0, '.\PointClcoudTools')

from ImageTools.Hide_Laser_Saturation import Hide_Laser_Saturation
from ImageTools.Create_Light_Projection import Create_Light_Projection
from ImageTools.Laser_Color_Strip_Injection import Laser_Color_Strip_Injection
from ImageTools.EM_Truncation import EM_Truncation
from ImageTools.EM_Color_Strip import EM_Color_Strip
from ImageTools.Ultrasound_Blur import Ultrasound_Blur

from PointCloudTools.Hide_Laser_Point_Erase import Hide_Laser_Point_Erase
from PointCloudTools.Create_Laser_Object_Injection import Create_Laser_Object_Injection
from PointCloudTools.Laser_Arbitrary_Point_Injection import Laser_Arbitrary_Point_Injection
from PointCloudTools.Laser_Background_Noise_Injection import Laser_Background_Noise_Injection
from PointCloudTools.EM_Point_Interference import EM_Point_Interference


def process_images(corruption_name, img_file_pth, Output_file_pth):
    if corruption_name =='Hide_Laser_Saturation':
        laser_attack_pattern_path = '.\ImageTools\Hide_Laser_Saturation\Attack_Real_World.png'
        laser_attack_pattern = Image.open(laser_attack_pattern_path)
        output_path = Output_file_pth +'\\Camera_Corruption\\' + corruption_name
        os.makedirs(output_path, exist_ok=True)

        for file_name in tqdm(os.listdir(img_file_pth), desc=f"Processing {corruption_name}"):
            img_path = img_file_pth+'\\'+file_name
            origin = Image.open(img_path)
            image_corrupted = Hide_Laser_Saturation.hide_laser_saturation(origin,laser_attack_pattern)
            # image_corrupted = Hide_Laser_Saturation.hide_laser_saturation(origin,laser_attack_pattern)
            image_corrupted.save(output_path + '\\' + file_name)

    if corruption_name =='Create_Light_Projection':

        creating_object_path = './ImageTools/Create_Light_Projection/car.png' # the object you want to project
        creating_object = Image.open(creating_object_path)
        output_path = Output_file_pth +'\\Camera_Corruption\\'+ corruption_name
        os.makedirs(output_path, exist_ok=True)

        for file_name in tqdm(os.listdir(img_file_pth), desc=f"Processing {corruption_name}"):
            img_path = img_file_pth+'\\'+file_name
            origin_img = Image.open(img_path)
            image_corrupted = Create_Light_Projection.light_projection(creating_object,origin_img)
            
            image_corrupted.save(output_path + '\\' + file_name)

    if corruption_name =='Laser_Color_Strip_Injection':
        
        output_path = Output_file_pth +'\\Camera_Corruption\\'+ corruption_name
        os.makedirs(output_path, exist_ok=True)

        for file_name in tqdm(os.listdir(img_file_pth), desc=f"Processing {corruption_name}"):
            img_path = img_file_pth+'\\'+file_name
            origin_img = Image.open(img_path)
            image_corrupted = Laser_Color_Strip_Injection.color_strip_injection(origin_img)
            
            image_corrupted.save(output_path + '\\' + file_name)

    if corruption_name =='EM_Truncation':
        
        output_path = Output_file_pth +'\\Camera_Corruption\\'+ corruption_name
        os.makedirs(output_path, exist_ok=True)

        for file_name in tqdm(os.listdir(img_file_pth), desc=f"Processing {corruption_name}"):
            img_path = img_file_pth+'\\'+file_name
            origin_img = Image.open(img_path)
            image_corrupted =EM_Truncation.em_truncation(origin_img)
            
            image_corrupted.save(output_path + '\\' + file_name)

    if corruption_name =='EM_Color_Strip':
        
        output_path = Output_file_pth +'\\Camera_Corruption\\'+ corruption_name
        os.makedirs(output_path, exist_ok=True)

        for file_name in tqdm(os.listdir(img_file_pth), desc=f"Processing {corruption_name}"):
            img_path = img_file_pth+'\\'+file_name
            origin_img = Image.open(img_path)
            image_corrupted =EM_Color_Strip.em_color_strip(origin_img)
            
            image_corrupted.save(output_path + '\\' + file_name)

    if corruption_name =='Ultrasound_Blur':
        
        output_path = Output_file_pth +'\\Camera_Corruption\\'+ corruption_name
        os.makedirs(output_path, exist_ok=True)

        for file_name in tqdm(os.listdir(img_file_pth), desc=f"Processing {corruption_name}"):
            img_path = img_file_pth+'\\'+file_name
            origin_img = Image.open(img_path)
            image_corrupted = Ultrasound_Blur.ultrasound_blur(origin_img)
            
            image_corrupted.save(output_path + '\\' + file_name)

def process_pointcloud(corruption_name, point_cloud_file_pth, Output_file_pth):
    if corruption_name =='Hide_Laser_Point_Erase':

        output_path = Output_file_pth +'\\LiDAR_Corruption\\' + corruption_name
        os.makedirs(output_path, exist_ok=True)

        for file_name in tqdm(os.listdir(point_cloud_file_pth), desc=f"Processing {corruption_name}"):
            pc_path = point_cloud_file_pth+'\\'+file_name
            point_cloud_origin = np.fromfile(pc_path, dtype=np.float32, count=-1).reshape([-1, 4])
            pointcloud_corrupted = Hide_Laser_Point_Erase.hide_laser_point_erase(point_cloud_origin)
            
            pointcloud_corrupted.tofile(output_path + '\\' + file_name)
    
    if corruption_name =='Create_Laser_Object_Injection':
        
        output_path = Output_file_pth +'\\LiDAR_Corruption\\' + corruption_name
        os.makedirs(output_path, exist_ok=True)

        car_file = '.\\PointCloudTools\\Create_Laser_Object_Injection\\car_000010_bmw.bin'
        creating_object = np.fromfile(car_file,dtype=np.float32, count=-1).reshape([-1, 4])

        for file_name in tqdm(os.listdir(point_cloud_file_pth), desc=f"Processing {corruption_name}"):
            pc_path = point_cloud_file_pth+'\\'+file_name
            point_cloud_origin = np.fromfile(pc_path, dtype=np.float32, count=-1).reshape([-1, 4])
            pointcloud_corrupted = Create_Laser_Object_Injection.create_laser_object_injection(point_cloud_origin,creating_object)
            
            pointcloud_corrupted.tofile(output_path + '\\' + file_name)

    if corruption_name =='Laser_Arbitrary_Point_Injection':
        
        output_path = Output_file_pth +'\\LiDAR_Corruption\\' + corruption_name
        os.makedirs(output_path, exist_ok=True)

        object_file = r'.\PointCloudTools\Laser_Arbitrary_Point_Injection\object.bin'

        creating_object = np.fromfile(object_file,dtype=np.float32, count=-1).reshape([-1, 4])

        for file_name in tqdm(os.listdir(point_cloud_file_pth), desc=f"Processing {corruption_name}"):
            pc_path = point_cloud_file_pth+'\\'+file_name
            point_cloud_origin = np.fromfile(pc_path, dtype=np.float32, count=-1).reshape([-1, 4])
            pointcloud_corrupted = Laser_Arbitrary_Point_Injection.laser_arbitrary_point_injection(point_cloud_origin,creating_object)
            
            pointcloud_corrupted.tofile(output_path + '\\' + file_name)

    if corruption_name =='Laser_Background_Noise_Injection':
        
        output_path = Output_file_pth +'\\LiDAR_Corruption\\' + corruption_name
        os.makedirs(output_path, exist_ok=True)

        for file_name in tqdm(os.listdir(point_cloud_file_pth), desc=f"Processing {corruption_name}"):
            pc_path = point_cloud_file_pth+'\\'+file_name
            point_cloud_origin = np.fromfile(pc_path, dtype=np.float32, count=-1).reshape([-1, 4])
            pointcloud_corrupted = Laser_Background_Noise_Injection.laser_background_noise_injection(point_cloud_origin)
            
            pointcloud_corrupted.tofile(output_path + '\\' + file_name)
    
    if corruption_name =='EM_Point_Interference':
        
        output_path = Output_file_pth +'\\LiDAR_Corruption\\' + corruption_name
        os.makedirs(output_path, exist_ok=True)

        for file_name in tqdm(os.listdir(point_cloud_file_pth), desc=f"Processing {corruption_name}"):
            pc_path = point_cloud_file_pth+'\\'+file_name
            point_cloud_origin = np.fromfile(pc_path, dtype=np.float32, count=-1).reshape([-1, 4])
            pointcloud_corrupted = EM_Point_Interference.em_point_interference(point_cloud_origin)
            
            pointcloud_corrupted.tofile(output_path + '\\' + file_name)


# If you aim to generate a corrupted dataset.

# If you want to test the physical sensor attack on a single image.

def main():
    # Corruption Zoo
    # Camera_Corruption = ['Hide_Laser_Saturation','Create_Light_Projection','Laser_Color_Strip_Injection','EM_Truncation','EM_Color_Strip','Ultrasound_Blur']
    # LiDAR_Corruption = ['Hide_Laser_Point_Erase','Create_Laser_Object_Injection','Laser_Arbitrary_Point_Injection','Laser_Background_Noise_Injection','EM_Point_Interference']

    # The corruption you want to generate.
    Camera_Corruption = ['Hide_Laser_Saturation','Create_Light_Projection','Laser_Color_Strip_Injection','EM_Truncation','EM_Color_Strip','Ultrasound_Blur']
    LiDAR_Corruption = ['Hide_Laser_Point_Erase','Create_Laser_Object_Injection','Laser_Arbitrary_Point_Injection','Laser_Background_Noise_Injection','EM_Point_Interference']

    # The file path of original images and point cloud.
    img_file_pth = '.\Original_Dataset_Demo\Image' 
    point_cloud_file_pth = '.\Original_Dataset_Demo\PointCloud'

    Output_file_pth = '.\Output_Demo'

    for corruption_name in Camera_Corruption:
        process_images(corruption_name, img_file_pth, Output_file_pth)
    
    for corruption_name in LiDAR_Corruption:
        process_pointcloud(corruption_name, point_cloud_file_pth, Output_file_pth)

if __name__ == "__main__":
    main()