# creating一个目标，并遮盖掉目标背后的点
import numpy as np
import os
# from tqdm import tqdm



def remove_points_in_angle_range(point_cloud, angle_range_azimuth,angle_range_vertical):
   
    polar_angles_azimuth = np.arctan2(point_cloud[:, 1], point_cloud[:, 0])
    polar_angles_vertical = np.arctan2(point_cloud[:, 2], point_cloud[:, 0])
    filtered_indices = np.logical_or(np.logical_or(polar_angles_azimuth < angle_range_azimuth[0], polar_angles_azimuth > angle_range_azimuth[1]), \
        np.logical_or(polar_angles_vertical < angle_range_vertical[0], polar_angles_vertical > angle_range_vertical[1]))
    return point_cloud[filtered_indices]
    


def create_laser_object_injection(point_cloud_benign, creating_object):

    # point_path =  os.path.dirname(os.path.abspath(__file__))+'\\000003.bin'
    # point_cloud_benign = np.fromfile(point_path, dtype=np.float32, count=-1).reshape([-1, 4])
    filtered_point_cloud = point_cloud_benign
    # 创建遮挡形成的阴影
    for j in range(0,creating_object.shape[0]):
        polar_angles_azimuth = np.arctan2(creating_object[j, 1], creating_object[j, 0])
        polar_angles_vertical = np.arctan2(creating_object[j, 2], creating_object[j, 0])
        
        # 定义每个点要hiding的角度范围，
        delta = 0.01
        angle_range_azimuth = (polar_angles_azimuth-delta, polar_angles_azimuth+delta)
        angle_range_vertical = (polar_angles_vertical-delta, polar_angles_vertical+delta)
        # hiding角度范围内的点
        filtered_point_cloud = remove_points_in_angle_range(filtered_point_cloud, angle_range_azimuth, angle_range_vertical)
    # 将结果写入新的bin文件中
        
    point_cloud_merge = np.concatenate((filtered_point_cloud,creating_object),axis=0)
    return point_cloud_merge
    # point_cloud_merge.tofile(os.path.dirname(os.path.abspath(__file__))+'\\000003_create.bin')




