# 随机生成一个目标并注入，并遮盖掉目标背后的点
import numpy as np
import os
# from tqdm import tqdm


def remove_points_in_angle_range(point_cloud, angle_range_azimuth,angle_range_vertical):
   
    polar_angles_azimuth = np.arctan2(point_cloud[:, 1], point_cloud[:, 0])
    polar_angles_vertical = np.arctan2(point_cloud[:, 2], point_cloud[:, 0])
    
    filtered_indices = np.logical_or(np.logical_or(polar_angles_azimuth < angle_range_azimuth[0], polar_angles_azimuth > angle_range_azimuth[1]), \
        np.logical_or(polar_angles_vertical < angle_range_vertical[0], polar_angles_vertical > angle_range_vertical[1]))
    return point_cloud[filtered_indices]
    

def convert_to_polar_coordinates(point_cloud):
    x = point_cloud[:, 0]
    y = point_cloud[:, 1]
    z = point_cloud[:, 2]
    intensity = point_cloud[:,3]

    distance = np.sqrt(x**2 + y**2 + z**2)  
    azimuth = np.arctan2(y, x)  
    elevation = np.arcsin(z / distance)  

    polar_coordinates = np.column_stack((distance, azimuth, elevation,intensity))
    return polar_coordinates

def add_random_noise(polar_coordinates):
    noise = np.random.uniform(-3, 3, size=polar_coordinates.shape[0])
    polar_coordinates[:, 0] += noise
    return polar_coordinates

def convert_to_cartesian_coordinates(creating_object):
    distance = creating_object[:, 0]
    azimuth = creating_object[:, 1]
    elevation = creating_object[:, 2]

    x = distance * np.cos(azimuth) * np.cos(elevation)
    y = distance * np.sin(azimuth) * np.cos(elevation)
    z = distance * np.sin(elevation)
    intensity = creating_object[:,3]

    cartesian_coordinates = np.column_stack((x, y, z, intensity))
    return cartesian_coordinates



# object_path = os.path.dirname(os.path.abspath(__file__))+'\\car_000010_bmw.bin'
# creating_object = np.fromfile(object_path,dtype=np.float32, count=-1).reshape([-1, 4])

def laser_arbitrary_point_injection(point_cloud_benign,creating_object):
    
    arbitrary_object = convert_to_cartesian_coordinates(add_random_noise(convert_to_polar_coordinates(creating_object)))

    # point_path =  os.path.dirname(os.path.abspath(__file__))+'\\000003.bin'
    # point_cloud_benign = np.fromfile(point_path, dtype=np.float32, count=-1).reshape([-1, 4])
    filtered_point_cloud = point_cloud_benign
    
    for j in range(0,arbitrary_object.shape[0]):
        polar_angles_azimuth = np.arctan2(arbitrary_object[j, 1], arbitrary_object[j, 0])
        polar_angles_vertical = np.arctan2(arbitrary_object[j, 2], arbitrary_object[j, 0])
        
        delta = 0.01
        angle_range_azimuth = (polar_angles_azimuth-delta, polar_angles_azimuth+delta)
        angle_range_vertical = (polar_angles_vertical-delta, polar_angles_vertical+delta)

        filtered_point_cloud = remove_points_in_angle_range(filtered_point_cloud, angle_range_azimuth, angle_range_vertical)
    
    point_cloud_merge = np.concatenate((filtered_point_cloud,arbitrary_object),axis=0)
    return point_cloud_merge
    # point_cloud_merge.tofile(os.path.dirname(os.path.abspath(__file__))+'\\000003_arbitrary.bin')

