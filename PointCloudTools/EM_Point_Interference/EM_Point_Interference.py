
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
    intensity = point_cloud[:, 3]

    distance = np.sqrt(x**2 + y**2 + z**2)  # 距离
    azimuth = np.arctan2(y, x)  # 方位角
    elevation = np.arcsin(z / distance)  # 仰角

    polar_coordinates = np.column_stack((distance, azimuth, elevation,intensity))
    return polar_coordinates

def add_random_noise(polar_coordinates,Noise):
    noise = np.random.uniform(-Noise, Noise, size=polar_coordinates.shape[0])
    polar_coordinates[:, 0] += noise
    return polar_coordinates

def convert_to_cartesian_coordinates(polar_coordinates):
    distance = polar_coordinates[:, 0]
    azimuth = polar_coordinates[:, 1]
    elevation = polar_coordinates[:, 2]

    x = distance * np.cos(azimuth) * np.cos(elevation)
    y = distance * np.sin(azimuth) * np.cos(elevation)
    z = distance * np.sin(elevation)
    intensity = polar_coordinates[:,3]

    cartesian_coordinates = np.column_stack((x, y, z, intensity))
    return cartesian_coordinates




def em_point_interference(point_cloud_benign):
    Noise = 0.1 #set noise level, meters
    # point_path = os.path.dirname(os.path.abspath(__file__))+'\\000003.bin'
    # point_cloud_benign = np.fromfile(point_path, dtype=np.float32, count=-1).reshape([-1, 4])
    point_cloud_noise = convert_to_cartesian_coordinates(add_random_noise(convert_to_polar_coordinates(point_cloud_benign),Noise))
    return point_cloud_noise

    # point_cloud_noise.tofile(os.path.dirname(os.path.abspath(__file__))+'./000003_Noise.bin')

