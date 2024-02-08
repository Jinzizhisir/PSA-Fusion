# hiding掉指定锥形范围内的点
import numpy as np
import os
# from tqdm import tqdm

def hide_laser_point_erase(point_cloud):
    angle_range_azimuth = (np.deg2rad(-10), np.deg2rad(10))
    angle_range_vertical = (np.deg2rad(-20), np.deg2rad(20))
    # angle_range = np.radians(angle_range)
    
    polar_angles_azimuth = np.arctan2(point_cloud[:, 1], point_cloud[:, 0])
    polar_angles_vertical = np.arctan2(point_cloud[:, 2], point_cloud[:, 0])
    
    filtered_indices = np.logical_or(np.logical_or(polar_angles_azimuth < angle_range_azimuth[0], polar_angles_azimuth > angle_range_azimuth[1]), \
        np.logical_or(polar_angles_vertical < angle_range_vertical[0], polar_angles_vertical > angle_range_vertical[1]))
    
    return point_cloud[filtered_indices]
    


if __name__ == "__main__":
    point_path = os.path.dirname(os.path.abspath(__file__))+'\\000003.bin'
    point_cloud = np.fromfile(point_path, dtype=np.float32, count=-1).reshape([-1, 4])
        # hiding角度范围内的点
    filtered_point_cloud = remove_points_in_angle_range(point_cloud, angle_range_azimuth, angle_range_vertical)
    filtered_point_cloud.tofile(os.path.dirname(os.path.abspath(__file__))+'\\000003test.bin')
