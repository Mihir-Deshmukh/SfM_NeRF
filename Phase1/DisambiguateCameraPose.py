import numpy as np

def disambiguate_camera_pose(camera_poses, triangulated_points):
    max_points_in_front = -1
    best_camera_pose = None
    for i, camera_pose in enumerate(camera_poses):
        num_points_in_front = check_cheirality_condition(camera_pose, triangulated_points[i])
        print(f"Pos points in pose {i}: {num_points_in_front}")
        if num_points_in_front > max_points_in_front:
            max_points_in_front = num_points_in_front
            best_camera_pose = camera_pose
            world_points = triangulated_points[i]
            
    return best_camera_pose, world_points

def check_cheirality_condition(camera_pose, triangulated_points):
    R, C = camera_pose
    r3 = R[:, 2]
    r3 = r3.reshape(3,1)
    
    num_points_in_front = 0
    # Number of points in front of the camera
    for i in range(triangulated_points.shape[0]):
        X = triangulated_points[i].reshape(3,1)
        
        condition = r3.T @ (X - C)
        if condition>0 and X[2]> 0:
            num_points_in_front += 1
        
    return num_points_in_front