import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as R

def reprojection_error(params, all_points, visibility_matrix, num_cameras, num_points):
    # Extract camera parameters and world points from params
    camera_params = params[:num_cameras * 6].reshape((num_cameras, 6))
    world_points = params[num_cameras * 6:].reshape((num_points, 3))
    
    error = []
    for point_idx, visibility_row in enumerate(visibility_matrix):
        x0 = np.append(world_points[point_idx], 1)  # Convert world point to homogeneous coordinates
        
        for image_idx, visible in enumerate(visibility_row):
            if visible:  # Check if the point is visible in the current image
                # Find the observed 2D point in the current image
                if f'image{image_idx}_uv' in all_points[point_idx]:
                    uv = all_points[point_idx][f'image{image_idx}_uv']
                else:
                    continue  # Skip if no observation in this image for some reason
                
                # Extract and convert camera parameters (rotation and translation)
                euler_angles = camera_params[image_idx, :3]
                translation = camera_params[image_idx, 3:]
                rotation_matrix = R.from_euler('xyz', euler_angles).as_matrix()
                
                # Construct the projection matrix P
                P = np.hstack((rotation_matrix, -rotation_matrix @ translation.reshape(-1, 1)))
                
                # Project the world point onto the image plane
                u1_proj = np.dot(P[0], x0) / np.dot(P[2], x0)
                v1_proj = np.dot(P[1], x0) / np.dot(P[2], x0)
                
                # Compute reprojection error for the current point in the current image
                u1, v1 = uv  # Observed 2D point
                error.extend([u1 - u1_proj, v1 - v1_proj])  # Append differences in both u and v directions

    return np.array(error)

def bundleAdjustment(all_points, world_points, visibility_matrix, R_All, C_All):
    num_cameras = len(R_All)
    num_points = len(world_points)
    
    # Convert R_All, C_All, world_points into a single parameter vector
    initial_params = np.hstack((R_All.flatten(), C_All.flatten(), world_points.flatten()))
    
    # Optimize
    result = least_squares(reprojection_error, initial_params, args=(all_points, visibility_matrix, num_cameras, num_points))
    
    # Extract optimized parameters
    optimized_params = result.x
    optimized_R_All = optimized_params[:num_cameras * 9].reshape((num_cameras, 3, 3))
    optimized_C_All = optimized_params[num_cameras * 9:num_cameras * 12].reshape((num_cameras, 3))
    optimized_world_points = optimized_params[num_cameras * 12:].reshape((num_points, 3))
    
    return optimized_R_All, optimized_C_All, optimized_world_points