import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as R
from scipy.sparse import lil_matrix
import scipy

def projectionMatrix(K,R,C):
    C = np.reshape(C, (3, 1))      
    I = np.identity(3)
    projectionMatrix = np.dot(K, np.dot(R, np.hstack((I, -C))))
    return projectionMatrix

def get_camera_indices(visibility_matrix):
    camera_indices = []
    point_indices = []
    for i, row in enumerate(visibility_matrix):
        for j, visible in enumerate(row):
            if visible:
                camera_indices.append(j)
                point_indices.append(i)
    return np.array(camera_indices), np.array(point_indices)

def bundle_adjustment_sparsity(n_cameras, n_points, visibility_matrix):
    camera_indices, point_indices = get_camera_indices(visibility_matrix)
    print(f"Camera indices: {camera_indices.shape}")
    print(f"Point indices: {point_indices.shape}")
    m = camera_indices.size * 2
    n = n_cameras * 6 + n_points * 3
    A = lil_matrix((m, n), dtype=int)
    # n_observations = np.sum(visibility_matrix)

    i = np.arange(camera_indices.size)
    for s in range(6):
        A[2 * i, camera_indices * 6 + s] = 1
        A[2 * i + 1, camera_indices * 6 + s] = 1

    for s in range(3):
        A[2 * i, n_cameras * 6 + point_indices * 3 + s] = 1
        A[2 * i + 1, n_cameras * 6 + point_indices * 3 + s] = 1
    A = A.tocsc()
    
    # print(f"Sparsity: {A.shape}")
    return A

def reprojection_error(params, all_points, visibility_matrix, num_cameras, num_points, intrinsic_matrix):
    # Extract camera parameters and world points from params
    camera_euler_angles = params[:num_cameras * 3].reshape((num_cameras, 3))
    camera_rotation_matrices = [R.from_euler('zyx', r).as_matrix() for r in camera_euler_angles]
    camera_rotation_matrices = np.array(camera_rotation_matrices)
    # print(f"Camera rotation matrices: {camera_rotation_matrices.shape}")
    camera_translations = params[num_cameras * 3:num_cameras * 6].reshape((num_cameras, 3))
    world_points = params[num_cameras * 6:].reshape((num_points, 3))
    
    # error = []
    E1 = 0
    count = 0
    for point_idx, visibility_row in enumerate(visibility_matrix):
        x0 = np.append(world_points[point_idx], 1)
        # print(visibility_row)
        for image_idx, visible in enumerate(visibility_row):
            if visible:  # Check if the point is visible in the current image
                # Find the observed 2D point in the current image
                if f'image{image_idx+1}_uv' in all_points[point_idx].keys():
                    uv = all_points[point_idx][f'image{image_idx+1}_uv']
                else:
                    print(f"Image {image_idx} not found")
                    continue
                
                # Extract and convert camera parameters (rotation and translation)
                rotation_matrix = camera_rotation_matrices[image_idx]
                translation = camera_translations[image_idx]
                
                # Construct the projection matrix P
                # P = np.hstack((rotation_matrix, -rotation_matrix @ translation.reshape(-1, 1)))
                P = projectionMatrix(intrinsic_matrix, rotation_matrix, translation)
                # Project the world point onto the image plane
                p1_1T, p1_2T, p1_3T = P # rows of P
                p1_1T, p1_2T, p1_3T = p1_1T.reshape(1,-1), p1_2T.reshape(1,-1),p1_3T.reshape(1,-1)
                
                u1,v1 = uv[0], uv[1]
                u1_proj = np.divide(p1_1T.dot(x0) , p1_3T.dot(x0))
                v1_proj =  np.divide(p1_2T.dot(x0) , p1_3T.dot(x0))
    
                E1 += np.square(v1 - v1_proj) + np.square(u1 - u1_proj)
                count += 1
                

    print(f"Error: {E1/count}")
    return E1/count

def bundleAdjustment(all_points, world_points, visibility_matrix, R_All, C_All, num_cameras, intrinsic_matrix):
    num_points = len(world_points)
    
    # Convert R_All, C_All, world_points into a single parameter vector
    # r to euler
    # print(R_All[0])
    
    R_All = [R.from_matrix(r).as_euler('zyx') for r in R_All]
    R_All = np.array(R_All)
    # print(R.from_euler('zyx', R_All[0], degrees=True).as_matrix())
    print(world_points.shape)
    print(R_All.shape)
    print(C_All.shape)
    
    initial_params = np.hstack((R_All.flatten(), C_All.flatten(), world_points.flatten()))
    print(f"Initial params: {initial_params.shape}")
    # Optimize
    A = bundle_adjustment_sparsity(num_cameras, num_points, visibility_matrix)
    print(f"Sparsity: {A.shape}")
    
    # jac_sparsity=A, 
    result = least_squares(fun=reprojection_error, x0=initial_params, ftol=1e-7, verbose=2, x_scale='jac', args=(all_points, visibility_matrix, num_cameras, num_points, intrinsic_matrix))
    
    # Extract optimized parameters
    optimized_params = result.x
    print(f"Optimized params: {optimized_params.shape}")
    optimized_R_All = optimized_params[:num_cameras * 3].reshape((num_cameras, 3))
    # euler to rotation matrix
    optimized_R_All = [R.from_euler('zyx', r, degrees=True).as_matrix() for r in optimized_R_All]
    optimized_C_All = optimized_params[num_cameras * 3:num_cameras * 6].reshape((num_cameras, 3))
    optimized_world_points = optimized_params[num_cameras * 6:].reshape((num_points, 3))
    
    return optimized_R_All, optimized_C_All, optimized_world_points