import numpy as np
from scipy.spatial.transform import Rotation
from LinearPnP import *
import cv2



def get_projectionMatrix(K,R,C):
    C = np.reshape(C, (3, 1))        
    I = np.identity(3)
    projectionMatrix = np.dot(K, np.dot(R, np.hstack((I, -C))))
    return projectionMatrix

def reprojection_error_pnp(P, point, world_point):
    
    x0 = world_point
    p1_1T, p1_2T, p1_3T = P # rows of P
    p1_1T, p1_2T, p1_3T = p1_1T.reshape(1,-1), p1_2T.reshape(1,-1),p1_3T.reshape(1,-1)
    
    u1,v1 = point[0], point[1]
    u1_proj = np.divide(p1_1T.dot(x0) , p1_3T.dot(x0))
    v1_proj =  np.divide(p1_2T.dot(x0) , p1_3T.dot(x0))
    
    E1 = np.square(v1 - v1_proj) + np.square(u1 - u1_proj)
    
    return E1

def PnPRANSAC(points, world_points, K, no_of_iterations=1000):
    best_inliers = []
    best_pose = None
    threshold = 10
    best_error = float('inf')
    
    points = np.array(points)
    world_points = np.array(world_points)
    
    for _ in range(no_of_iterations):
        
        # Randomly select 6 points
        indices = np.random.choice(len(points), 6, replace=False)
        # print(f"indices: {indices}")
        sampled_points = points[indices]
        sampled_world_points = world_points[indices]
        
        # print(f"sampled_points: {sampled_points}")
        # print(f"sampled_world_points: {sampled_world_points}")
        

        # Estimate pose using Linear PnP
        R, C, P = LinearPnP(sampled_points, sampled_world_points, K)
        
        P_ = get_projectionMatrix(K, R, C)

        # Calculate reprojection error
        inliers = []
        error = []
        for i in range(len(points)):
            reprojection_error = reprojection_error_pnp(P, points[i], world_points[i])
            # print(f"reprojection_error: {reprojection_error}")
            error.append(reprojection_error)
            
            if reprojection_error < threshold:
                inliers.append(i)

        # Update best pose and inliers
        if len(inliers) > len(best_inliers):
            best_inliers = inliers
            best_pose = (R, C)
            best_error = np.mean(error)
            
    print(f"LinearPnP Ransac Error: {best_error}")
    return best_pose, best_inliers


    