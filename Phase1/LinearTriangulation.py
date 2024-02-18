import numpy as np

def make_skew(x):
    return np.array([[0, -x[2], x[1]], [x[2], 0, -x[0]], [-x[1], x[0], 0]])

def triangulate_points(R1, C1, R2, C2, matched_points, K):

    # Compute the projection matrices for the two cameras
    Translation1 = - R1 @ C1
    P1 = K @ np.hstack((R1, Translation1)) # 3x4

    Translation2 = - R2 @ C2
    P2 = K @ np.hstack((R2, Translation2))
    
    image1_uv = np.array([match['image1_uv'] + (1,) for match in matched_points])
    image2_uv = np.array([match['image2_uv'] + (1,) for match in matched_points])

    # Convert homogeneous coordinates to 3D points
    world_points = []
    
    for i in range(matched_points.shape[0]):
        
        A = np.vstack((make_skew(image1_uv[i]) @ P1, make_skew(image2_uv[i]) @ P2))
        # print(A.shape)
        
        U, S, Vt = np.linalg.svd(A)
        V = Vt.T
        
        print(V)
        
        world_point = V[:,-1] / V[-1,-1]
        # print(world_point)
        
        world_points.append(world_point[:3])
        
    return np.array(world_points)