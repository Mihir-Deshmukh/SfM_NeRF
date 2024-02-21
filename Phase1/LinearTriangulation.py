import numpy as np

def triangulate_points(R1, C1, R2, C2, matched_points, K):

    # Compute the projection matrices for the two cameras
    Translation1 = - R1 @ C1
    P1 = K @ np.hstack((R1, Translation1)) # 3x4

    Translation2 = - R2 @ C2
    P2 = K @ np.hstack((R2, Translation2))
    # print(P2)
    
    I = np.identity(3)
    P1 = np.dot(K, np.dot(R1, np.hstack((I,-C1))))
    P2 = np.dot(K, np.dot(R2, np.hstack((I,-C2))))
    # print(P2)
    
    p1T = P1[0,:].reshape(1,4)
    p2T = P1[1,:].reshape(1,4)
    p3T = P1[2,:].reshape(1,4)

    p_1T = P2[0,:].reshape(1,4)
    p_2T = P2[1,:].reshape(1,4)
    p_3T = P2[2,:].reshape(1,4)
    
    image1_uv = np.array([match['image1_uv'] + (1,) for match in matched_points])
    image2_uv = np.array([match['image2_uv'] + (1,) for match in matched_points])

    # Convert homogeneous coordinates to 3D points
    world_points = []
    
    for i in range(len(matched_points)):
        
        x = image1_uv[i,0] 
        y = image1_uv[i,1]
        x_ = image2_uv[i,0]
        y_ = image2_uv[i,1]

        A = []
        A.append((y * p3T) - p2T)
        A.append(p1T - (x * p3T))
        A.append((y_ * p_3T) - p_2T)
        A.append(p_1T - (x_ * p_3T))

        A = np.array(A).reshape(4,4)

        _,_,vt = np.linalg.svd(A)
        v = vt.T
        x = v[:,-1]
        world_points.append(x)

    X = np.array(world_points)
    X = X/X[:,3].reshape(-1,1)
    return X