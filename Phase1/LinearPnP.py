import numpy as np

def LinearPnP(points, world_points, K):
    # Convert points and world_points to homogeneous coordinates
    points_normalized = points
    
    # Normalize image coordinates using the camera intrinsic matrix K
    # points_normalized = np.linalg.inv(K) @ points.T
    # print(f"points_normalized: {points_normalized.shape}")
    # points_normalized = points_normalized.T

    # Construct the A matrix
    A = []
    for i in range(points.shape[0]):
        # print(f"world_points: {world_points[i]}")
        # print(f"points_normalized: {points_normalized[i]}")
        X, Y, Z, _ = world_points[i]
        u, v, _ = points_normalized[i]
        A.append([X, Y, Z, 1, 0, 0, 0, 0, -u*X, -u*Y, -u*Z, -u])
        A.append([0, 0, 0, 0, X, Y, Z, 1, -v*X, -v*Y, -v*Z, -v])
    A = np.array(A)
    # print(A.shape)

    # Perform singular value decomposition (SVD) on A
    U, S, V = np.linalg.svd(A)

    # Extract the solution from the last column of V
    P = V[-1].reshape(3, 4)
    
    R = P[:, :3]
    U, D, Vt = np.linalg.svd(R)
    R = U.dot(Vt)
    
    C = P[:, 3]/S[0]
    C = - np.linalg.inv(R).dot(C)
    
    if np.linalg.det(R) < 0:
        R = -R
        C = -C
        
    return R, C, P