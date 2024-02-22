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
    U, D, V = np.linalg.svd(A)
    P = V[-1, :].reshape(3, 4)
    temp = np.linalg.inv(K) @ P[:, :3]
    U_dash, D_dash, V_dash = np.linalg.svd(temp)
    R = U_dash @ V_dash
    C = np.linalg.inv(K) @ P[:, 3]/D_dash[0]
    if np.linalg.det(R) < 0:
        R = -R
        C = -C
    # C=C.reshape(3,1)
    C = -R.T@C
    return R, C