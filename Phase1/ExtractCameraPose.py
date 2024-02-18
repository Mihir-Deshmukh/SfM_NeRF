import numpy as np

def get_camera_poses(E):
    U, _, Vt = np.linalg.svd(E)
    W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])

    # Four possible camera rotations
    R1 = np.dot(np.dot(U, W), Vt)
    R2 = np.dot(np.dot(U, W), Vt)
    R3 = np.dot(np.dot(U, W.T), Vt)
    R4 = np.dot(np.dot(U, W.T), Vt)

    # Four possible camera positions
    C1 = U[:, 2].reshape(3, 1)
    C2 = -U[:, 2].reshape(3, 1)
    C3 = U[:, 2].reshape(3, 1)
    C4 = -U[:, 2].reshape(3, 1)
    
    # check if determinant of all R is negative
    if np.linalg.det(R1) < 0:
        R1 = -R1
        C1 = -C1
    if np.linalg.det(R2) < 0:
        R2 = -R2
        C2 = -C2
    if np.linalg.det(R3) < 0:
        R3 = -R3
        C3 = -C3
    if np.linalg.det(R4) < 0:
        R4 = -R4
        C4 = -C4

    return [(R1, C1), (R2, C2), (R3, C3), (R4, C4)]