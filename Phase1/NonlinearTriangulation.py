import scipy
import numpy as np

def get_projectionMatrix(K,R,C):
    C = np.reshape(C, (3, 1))        
    I = np.identity(3)
    projectionMatrix = np.dot(K, np.dot(R, np.hstack((I, -C))))
    return projectionMatrix

def NonlinearTriangulation(K, R1, C1, R2, C2, world_points, pts1, pts2):
    
    P1 = get_projectionMatrix(K,R1,C1)
    P2 = get_projectionMatrix(K,R2,C2)

    reprojected_worldpoints = []
    for i in range(len(world_points)):
        optimized_params = scipy.optimize.least_squares(fun=reprojection_loss, x0=world_points[i], method="trf", args=[P1, P2, pts1[i], pts2[i]])
        X1 = optimized_params.x
        reprojected_worldpoints.append(X1)
    reprojected_worldpoints = np.array(reprojected_worldpoints)
    reprojected_worldpoints = reprojected_worldpoints / reprojected_worldpoints[:,3].reshape(-1,1)
    return reprojected_worldpoints


def reprojection_loss(x0, P1, P2, pts1, pts2):
    p1_1T, p1_2T, p1_3T = P1
    p1_1T, p1_2T, p1_3T = p1_1T.reshape(1,-1), p1_2T.reshape(1,-1),p1_3T.reshape(1,-1)

    p2_1T, p2_2T, p2_3T = P2
    p2_1T, p2_2T, p2_3T = p2_1T.reshape(1,-1), p2_2T.reshape(1,-1), p2_3T.reshape(1,-1)

    ## reprojection error for reference camera points
    u1,v1 = pts1[0], pts1[1]
    u1_proj = np.divide(p1_1T.dot(x0) , p1_3T.dot(x0))
    v1_proj =  np.divide(p1_2T.dot(x0) , p1_3T.dot(x0))
    E1= np.square(v1 - v1_proj) + np.square(u1 - u1_proj)
    
    # print(f"u1: {u1}, u1_proj: {u1_proj}, v1: {v1}, v1_proj: {v1_proj}, E1: {E1}")

    
    ## reprojection error for second camera points - j = 2
    u2,v2 = pts2[0], pts2[1]
    u2_proj = np.divide(p2_1T.dot(x0) , p2_3T.dot(x0))
    v2_proj =  np.divide(p2_2T.dot(x0) , p2_3T.dot(x0))    
    E2= np.square(v2 - v2_proj) + np.square(u2 - u2_proj)
    
    # print(f"u2: {u2}, u2_proj: {u2_proj}, v2: {v2}, v2_proj: {v2_proj}, E2: {E2}")
    
    error = E1 + E2
    # print(error)
    return error