from scipy.spatial.transform import Rotation
from scipy.optimize import least_squares
from PnPRANSAC import *



def NonLinearPnP(points, world_points, R, C, K):
    
    q = Rotation.from_matrix(R).as_quat()
    
    optim_params = least_squares(fun=reprojection_loss_pnp, x0=np.hstack([C, q]), method="trf", args=[points, world_points, K])
    params = optim_params.x
    # print(f"Optim params: {optim_params}")
    
    C = params[:3]
    R = Rotation.from_quat(params[3:]).as_matrix()
    
    return R, C


def reprojection_loss_pnp(x0, points, world_points, K):
    
    C = x0[:3]
    q = x0[3:]
    R = Rotation.from_quat(q).as_matrix()
    P = get_projectionMatrix(K, R, C)
    
    error = []
    for i in range(len(points)):
        reprojection_error = reprojection_error_pnp(P, points[i], world_points[i])
        error.append(reprojection_error)
    
    # print(f"Error after NonLinear: {np.mean(error)}")
    return np.mean(error)
    
    