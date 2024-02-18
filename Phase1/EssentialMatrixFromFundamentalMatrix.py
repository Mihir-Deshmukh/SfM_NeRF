import numpy as np
import cv2

def get_essential_matrix(F, K):
    
    E = np.dot(np.dot(K.T, F), K)
    U,S,Vt = np.linalg.svd(E)
    S = [1,1,0]
    
    E = U @ np.diag(S) @ Vt
    return E