import numpy as np
import cv2

def get_essential_matrix(F, K):
    
    E = np.dot(np.dot(K.T, F), K)
    return E