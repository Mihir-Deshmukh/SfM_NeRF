import numpy as np
import cv2
from EstimateFundamentalMatrix import *


def calculate_error(x1, x2, F):

    return np.abs(np.dot(np.dot(x1.T, F), x2)) 

def get_inlier_RANSAC(matches, threshold):
    
    best_n = 0
    best_inliers = set()
    
    for i in range(100):

        # Select 8 random matches
        random_matches = np.random.choice(matches, size=8, replace=False)

        # Estimate the fundamental matrix
        F = estimate_fundamental_matrix(random_matches)

        inliers = set()

        for j in range(len(random_matches)):
            
            points1 = np.array(random_matches[j]['image1_uv'] + (1,))
            points2 = np.array(random_matches[j]['image2_uv'] + (1,))

            error = calculate_error(points1, points2, F)
            
            if error < threshold:
                
                inliers.add(j)
                
        if len(inliers) > best_n:
            
            best_n = len(inliers)
            best_inliers = inliers
            best_matches = random_matches[list(best_inliers)]

    return best_matches