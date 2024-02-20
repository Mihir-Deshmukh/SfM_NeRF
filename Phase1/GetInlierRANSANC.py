import numpy as np
import cv2
import random
from EstimateFundamentalMatrix import *

np.random.seed(300)


def calculate_error(x1, x2, F):

    return np.abs(np.dot(np.dot(x1.T, F), x2)) 

def get_inlier_RANSAC(matches, threshold):
    
    best_n = 0
    best_inliers = set()
    
    for i in range(100):

        # Select 8 random matches
        # random_matches = np.random.choice(matches, size=8, replace=False)

        random_matches_1 = random.sample(matches[0], 8)
        random_matches_2 = random.sample(matches[1], 8) 

        # Estimate the fundamental matrix
        F = estimate_fundamental_matrix(random_matches_1, random_matches_2)

        inliers = set()

        for j in range(len(random_matches_1)):
            
            #points1 = np.array(matches[j]['image1_uv'] + (1,))
            #points2 = np.array(matches[j]['image2_uv'] + (1,))

            points1 = np.array([random_matches_1[j][0], random_matches_1[j][1], 1])  
            points2 = np.array([random_matches_2[j][0], random_matches_1[j][1], 1])

            error = calculate_error(points1, points2, F)

            if error < threshold:
                
                inliers.add(j)
                # print(error)
                
        if len(inliers) > best_n:
            
            best_n = len(inliers)
            best_inliers = inliers
            best_matches1 = np.array(random_matches_1)[list(best_inliers)]
            best_matches2 = np.array(random_matches_2)[list(best_inliers)]

            #print(best_matches1)
            #print(best_matches2)

    return (best_matches1, best_matches2)

