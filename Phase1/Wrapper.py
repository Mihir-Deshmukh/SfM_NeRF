# Create Your Own Starter Code :)
#imports
import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt
from EstimateFundamentalMatrix import *
from GetInlierRANSANC import *

# Read images
def read_images(path):
    images = []
    for i in range(5):
        images.append(cv2.imread(f"{path}{i+1}.png"))
        # cv2.imshow(f"Image {i+1}", images[i])
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
    
    return images

def read_intrinsics(path):
    with open(path, 'r') as file:
        lines = file.readlines()
        intrinsics = []
        for line in lines:
            values = line.strip().split()
            row = [float(value) for value in values]
            intrinsics.append(row)
        return np.array(intrinsics)



def main(args):
    
    basepath = args.path
    images = read_images(basepath)
     
    instrinsic_parameters = read_intrinsics(basepath + 'calibration.txt')
    
    image_pairs = [(1,2), (1,3), (1,4), (1,5), (2,3), (2,4), (2,5), (3,4), (3,5), (4,5)]
    
    matched_pairs = []
    
    for pair in image_pairs:
        matched_pairs.append(parse_matching(basepath + f'matching{pair[0]}.txt', pair))
        
    # print(len(matched_pairs[9]))
    
    # Ransac and only send the top 8 feature in the fundamental matrix

    inliers = get_inlier_RANSAC(matched_pairs[0], 0.01)

    # print(inliers)
    
    F = estimate_fundamental_matrix(inliers)

    image1_uv = np.array([match['image1_uv'] + (1,) for match in matched_pairs[0]])
    image2_uv = np.array([match['image2_uv'] + (1,) for match in matched_pairs[0]])

    # print(image1_uv)

    # Normalize the coordinates
    # image1_uv, T1 = normalize_pts(image1_uv)
    # image2_uv, T2 = normalize_pts(image2_uv)
    
    # Estimate the fundamental matrix
    F_, _ = cv2.findFundamentalMat(image1_uv, image2_uv, cv2.FM_RANSAC)
    print(F)
    print("cv2",F_)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", help="Path to the images", default='Phase1/P3Data/')
    args = parser.parse_args()
    main(args)
