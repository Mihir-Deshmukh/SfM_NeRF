# Create Your Own Starter Code :)
#imports
import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt
from EstimateFundamentalMatrix import *
from GetInlierRANSANC import *
from EssentialMatrixFromFundamentalMatrix import *
from ExtractCameraPose import *
from LinearTriangulation import *
from DisambiguateCameraPose import *
from NonlinearTriangulation import *
from LinearPnP import *


def drawlines(img1,img2,lines,pts1,pts2):
    ''' img1 - image on which we draw the epilines for the points in img2
        lines - corresponding epilines '''
    r,c = img1.shape
    img1 = cv2.cvtColor(img1,cv2.COLOR_GRAY2BGR)
    img2 = cv2.cvtColor(img2,cv2.COLOR_GRAY2BGR)
    for r,pt1,pt2 in zip(lines,pts1,pts2):
        # print(pt1, pt2)
        color = tuple(np.random.randint(0,255,3).tolist())
        x0,y0 = map(int, [0, -r[2]/r[1] ])
        x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
        img1 = cv2.line(img1, (x0,y0), (x1,y1), color,1)
        img1 = cv2.circle(img1,(int(pt1[0]), int(pt1[1])),5,color, -1)
        img2 = cv2.circle(img2,(int(pt2[0]), int(pt2[1])),5,color, -1)
    return img1,img2

# Read images
def read_images(path):
    images = []
    for i in range(5):
        images.append(cv2.imread(f"{path}{i+1}.png", cv2.IMREAD_GRAYSCALE))
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
    
    
def find_common_features(target_images, matched_pairs_list):
    # Initialize common_features with the first list, avoiding duplicates
    common_features = {}
    for match in matched_pairs_list[0]:
        if match['image1_uv'] not in common_features:
            common_features[match['image1_uv']] = {
                'image1_uv': match['image1_uv'],
                f'image{target_images[1]}_uv': match['image2_uv']
            }

    print(len(common_features))
    # Iterate over each additional list of matched pairs
    for i, matches in enumerate(matched_pairs_list[1:], start=1):
        # Filter current matches to remove duplicates based on image1_uv
        current_matches_filtered = {}
        for match in matches:
            if match['image1_uv'] not in current_matches_filtered:
                current_matches_filtered[match['image1_uv']] = match

        # Use filtered matches for current iteration
        current_features = {}
        for uv, match in current_matches_filtered.items():
            if uv in common_features:
                new_entry = common_features[uv].copy()
                new_entry[f'image{target_images[i+1]}_uv'] = match['image2_uv']
                current_features[uv] = new_entry

        # Update common_features for the next iteration
        common_features = current_features

    # Convert back to list format as specified
    output_features = list(common_features.values())

    return output_features





def main(args):
    
    basepath = args.path
    images = read_images(basepath)
     
    instrinsic_parameters = read_intrinsics(basepath + 'calibration.txt')
    
    image_pairs = [(1,2), (1,3), (1,4), (1,5), (2,3), (2,4), (2,5), (3,4), (3,5), (4,5)]
    
    matched_pairs = []
    
    for pair in image_pairs:
        matched_pairs.append(parse_matching(basepath + f'matching{pair[0]}.txt', pair))
        
    print(len(matched_pairs[0]))
    print(matched_pairs[0][0])
    
    # contains best inliers for all the images
    bestInliers = []
    
    for i in range(10):
        
        inliers_homo = ransac_for_robust_features(matched_pairs[i], images[i], images[i+1])
        
        inliers = get_inlier_RANSAC(inliers_homo, 0.1)
        bestInliers.append(inliers)
        
    
   
    
    # Ransac and only send the top 8 feature in the fundamental matrix
    inliers = get_inlier_RANSAC(matched_pairs[0], 0.1)
    print(len(inliers))
    
    # print(len(matched_pairs[0]))
    
    F = estimate_fundamental_matrix(inliers)

    image1_uv = np.array([match['image1_uv'] + (1,) for match in matched_pairs[0]])
    image2_uv = np.array([match['image2_uv'] + (1,) for match in matched_pairs[0]])
    
    # Estimate the fundamental matrix
    F_, mask = cv2.findFundamentalMat(image1_uv, image2_uv, cv2.FM_RANSAC)
    print(f"Fundamental Matrix(cv2): {F_}")
    print(f"Fundamental Matrix: {F}")
    print(f"Rank of Fundamental Matrix: {np.linalg.matrix_rank(F)}")
    
    pts1 = image1_uv[mask.ravel()==1]
    pts2 = image2_uv[mask.ravel()==1]
    
    
    lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1,1,2), 2,F)
    lines1 = lines1.reshape(-1,3)
    img5,img6 = drawlines(images[0],images[1],lines1,pts1,pts2)
    # Find epilines corresponding to points in left image (first image) and
    # drawing its lines on right image
    lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1,1,2), 1,F)
    lines2 = lines2.reshape(-1,3)
    img3,img4 = drawlines(images[1],images[0],lines2,pts2,pts1)
    plt.subplot(121),plt.imshow(img5)
    plt.subplot(122),plt.imshow(img3)
    plt.show()
    
    
    
    # Get Essential matrix
    E = get_essential_matrix(F_, instrinsic_parameters)
    print(f"Essential Matrix: {E}")
    print(f"Rank of Essential Matrix: {np.linalg.matrix_rank(E)}")
    
    # Get Camera Poses
    camera_poses = get_camera_poses(E)
    # print(camera_poses[1])
    
    # Triangulate the points
    R1 = np.eye(3)
    C1 = np.zeros((3,1))
    Triangulated_points = []
    
    for i in range(4):
        points = triangulate_points(R1, C1, camera_poses[i][0], camera_poses[i][1], inliers, instrinsic_parameters)
        # print(points)
        Triangulated_points.append(points)
    
    Triangulated_points = np.array(Triangulated_points)
    print(Triangulated_points.shape) # (4, n, 3)
    
    for i in range(4):
        plt.axis([-20, 20, -20, 20])
        plt.scatter(Triangulated_points[i,:,0], Triangulated_points[i,:,2], s=4)
        plt.scatter(camera_poses[i][1][0], camera_poses[i][1][2], c='r', s=4)
        
        # plt.plot()
        
    plt.show()
    
    # Disambiguate the camera poses
    camera_pose, correct_worldpoints = disambiguate_camera_pose(camera_poses, Triangulated_points)

    plt.axis([-20, 20, -20, 20])
    plt.scatter(correct_worldpoints[:,0], correct_worldpoints[:,2], s=4)
    plt.scatter(camera_pose[1][0], camera_pose[1][2], s=4, c='r')
    plt.show()
    print(f"Correct Camera Pose: {camera_pose}")
    
    
    pts1 = np.array([match['image1_uv'] + (1,) for match in inliers])
    pts2 = np.array([match['image2_uv'] + (1,) for match in inliers])
    
    
    # Non Linear Triangulation
    reprojected_points = NonlinearTriangulation(instrinsic_parameters, np.eye(3), np.zeros((3,1)), camera_pose[0], camera_pose[1], correct_worldpoints, pts1, pts2)
    
    P1 = get_projectionMatrix(instrinsic_parameters, np.eye(3), np.zeros((3,1)))
    P2 = get_projectionMatrix(instrinsic_parameters, camera_pose[0], camera_pose[1])
    
    error = []
    # # Linear Reprojection error
    for i in range(len(correct_worldpoints)):
        error.append(reprojection_loss(correct_worldpoints[i], P1, P2, pts1[i], pts2[i]))
        
    print(f"Linear Reprojection Error: {np.mean(error)}")
    
    error = []
  
    # Non Linear Reprojection error
    for i in range(len(reprojected_points)):
        error.append(reprojection_loss(reprojected_points[i], P1, P2, pts1[i], pts2[i]))

    print(f"Non-Linear Reprojection Error: {np.mean(error)}")
    
    plt.axis([-20, 20, -20, 20])
    plt.scatter(reprojected_points[:,0], reprojected_points[:,2], s=4)
    plt.scatter(camera_pose[1][0], camera_pose[1][2], c='r', s=4)
    plt.show()
    
    R_All = []
    C_All = []
    R_All.append(np.eye(3))
    C_All.append(np.zeros((3,1)))
    R_All
    
    
    # Linear PnP and Pnp Ransac
    
    common = find_common_features([1,2,3], matched_pairs[0:2])
    

    # for i in range(4):
    #     # pts = 
        
    #     R_All.append(R)
    #     C_All.append(C)
    
    # Non Linear PnP
    
    
    
    
    R_All = []
    C_All = []
    
    #Bundle Adjustment
    
    
    
    
    
    
    
    
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", help="Path to the images", default='Phase1/P3Data/')
    args = parser.parse_args()
    main(args)
