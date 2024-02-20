# Create Your Own Starter Code :)
#imports
import argparse
import cv2
import numpy as np
import random
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
from EstimateFundamentalMatrix import *
from GetInlierRANSANC import *
from EssentialMatrixFromFundamentalMatrix import *
from ExtractCameraPose import *
from LinearTriangulation import *
from DisambiguateCameraPose import *
from NonlinearTriangulation import *
from LinearPnP import *

np.random.seed(50)

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
    
    output_indices = []
    
    # Directly return the matches in the specified format if only one pair is provided
    if len(target_images) == 2 and len(matched_pairs_list) == 1:
        output_features = [{
            'image1_uv': match['image1_uv'],
            f'image{target_images[1]}_uv': match['image2_uv']
        } for match in matched_pairs_list[0]]
        
        output_indices = list(range(len(matched_pairs_list[0])))
        return output_features, output_indices

    # Initialize a dictionary to store the initial set of common points
    # using image1_uv from the first pair as the unique key
    common_features = {}
    feature_indices = {}
    for index, match in enumerate(matched_pairs_list[0]):
        common_features[match['image1_uv']] = {'image1_uv': match['image1_uv'], f'image{target_images[1]}_uv': match['image2_uv']}
        feature_indices[match['image1_uv']] = [index]  # Initialize list with the index of the match in the first list

    # Iterate over each additional list of matched pairs
    for i, matches in enumerate(matched_pairs_list[1:], start=1):
        current_features = {}
        for j, match in enumerate(matches):
            if match['image1_uv'] in common_features:
                new_entry = common_features[match['image1_uv']].copy()
                new_entry[f'image{target_images[i+1]}_uv'] = match['image2_uv']
                current_features[match['image1_uv']] = new_entry
                # Append the index of the match in the current list to the tracking list
                feature_indices[match['image1_uv']].append(j)
        common_features = current_features

    # Convert the common features dictionary back to a list format as specified
    output_features = list(common_features.values())

    # For indices, flatten the list of indices for common features
    output_indices = [indices for key, indices in feature_indices.items() if key in common_features]

    return output_features, output_indices



def ransac_for_robust_features(matched_pairs, image1, image2):

    src_pts = np.float32([m['image1_uv'] for m in matched_pairs])
    dst_pts = np.float32([m['image2_uv'] for m in matched_pairs])

    max_inliers_count = 0
    best_H = None
    best_inliers = []

    for _ in range(1):

        random_matches = np.random.choice(matched_pairs, 4)
        src_pts = np.float32([m['image1_uv'] for m in random_matches])
        dst_pts = np.float32([m['image2_uv'] for m in random_matches])

        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC)

        inliers = []

        for i in range(len(matched_pairs)):
            point1_homogeneous = np.append(matched_pairs[i]['image1_uv'], 1) # Convert to homogeneous coordinates
            projected_point = np.dot(H, point1_homogeneous) 

            projected_point = projected_point[:2] / projected_point[2]

            # Calculate the distance between the projected point and the actual destination point
            point2 = np.array(matched_pairs[i]['image2_uv'])
            distance = np.linalg.norm(projected_point - point2)

            # If the distance is below a threshold, count it as an inlier
            if distance < 50: # Threshold of 5 pixels
                inliers.append(matched_pairs[i])

        # Update the best homography if this iteration's inliers are the maximum found so far
        if len(inliers) > max_inliers_count:
            max_inliers_count = len(inliers)
            best_H = H
            best_inliers = inliers

    return best_inliers

def main(args):
    
    print("Started SFM")
    basepath = args.path
    images = read_images(basepath)
     
    instrinsic_parameters = read_intrinsics(basepath + 'calibration.txt')
    
    image_pairs = [(1,2), (1,3), (1,4), (1,5), (2,3), (2,4), (2,5), (3,4), (3,5), (4,5)]
    
    matched_pairs = []
    
    for pair in image_pairs:
        matched_pairs.append(parse_matching(basepath + f'matching{pair[0]}.txt', pair))
        
    # Will contain best inliers for all the images
    bestInliers = []
    
    # Run RANSAC for each image pair
    for i in range(10):

        # image_idx1 = image_pairs[i][0] - 1
        # image_idx2 = image_pairs[i][1] - 1
        # print("all", len(matched_pairs[i]))
        # inliers_homo = ransac_for_robust_features(matched_pairs[i], images[image_idx1], images[image_idx2])

        inliers = get_inlier_RANSAC(matched_pairs[i], 0.1)
        bestInliers.append(inliers)
        
        print(f"Unique matches in Image pair {image_pairs[i]} after RANSAC:", len(inliers))
    
    
    
    # Fundamental matrix between Image 1 and Image 2    
    print("\n----------------------------Finding Fundamental Matrix--------------------------------")
    F = estimate_fundamental_matrix(bestInliers[0])

    image1_uv = np.array([match['image1_uv'] + (1,) for match in bestInliers[0]])
    image2_uv = np.array([match['image2_uv'] + (1,) for match in bestInliers[0]])
    
    # Estimate the fundamental matrix
    F_, mask = cv2.findFundamentalMat(image1_uv, image2_uv, cv2.FM_RANSAC)
    print(f"Fundamental Matrix(cv2): {F_}")
    print(f"Fundamental Matrix: {F}")
    print(f"Rank of Fundamental Matrix: {np.linalg.matrix_rank(F)}")
    
    pts1 = image1_uv[mask.ravel()==1]
    pts2 = image2_uv[mask.ravel()==1]    
    
    bestInliers[0] = bestInliers[0][mask.ravel()==1]
   
    
    lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1,1,2), 2,F_)
    lines1 = lines1.reshape(-1,3)
    img5,img6 = drawlines(images[0],images[1],lines1,pts1,pts2)
    # Find epilines corresponding to points in left image (first image) and
    # drawing its lines on right image
    lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1,1,2), 1,F_)
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
    
    
    print(f"\n---------------------------------------Started Linear Triangulation-------------------")
    # Triangulate the points
    R1 = np.eye(3)
    C1 = np.zeros((3,1))
    Triangulated_points = []
    

    for i in range(4):
        points = triangulate_points(R1, C1, camera_poses[i][0], camera_poses[i][1], bestInliers[0], instrinsic_parameters)
        # print(points)
        # print(points.shape)
        Triangulated_points.append(points)
    
    Triangulated_points = np.array(Triangulated_points)
    print(Triangulated_points.shape) # (4, n, 3)
    
    for i in range(4):
        plt.axis([-20, 20, -20, 20])
        plt.scatter(Triangulated_points[i,:,0], Triangulated_points[i,:,2], s=1)
        plt.scatter(camera_poses[i][1][0], camera_poses[i][1][2], c='r', s=4)
        
    plt.title("Linear Triangulation") 
    plt.show()
        
    
    # Disambiguate the camera poses
    print(f"---------------------------------------Disambiguated Camera Pose -------------------")
    camera_pose, correct_worldpoints = disambiguate_camera_pose(camera_poses, Triangulated_points)
    print(f"Correct Camera Pose: {camera_pose}")
    
    
    pts1 = np.array([match['image1_uv'] + (1,) for match in bestInliers[0]])
    pts2 = np.array([match['image2_uv'] + (1,) for match in bestInliers[0]])
    
    
    
    print(f"---------------------------------------Started Non Linear Triangulation-------------------")
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
    
    print(f"---------------------------------------Plotting-------------------")
    fig, ax = plt.subplots()

    # Plot the reprojected points
    plt.scatter(correct_worldpoints[:,0], correct_worldpoints[:,2], s=1, c='r', label="Linear Triangulation")
    plt.scatter(reprojected_points[:,0], reprojected_points[:,2], s=1, c='b', label="Non-Linear Triangulation")
    
    # Plot camera positions and orientations
    for rotation, position, label in  [(R1, C1, "1"), (camera_pose[0], camera_pose[1], "2")]:
        # Convert rotation matrix to Euler angles
        angles = Rotation.from_matrix(rotation).as_euler('XYZ')
        angles_deg = np.rad2deg(angles)
        # print(f"Camera {label} position: {position}, orientation: {angles_deg}")
        
        ax.plot(position[0], position[2], marker=(3, 0, int(angles_deg[1])), markersize=15, linestyle='None', label=f'Camera {label}') 
        
        # Annotate camera with label
        correction = -0.1
        ax.annotate(label, xy=(position[0] + correction, position[2] + correction))

    # Setting the plot axis limits
    plt.axis([-15, 15, -5, 25])

    # Adding labels and legend
    plt.xlabel('X')
    plt.ylabel('Z')
    plt.legend()
    plt.show()
    
    R_All = []
    C_All = []
    R_All.append(np.eye(3))
    C_All.append(np.zeros((3,1)))
    
    
    # Linear PnP and Pnp Ransac
    print(f"---------------------------------------Started LinearPnP RANSAC-------------------")    
    start_imgs = [1,2]
    
    # Do this for the remaining images
    for i in range(3):
        
        start_imgs.append(i+3)
        common_points, orig_indices = find_common_features(start_imgs, bestInliers[0:i+2])
        # print(common_points)
        relevant_world_points = reprojected_points[orig_indices[:][0]]
        
        # PnP Ransac
        
        # R, C = PnPRANSAC(common_points, relevant_world_points, instrinsic_parameters)
        
        
        
        # R_All.append(R)
        # C_All.append(C)
    
    # Non Linear PnP
    
    
    
    
    R_All = []
    C_All = []
    
    #Bundle Adjustment
    
    
    
    
    
    
    
    
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", help="Path to the images", default='Phase1/P3Data/')
    args = parser.parse_args()
    main(args)
