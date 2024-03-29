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
from PnPRANSAC import *
from NonlinearPnP import *
from BuildVisibilityMatrix import *
from BundleAdjustment import *

np.random.seed(100)

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


def find_exclusive_points(uv_list, images):
    """
    Finds points exclusive to the last image specified, ensuring they do not appear in any other specified images.
    
    :param uv_list: List of dictionaries containing UV coordinates for various images.
    :param images: List of image numbers including the target for exclusive points.
    :return: Two lists:
             - exclusive_features: Dictionaries with points exclusive to the target image.
             - exclusive_indices: Indices of these exclusive points in the original uv_list.
    """
    exclusive_features = []
    exclusive_indices = []

    # Construct keys for images to check exclusivity against (all except the last one)
    comparison_keys = [f'image{num}_uv' for num in images[1:-1]]
    # print(comparison_keys)
    # Key for the target image
    target_key = f'image{images[-1]}_uv'
    origin_key = f'image{images[0]}_uv'
    # print(target_key)
    for index, item in enumerate(uv_list):
        
        all_keys = item.keys()
        flag = True
        for key in comparison_keys:
            if key in all_keys:
                flag = False
                break
            
        # Check if the point exists in the target image and not in any of the comparison images
        if (target_key in item.keys()) and (origin_key in item.keys()) and flag:
            exclusive_features.append({'image2_uv': item[target_key], 'image1_uv': item[origin_key]})
            exclusive_indices.append(index)

    return exclusive_features, exclusive_indices

def merge_uv_coordinates(image_pairs, list_of_uv_pairs):
    merged = []

    # Convert each UV pair to its respective image numbers based on image_pairs
    for index, uv_pairs in enumerate(list_of_uv_pairs):
        img1, img2 = image_pairs[index]  # Current image pair
        for uv_pair in uv_pairs:
            # Create a new entry for the UV coordinates with their correct image labels
            new_uv_pair = {
                f'image{img1}_uv': uv_pair['image1_uv'],
                f'image{img2}_uv': uv_pair['image2_uv']
            }

            # Look for an existing entry that these UV coordinates can merge into
            found = False
            for existing_uv_pair in merged:
                # Determine if the current UV pair should merge with the existing one
                common_images = set(existing_uv_pair.keys()) & set(new_uv_pair.keys())
                # Proceed with merging only if the UV coordinates match for all common images
                if common_images and all(existing_uv_pair[img] == new_uv_pair[img] for img in common_images):
                    # Merge the UV coordinates into the existing entry
                    existing_uv_pair.update(new_uv_pair)
                    found = True
                    break

            # If no suitable existing entry is found, add this as a new entry
            if not found:
                merged.append(new_uv_pair)

    return merged


def find_common_points_and_indices(uv_list, images):
    """
    Finds common points among the specified images and their indices in the uv_list.
    
    :param uv_list: List of dictionaries containing UV coordinates for various images.
    :param images: List of image numbers to find common points for (e.g., [1, 2, 3]).
    :return: A tuple of two lists: 
             - output_features: List of dictionaries with common points among the specified images.
             - output_indices: List of indices for these common points in the uv_list.
    """
    output_features = []
    output_indices = []
    image_keys = [f'image{image_num}_uv' for image_num in images]
    # print(image_keys)
    
    for index, uv_dict in enumerate(uv_list):
        if all(key in uv_dict.keys() for key in image_keys):
            common_uv = {key: uv_dict[key] for key in image_keys}
            output_features.append(common_uv)
            output_indices.append(index)
    
    return output_features, output_indices

def plotter(image, P_mat, world_points, pts, title):

    # print(world_points)
    # print("reproj", len(reprojected_points))

    image = cv2.cvtColor(image,cv2.COLOR_GRAY2BGR)

    for i in range(len(world_points)):

        p2_1T, p2_2T, p2_3T = P_mat
        p2_1T, p2_2T, p2_3T = p2_1T.reshape(1,-1), p2_2T.reshape(1,-1), p2_3T.reshape(1,-1)

        pts_1, pts_2 = pts[i][0], pts[i][1]

        # linear reprojection error
        u2_proj = np.divide(p2_1T.dot(world_points[i]) , p2_3T.dot(world_points[i]))
        v2_proj =  np.divide(p2_2T.dot(world_points[i]) , p2_3T.dot(world_points[i]))

        cv2.circle(image, (int(pts_1), int(pts_2)), 3, [0, 0, 255], -1)
        cv2.circle(image, (int(u2_proj), int(v2_proj)), 3, [0, 255, 0], -1)

    cv2.imshow(title, image)

    if cv2.waitKey(0) & 0xff == 27: 
        cv2.destroyAllWindows() 



def main(args):
    
    print("Started SFM")
    basepath = args.path
    images = read_images(basepath)
     
    instrinsic_parameters = read_intrinsics(basepath + 'calibration.txt')
    
    image_pairs = [(1,2), (1,3), (1,4), (1,5), (2,3), (2,4), (2,5), (3,4), (3,5), (4,5)]
    
    matched_pairs = []
    n_views = 5
    
    for pair in image_pairs:
        matched_pairs.append(parse_matching(basepath + f'matching{pair[0]}.txt', pair))
        
    # Will contain best inliers for all the images
    bestInliers = []
    
    # Run RANSAC for each image pair
    for i in range(10):

        inliers = get_inlier_RANSAC(matched_pairs[i], 0.1)
        bestInliers.append(inliers)
        
        print(f"Unique matches in Image pair {image_pairs[i]} after RANSAC:", len(inliers))
    
    
    # Aggregate the UV points
    aggregated_uv_points = merge_uv_coordinates(image_pairs[0:n_views-1], bestInliers[0:n_views-1])
    # aggregated_uv_points = merge_uv_coordinates(image_pairs, bestInliers)

    
    total_image_uv = sum(len(item) for item in aggregated_uv_points)
    print(f"Total number of points in all images: {total_image_uv}")
    print(f"Total number of world points to compute: {len(aggregated_uv_points)}")
    
    bestInliers = []
    bestIndices = []
    for pair in image_pairs[0:n_views-1]:
    # for pair in image_pairs:
    
        common, indices = find_common_points_and_indices(aggregated_uv_points, pair)
        bestInliers.append(common)
        bestIndices.append(indices)
        print(f"Pair {pair}: {len(common)}")
        
    
    # Fundamental matrix between Image 1 and Image 2
    print("\n----------------------------Finding Fundamental Matrix--------------------------------")
    F = estimate_fundamental_matrix(bestInliers[0])

    image1_uv = np.array([match['image1_uv'] + (1,) for match in bestInliers[0]])
    image2_uv = np.array([match['image2_uv'] + (1,) for match in bestInliers[0]])
    
    # Estimate the fundamental matrix
    F_, mask = cv2.findFundamentalMat(image1_uv, image2_uv, cv2.FM_RANSAC)
    # print(f"Fundamental Matrix(cv2): {F_}")
    print(f"Fundamental Matrix: {F}")
    print(f"Rank of Fundamental Matrix: {np.linalg.matrix_rank(F)}")
    
    # pts1 = image1_uv[mask.ravel()==1]
    # pts2 = image2_uv[mask.ravel()==1]
    
    pts1 = np.array([match['image1_uv'] for match in bestInliers[0]])
    pts2 = np.array([match['image2_uv'] for match in bestInliers[0]])
    
    # bestInliers[0] = bestInliers[0][mask.ravel()==1]
   
    
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
    E = get_essential_matrix(F, instrinsic_parameters)
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
    plotter(images[0], P1, correct_worldpoints, pts1, 'Linear Reprojection')
    plotter(images[0], P1, reprojected_points, pts1, 'Non-Linear Reprojection')
    
    
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
    R_All.append(camera_pose[0])
    C_All.append(camera_pose[1])
    
    print(f"R_All: {R_All}")
    print(f"C_All: {C_All}")
    
    
    # Linear PnP and Pnp Ransac
    print(f"---------------------------------------Starting PnP----------------------------------------------------")    
    start_imgs = [1,2]
    pnpError = []
    
    
    world_points = np.zeros((len(aggregated_uv_points), 4))
    print(f"World Points: {world_points.shape}")
    # print(bestIndices[0])
    
    for i in range(reprojected_points.shape[0]):
        world_points[bestIndices[0][i]] = reprojected_points[i]
        
    # Do this for the remaining images
    for i in range(n_views-2):
        
        print(f"...................................Image {i+3}...........................................")
        
        start_imgs.append(i+3)
        common_points, orig_indices = find_common_points_and_indices(aggregated_uv_points, start_imgs)
        print(f"Common features with {start_imgs} images: {len(common_points)}")

        indices = np.array(orig_indices)
        relevant_world_points = world_points[indices]
        # print(relevant_world_points)
        
        common_points_img = [match[f'image{i+3}_uv'] + (1,) for match in common_points]
        pts1 = [match['image1_uv'] + (1,) for match in common_points]
        P1 = get_projectionMatrix(instrinsic_parameters, np.eye(3), np.zeros((3,1)))
        
        # Linear & non linear PnP error
        # PnP Ransac
        camera_pose, new_inliers = PnPRANSAC(common_points_img, relevant_world_points, instrinsic_parameters)
        R_new, C_new = camera_pose
        P = get_projectionMatrix(instrinsic_parameters, R_new, C_new)
        print(f"Inliers after PnP RANSAC: {len(new_inliers)}")
        
        # Non Linear PnP
        R_new, C_new = NonLinearPnP(common_points_img, relevant_world_points, R_new, C_new, instrinsic_parameters)
        print(f"Camera Pose after Non Linear PnP: {R_new}, {C_new}")
        P = get_projectionMatrix(instrinsic_parameters, R_new, C_new)
        
        # NonLinear PnP error
        pnpError = []
        for j in range(len(relevant_world_points)):
            pnpError.append(reprojection_error_pnp(P, common_points_img[j], relevant_world_points[j]))
            
        print(f"NonLinear PnP Error: {np.mean(pnpError)}")
        
        # Find the uncommon points between (1, i+3) which are not in the common points and generate world points for them
        unique_points, unique_indices = find_exclusive_points(aggregated_uv_points, start_imgs)
        print(f"Unique features for image {i+3}: {len(unique_points)}")
        pts1 = np.array([match['image1_uv'] + (1,) for match in unique_points])
        pts2 = np.array([match[f'image2_uv'] + (1,) for match in unique_points])
        
        points = triangulate_points(R1, C1, R_new, C_new, unique_points, instrinsic_parameters)
        reprojected_world_points = NonlinearTriangulation(instrinsic_parameters, R1, C1, R_new, C_new, points, pts1, pts2)
        
        for j in range(reprojected_world_points.shape[0]):
            world_points[unique_indices[j]] = reprojected_world_points[j]
        print(f"Reprojected World Points: {len(world_points)}")
        
        R_All.append(R_new)
        C_All.append(C_new)
         
        
    
    print(f"--------------------------------------PnP Done-------------------------------------------------------")    
    print(f"R_All: {R_All}")
    print(f"C_All: {C_All}")

    print(f"--------------------------------------Building Visibility Matrix-------------------------------------------------------")
    
    number_of_cameras = n_views
    visibility_matrix = build_visibility_matrix(number_of_cameras, aggregated_uv_points)
    
    print(f"Visibility Matrix: {visibility_matrix.shape} and World Points: {world_points.shape}")
    #Bundle Adjustment
    print(f"--------------------------------------Bundle Adjustment (For all cameras)-------------------------------------------------------")
    R_News, C_News, X = bundleAdjustment(aggregated_uv_points, np.array(world_points[:, :3]), visibility_matrix, np.array(R_All), np.array(C_All), number_of_cameras, instrinsic_parameters)
    
    print(f"--------------------------------------Bundle Adjustment (For all cameras) Done-------------------------------------------------------")
    print(f"R_New: {R_News}")
    print(f"C_New: {C_News}")
    
    # plot the camera positions and orientations
    fig, ax = plt.subplots()
    plt.scatter(world_points[:,0], world_points[:,2], s=1, c='r', label="Before Bundle Adjustment")
    plt.scatter(X[:,0], X[:,2], s=1, c='b', label="After Bundle Adjustment")
    for i in range(n_views):
        # Convert rotation matrix to Euler angles
        angles = Rotation.from_matrix(R_News[i]).as_euler('XYZ')
        angles_deg = np.rad2deg(angles)
        # print(f"Camera {label} position: {position}, orientation: {angles_deg}")
        
        ax.plot(C_News[i][0], C_News[i][2], marker=(3, 0, int(angles_deg[1])), markersize=15, linestyle='None', label=f'Camera {i+1}') 
        
        # Annotate camera with label
        correction = -0.1
        ax.annotate(i+1, xy=(C_News[i][0] + correction, C_News[i][2] + correction))
    
    plt.axis([-20, 20, -10, 25])
    plt.xlabel('X')
    plt.ylabel('Z')
    plt.legend()
    plt.title("Camera Poses and World Points before and after Bundle Adjustment")
    plt.show()
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", help="Path to the images", default='Phase1/P3Data/')
    args = parser.parse_args()
    main(args)
