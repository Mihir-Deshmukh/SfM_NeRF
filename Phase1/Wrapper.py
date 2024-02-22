# Create Your Own Starter Code :)
#imports
import argparse
import cv2
import random
import numpy as np
import matplotlib.pyplot as plt
from EstimateFundamentalMatrix import *
from GetInlierRANSANC import *
from EssentialMatrixFromFundamentalMatrix import *
from ExtractCameraPose import *
from LinearTriangulation import *
from DisambiguateCameraPose import *


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
    
def get_ransac(dmatches, keypoints1, keypoints2):

    count = []
    inliers = []

    for _ in range(100): 

        random_pts = random.sample(range(len(dmatches)), 4)

        keypoints1_for_ransac = np.array([keypoints1[keypoint.queryIdx].pt for keypoint in dmatches])
        keypoints2_for_ransac = np.array([keypoints2[keypoint.trainIdx].pt for keypoint in dmatches])

        points1_for_ransac = keypoints1_for_ransac[random_pts]
        points2_for_ransac = keypoints2_for_ransac[random_pts]

        if (np.all(points1_for_ransac == points1_for_ransac[0]) or np.all(points2_for_ransac == points2_for_ransac[0]) or len(points1_for_ransac) < 4):
            # print("points are same")
            continue

        homography, mask = cv2.findHomography(points1_for_ransac, points2_for_ransac, cv2.RANSAC, 5.0)

        points = []
        final_keypoint1 = []
        final_keypoint2 = []
        count_inliers = 0

        for i in range(len(keypoints1_for_ransac)):

            keypoint1_array = np.array([keypoints1_for_ransac[i][0], keypoints1_for_ransac[i][1], 1])
            keypoint2_array = np.array([keypoints2_for_ransac[i][0], keypoints2_for_ransac[i][1], 1])

            keypoint1_array_for_homo = [keypoints1_for_ransac[i][0], keypoints1_for_ransac[i][1]]
            keypoint2_array_for_homo = [keypoints2_for_ransac[i][0], keypoints2_for_ransac[i][1]]

            ssd = np.linalg.norm(np.array(keypoint2_array.T) - np.dot(homography, keypoint1_array.T))

            if ssd < 5:
                final_keypoint1.append(keypoint1_array_for_homo)
                final_keypoint2.append(keypoint2_array_for_homo)
                count_inliers += 1

        count.append(count_inliers)
        inliers.append((homography, (final_keypoint1, final_keypoint2)))

    max_count_idx = np.argmax(count)
    final_matched_pairs = inliers[max_count_idx][1]

    return final_matched_pairs

def visualize_matches(img1, img2, matches):

    # keypoints1 = [cv2.KeyPoint(x=float(match['image1_uv'][0]), y=float(match['image1_uv'][1]), size=1) for match in matches]
    # keypoints2 = [cv2.KeyPoint(x=float(match['image2_uv'][0]), y=float(match['image2_uv'][1]), size=1) for match in matches]

    keypoints1 = matches[0]
    keypoints2 = matches[1]

    # Create dummy DMatch objects as placeholders to draw matches.
    # Note: This assumes that matches are ordered and correspond one-to-one.
    dmatches = [cv2.DMatch(_imgIdx=0, _queryIdx=i, _trainIdx=i, _distance=0) for i in range(len(matches))]
    
    # Draw matches
    img_matches = cv2.drawMatches(img1, keypoints1, img2, keypoints2, dmatches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    
    # Display the matches
    cv2.imshow("Matches", img_matches)

def convert_to_dmatches(matched_features):
    matches = [cv2.DMatch(i, i, 2) for i,j in enumerate(matched_features)]
    return matches

def convert_to_keypoints(corners):
    keypoints = [cv2.KeyPoint(c[0], c[1], 3) for c in corners]
    return keypoints

def plotter(image1, keypoints1, image2, keypoints2, dmatches):

    matched_image = cv2.drawMatches(image1, keypoints1, image2, keypoints2, dmatches, None, flags=2)

    plt.imshow(cv2.cvtColor(matched_image, cv2.COLOR_BGR2RGB))
    plt.show()

def main(args):
    
    basepath = args.path
    images = read_images(basepath)
     
    instrinsic_parameters = read_intrinsics(basepath + 'calibration.txt')
    
    image_pairs = [(1,2), (1,3), (1,4), (1,5), (2,3), (2,4), (2,5), (3,4), (3,5), (4,5)]
    
    matched_pairs = []
    
    for pair in image_pairs:
        matched_pairs.append(parse_matching(basepath + f'matching{pair[0]}.txt', pair))
        
    dmatches_before = convert_to_dmatches(matched_pairs[0])

    keypoints1_before = convert_to_keypoints([elem['image1_uv'] for elem in matched_pairs[0]])
    keypoints2_before = convert_to_keypoints([elem['image2_uv'] for elem in matched_pairs[0]])

    print(len(dmatches_before), len(keypoints1_before), len(keypoints2_before))

    # plotter(images[0], keypoints1, images[1], keypoints2, dmatches)

    matched_pairs_ransac = get_ransac(dmatches_before, keypoints1_before, keypoints2_before)

    dmatches = convert_to_dmatches(matched_pairs_ransac)

    keypoints1 = convert_to_keypoints(matched_pairs_ransac[0])
    keypoints2 = convert_to_keypoints(matched_pairs_ransac[1])

    print(len(dmatches), len(keypoints1), len(keypoints2))

    plotter(images[0], keypoints1, images[1], keypoints2, dmatches)

    # inliers = get_inlier_RANSAC(matched_pairs[0], 0.01)

    # print(matched_pairs_ransac)

    inliers = get_inlier_RANSAC(matched_pairs_ransac, 30)

    print(inliers)


    visualize_matches(images[0], images[1], inliers)
    
    F = estimate_fundamental_matrix(inliers[0], inliers[1])

    image1_uv = np.array([match['image1_uv'] + (1,) for match in matched_pairs[0]])
    image2_uv = np.array([match['image2_uv'] + (1,) for match in matched_pairs[0]])

    
    # Estimate the fundamental matrix
    F_, mask = cv2.findFundamentalMat(image1_uv, image2_uv, cv2.FM_RANSAC)
    print(f"Fundamental Matrix(cv2): {F_}")
    print(f"Fundamental Matrix: {F}")
    print(f"Rank of Fundamental Matrix: {np.linalg.matrix_rank(F)}")
    
    pts1 = image1_uv[mask.ravel()==1]
    pts2 = image2_uv[mask.ravel()==1]
    
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
    E = get_essential_matrix(F, instrinsic_parameters)
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
    # print(Triangulated_points.shape) # (4, n, 3)
    
    for i in range(4):
        plt.axis([-20, 20, -20, 20])
        plt.scatter(Triangulated_points[i,:,0], Triangulated_points[i,:,2])
        plt.scatter(camera_poses[i][1][0], camera_poses[i][1][2], c='r')
        
    plt.show()
    
    
    # Disambiguate the camera poses
    camera_pose, correct_worldpoints = disambiguate_camera_pose(camera_poses, Triangulated_points)

    plt.axis([-20, 20, -20, 20])
    plt.scatter(correct_worldpoints[:,0], correct_worldpoints[:,2])
    plt.scatter(camera_pose[1][0], camera_pose[1][2], c='r')
    plt.show()
    print(f"Correct Camera Pose: {camera_pose}")
    
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", help="Path to the images", default='Phase1/P3Data/')
    args = parser.parse_args()
    main(args)
