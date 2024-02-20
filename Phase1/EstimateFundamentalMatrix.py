import numpy as np
import cv2

# Reference: https://imkaywu.github.io/blog/2017/06/fundamental-matrix/

def estimate_fundamental_matrix(matches):

    image1_uv = np.array([match['image1_uv'] + (1,) for match in matches])
    image2_uv = np.array([match['image2_uv'] + (1,) for match in matches])
    # print(image1_uv)
    
    # Normalize the coordinates
    image1_uv, T1 = normalize_pts(image1_uv)
    image2_uv, T2 = normalize_pts(image2_uv)
    
    # print(image1_uv)
    
    # Estimate the fundamental matrix
    # F, _ = cv2.findFundamentalMat(image1_uv, image2_uv, cv2.FM_RANSAC)
    # return F
    
    A = np.zeros((image1_uv.shape[0],9))
    for i in range(image1_uv.shape[0]):
        
        x_1,y_1 = image1_uv[i][0], image1_uv[i][1]
        x_2,y_2 = image2_uv[i][0], image2_uv[i][1]
        A[i] = np.array([x_1*x_2, x_2*y_1, x_2, y_2*x_1, y_2*y_1, y_2, x_1, y_1, 1])
    
    # print(A)
    # print(A.shape)
    U,S,V = np.linalg.svd(A)
    
    # print(S)
    F = V.T[-1].reshape(3,3)

    U,S,V = np.linalg.svd(F)
    # print(S)
    S[2] = 0                            #rank 2 constraint
    F = np.dot(U,np.dot(np.diag(S),V))
    
    F = np.dot(T2.T, np.dot(F, T1))   #This is given in algorithm for normalization
    F = F / F[2,2]
    
    return F

def normalize_pts(pts):
    pts_xy = np.array(pts)[:, :2]
    
    # Compute the centroid of the points
    c = np.mean(pts_xy, axis=0)
    newpts_xy = pts_xy - c
    dist = np.mean(np.linalg.norm(newpts_xy, axis=1))
    
    scale = np.sqrt(2) / dist
    newpts_xy_scaled = newpts_xy * scale
    newpts = np.hstack([newpts_xy_scaled, np.ones((newpts_xy_scaled.shape[0], 1))])
    
    T = np.array([[scale, 0, -scale * c[0]],
                  [0, scale, -scale * c[1]],
                  [0, 0, 1]])
    
    return newpts, T


def parse_matching(file_path, pair):
    matches = []
    seen_uv = set()
    
    with open(file_path, 'r') as file:
        n_features = int(file.readline().split(":")[1].strip())
        for line in file:
            parts = line.strip().split()
            
            n_matches = int(parts[0])
            color = tuple(map(int, parts[1:4]))
            current_image_uv = tuple(map(float, parts[4:6]))
            
            # For handling duplicate matches
            if current_image_uv in seen_uv:
                continue
            seen_uv.add(current_image_uv)

            i = 0
            offset = 6  # Starting offset for match information
            flag = False
            
            while offset < len(parts):
                
                if int(parts[offset]) == pair[1]:
                    pair_uv = (float(parts[offset + 1]), float(parts[offset + 2]))
                    flag = True
                    break
                    
                
                offset += 3  # Moving to the next match set
                i += 1
                if i >= n_matches:  # Break if we've read the number of matches specified for this feature
                    break
            
            
            if flag:    
                matches.append({
                    'color': color,
                    'image1_uv': current_image_uv,
                    'image2_uv': pair_uv
                })
    return matches
