import numpy as np

def build_visibility_matrix(number_of_cameras, all_points):
    # Initialize the visibility matrix with zeros
    # Rows correspond to world points, columns to cameras
    visibility_matrix = np.zeros((len(all_points), number_of_cameras), dtype=int)
    
    # Iterate over each world point
    for point_idx, point_dict in enumerate(all_points):
        # Iterate over each key (image identifier) in the dictionary
        for key in point_dict.keys():
           
            camera_idx = int(key.split('_')[0][5:]) - 1  # Adjust index if necessary
            
            # Mark the world point as visible in this camera
            visibility_matrix[point_idx, camera_idx] = 1
            
    return visibility_matrix