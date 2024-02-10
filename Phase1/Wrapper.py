# Create Your Own Starter Code :)
#imports
import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read images
def read_images(path):
    images = []
    for i in range(5):
        images.append(cv2.imread(f"{path}{i+1}.png"))
        cv2.imshow(f"Image {i+1}", images[i])
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    return images


def read_features(path):
    pass



def main(args):
    
    basepath = args.path
    images = read_images(basepath)
    
    # Read sift features in the txt files
    # features = read_features(basepath)
  
    
    
    
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", help="Path to the images", default='Phase1/P3Data/')
    args = parser.parse_args()
    main(args)
