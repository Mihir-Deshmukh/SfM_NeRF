import argparse
import glob
from tqdm import tqdm
import random
from torch.utils.tensorboard import SummaryWriter
import imageio
import torch
import matplotlib.pyplot as plt
import os
import cv2
import numpy as np
import json

from NeRFModel import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(0)

def loadDataset(data_path, mode, device):
    """
    Input:
        data_path: dataset path
        mode: train or test
    Outputs:
        camera_info: image width, height, camera matrix 
        images: images
        pose: corresponding camera pose in world frame
    """
    # with open(data_path, 'r') as file:
    #     data = json.load(file)

    # # Accessing data
    # camera_angle_x = data['camera_angle_x']
    # frames = data['frames']

    # file_paths = []
    # rotations = []
    # poses = []

    # # Example of accessing specific information
    # for frame in frames:

    #     file_path = frame['file_path']
    #     rotation = frame['rotation']
    #     transform_matrix = frame['transform_matrix']

    #     file_paths.append(file_path)
    #     rotations.append(rotation)
    #     poses.append(transform_matrix)

    # image_path = "Phase2/Data/lego/lego/train"
    # images = []

    # for i in range(len(os.listdir(image_path))):
    #     img = cv2.imread(os.path.join(image_path, f"r_{i}.png"))
    #     images.append(img)
        
    # cv2.imshow("image", images[1])
    # cv2.waitKey(0)

    data = np.load("Phase2/Data/tiny_nerf_data.npz")
    images = data["images"]
    poses = data["poses"]
    poses = torch.from_numpy(poses).to(device)
    focal =  data["focal"]
    focal = torch.from_numpy(focal)

    # plt.imshow(images[1])
    # plt.show()

    H, W = images.shape[1:3]

    images = torch.from_numpy(images).to(device)

    return images, poses, focal, H, W

def PixelToRay(H, W, focal, pose):
    """
    Input:
        camera_info: image width, height, camera matrix 
        pose: camera pose in world frame
        pixelPoition: pixel position in the image
        args: get near and far range, sample rate ...
    Outputs:
        ray origin and direction
    """

    mesh_x, mesh_y = torch.meshgrid(torch.linspace(0, W-1, W), torch.linspace(0, H-1, H), indexing='ij')
    mesh_x = mesh_x.to(device)
    mesh_y = mesh_y.to(device)

    x = (mesh_x - W/2) / focal
    y = (mesh_y - H/2) / focal

    directions = torch.stack((x, -y, -torch.ones_like(x)), dim=-1)
    directions = directions[..., np.newaxis, :]

    rotation = pose[:3, :3]
    translation = pose[:3, -1].view(1, 1, 3)

    ray_direction = torch.sum(torch.matmul(directions, rotation), axis=-1)
    ray_direction = ray_direction/torch.linalg.norm(ray_direction, axis=-1, keepdim=True)

    # ray_origin = torch.broadcast_to(translation, ray_direction.shape)
    ray_origin = translation.expand(ray_direction.shape[0], -1, -1)

    print(ray_direction.shape, ray_origin.shape)

    return ray_direction, ray_origin

def generateBatch(images, poses, camera_info, args):
    """
    Input:
        images: all images in dataset
        poses: corresponding camera pose in world frame
        camera_info: image width, height, camera matrix
        args: get batch size related information
    Outputs:
        A set of rays
    """



def render(model, rays_origin, rays_direction, args):
    """
    Input:
        model: NeRF model
        rays_origin: origins of input rays
        rays_direction: direction of input rays
    Outputs:
        rgb values of input rays
    """

# def loss(groundtruth, prediction):
    
def visualize_rays(ray_direction, ray_origin):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for i in range(ray_direction.shape[0]):
        ax.plot3D(ray_origin[i, :, 0], ray_origin[i, :, 1], ray_origin[i, :, 2],
                  ray_direction[i, :, 0], ray_direction[i, :, 1], ray_direction[i, :, 2])

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Ray Directions')

    plt.show()

def train(images, poses, focal, H, W,args):

    model = NeRFmodel()
    model = model.to(device)

    optimiser = torch.optim.Adam(model.parameters(), lr=args.lrate)

    Loss = []
    Epochs = []

    # for i in range(args.epochs):
    for i in range(1):

        random_idx = random.randint(0, images.shape[0])
        img = images[random_idx]
        pose = poses[random_idx]

        rays_direction, rays_origin = PixelToRay(H, W, focal, pose)

        # visualize_rays(rays_direction, rays_origin)

        

# def test(images, poses, camera_info, args):


def main(args):
    # load data

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading data...")
    images, poses, focal, H, W = loadDataset(args.data_path, args.mode, device)

    if args.mode == 'train':
        print("Start training")
        train(images, poses, focal, H, W, args)
    elif args.mode == 'test':
        print("Start testing")
        args.load_checkpoint = True
        #test(images, poses, camera_info, args)

def configParser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path',default="./Phase2/Data/lego/lego/transforms_train.json",help="dataset path")
    parser.add_argument('--mode',default='train',help="train/test/val")
    parser.add_argument('--lrate',default=5e-4,help="training learning rate")
    parser.add_argument('--n_pos_freq',default=10,help="number of positional encoding frequencies for position")
    parser.add_argument('--n_dirc_freq',default=4,help="number of positional encoding frequencies for viewing direction")
    parser.add_argument('--n_rays_batch',default=32*32*4,help="number of rays per batch")
    parser.add_argument('--n_sample',default=400,help="number of sample per ray")
    parser.add_argument('--max_iters',default=10000,help="number of max iterations for training")
    parser.add_argument('--logs_path',default="./logs/",help="logs path")
    parser.add_argument('--checkpoint_path',default="./Phase2/example_checkpoint/",help="checkpoints path")
    parser.add_argument('--load_checkpoint',default=True,help="whether to load checkpoint or not")
    parser.add_argument('--save_ckpt_iter',default=1000,help="num of iteration to save checkpoint")
    parser.add_argument('--images_path', default="./image/",help="folder to store images")
    parser.add_argument('--epochs', default=1000, help="number of epochs")
    return parser

if __name__ == "__main__":
    parser = configParser()
    args = parser.parse_args()
    main(args)