import argparse
import glob
from tqdm import tqdm
import random
# from torch.utils.tensorboard import SummaryWriter
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

    data = np.load("amdisa_p2/Phase2/Data/tiny_nerf_data.npz")
    images = data["images"][:100]
    poses = data["poses"][:100]
    poses = torch.from_numpy(poses).to(device)
    focal =  data["focal"]
    focal = torch.from_numpy(focal)
    # print(images.shape)
    # plt.imshow(images[1])
    # plt.show()

    H, W = images.shape[1:3]
 
    images = torch.from_numpy(images).to(device)
    camera_info = (H, W, focal)
    return camera_info, images, poses

def PixelToRay(camera_info, pose):
    """
    Input:
        camera_info: image width, height, camera matrix
        pose: camera pose in world frame
        pixelPoition: pixel position in the image
        args: get near and far range, sample rate ...
    Outputs:
        ray origin and direction
    """
    H, W, focal = camera_info
    mesh_x, mesh_y = torch.meshgrid(torch.linspace(0, W-1, W), torch.linspace(0, H-1, H), indexing='ij')
    mesh_x = mesh_x.to(device)
    mesh_y = mesh_y.to(device)

    x = (mesh_x - W/2) / focal
    y = (mesh_y - H/2) / focal

    directions = torch.stack((x, -y, -torch.ones_like(x)), dim=-1)
    # print(directions.shape)
    directions = directions[..., np.newaxis, :]

    rotation = pose[:3, :3]
    translation = pose[:3, -1].view(1, 1, 3)

    ray_direction = torch.matmul(directions, rotation)
    ray_direction = ray_direction.squeeze(2)
    
    ray_direction = ray_direction/torch.linalg.norm(ray_direction, axis=-1, keepdim=True)
    ray_origin = translation.expand(ray_direction.shape[0], ray_direction.shape[1], -1)

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
    n_bins = 64
    t_near = 2
    t_far = 6
    
    t = torch.linspace(t_near, t_far, n_bins)
    t = t.expand(rays_origin.shape[0], rays_origin.shape[1], n_bins).clone().to(device)
    # print(f" T Shape: {t.shape}")
    
    random_offsets = torch.rand(*rays_origin.shape[:-1], n_bins) * (t_far - t_near) / n_bins
    # print(f" random_offsets Shape: {random_offsets.shape}")
    t += random_offsets.to(device)
    # print(f" T Shape: {t.shape}")
    # print(f" T: {t[0, 0, :]}")
    # print(f" t new shape: {t.unsqueeze(-1).shape}")
    query_input = rays_origin.unsqueeze(2) + rays_direction.unsqueeze(2) * t.unsqueeze(-1)
    # print(f" Input shape: {query_input.shape}")
    
    # colors, sigma = model(query_input, rays_direction)
    flattened_query_input = query_input.view(-1, 3)
    colors, sigma = model(flattened_query_input)
    colors = colors.view(*query_input.shape[:-1], 3)
    sigma = sigma.view(*query_input.shape[:-1])
    
    # colors = torch.rand(*query_input.shape)   
    # sigma = torch.rand(*t.shape)
    # print(f" Sigma shape: {sigma.shape}")
    # print(f" Colors shape: {colors.shape}")
    
    # Use Volume rendering to get the final image
    dists = torch.zeros_like(t)
    # Fill in the calculated distances into the new tensor, leaving the last entry as zero
    dists[..., :-1] =  t[..., 1:] - t[..., :-1]
    # Now, set the last entry to 1e10 to simulate an "infinite" distance for the last sample
    dists[..., -1] = 1e10
    #200*n_bins_*6
    
    # print(f" Dists shape: {dists.shape}")

    # Compute alpha values from sigma and distances
    alpha = 1.0 - torch.exp(-sigma * dists)
    # print(f" Alpha shape: {alpha.shape}")
    
    
    # Compute weights for RGB and depth accumulation
    adjusted_alpha = 1.0 - alpha + 1e-10

    # Pad the tensor with ones at the beginning of the dimension of interest
    padded_alpha = torch.cat([torch.ones(*adjusted_alpha.shape[:-1], 1).to(device), adjusted_alpha], dim=-1)

    # Perform the cumulative product on the padded tensor
    cumprod_padded = torch.cumprod(padded_alpha, dim=-1)

    # Remove the first element to get the "exclusive" cumulative product
    T = cumprod_padded[..., 1:]
    
    weights = alpha * T

    
    # Compute weighted sum of colors to get RGB map
    rgb_map = torch.sum(weights.unsqueeze(-1) * colors, dim=-2)
    # print(f" RGB shape: {rgb_map.shape}")
    
    # Compute weighted sum of z values to get depth map
    depth_map = torch.sum(weights * t, dim=-1)

    # Compute sum of weights to get accumulated map
    acc_map = torch.sum(weights, dim=-1)

    return rgb_map
    
        
def visualize_rays(ray_direction, ray_origin):

    distance = 0.1

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Subsample the rays for plotting
    for i in range(0, ray_direction.shape[0], 10):
        for j in range(0, ray_direction.shape[1], 10):
            
            # Plot a line from the ray origin to the end point
            ax.quiver(ray_origin[i, j, 0].item(), ray_origin[i, j, 1].item(), ray_origin[i, j, 2].item(),
                    ray_direction[i, j, 0].item(), ray_direction[i, j, 1].item(), ray_direction[i, j, 2].item(),
                    length=distance, normalize=True)

    # Set labels
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    ax.set_title('Ray Directions at Distance ' + str(distance))

    # Show the plot
    plt.show()

def loss(groundtruth, prediction):
    return F.mse_loss(groundtruth, prediction)
    

def train(images, poses, camera_info, args):

    # model = NeRFmodel().to(device)
    model = TinyNeRFmodel().to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=args.lrate)

    Loss = []
    Epochs = []

    for i in range(args.max_iters):
    # for i in range(400):
        print(f" Iteration: {i}")

        random_idx = random.randint(0, images.shape[0]-1)
        img = images[random_idx]
        pose = poses[random_idx]
       
        rays_direction, rays_origin = PixelToRay(camera_info, pose)
       
        
        # print(f" Ray direction: {rays_direction.shape}, Ray origin shape: {rays_origin.shape}")
        # visualize_rays(rays_direction, rays_origin)
        rgb_pred = render(model, rays_origin, rays_direction, args)
        # print(f" RGB shape: {rgb_pred.shape}")
        
        current_loss = loss(img, rgb_pred)
        
        optimiser.zero_grad()
        current_loss.backward()
        optimiser.step()
        
        Loss.append(current_loss.item())
        
        if i % 200 == 0:
            print(f" Iteration: {i}, Loss: {current_loss}")
            plt.imshow(rgb_pred.cpu().detach().numpy())
            plt.show()

        

# def test(images, poses, camera_info, args):


def main(args):
    # load data

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading data...")
    camera_info, images, poses = loadDataset(args.data_path, args.mode, device)
    print("Data loaded")

    if args.mode == 'train':
        print("Start training")
        train(images, poses, camera_info, args)
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