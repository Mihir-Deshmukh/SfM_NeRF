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
torch.random.manual_seed(0)

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
    images = data["images"][:100]
    test_images = data["images"][100:]
    
    poses = data["poses"][:100]
    poses = torch.from_numpy(poses).to(device)
    focal =  data["focal"]
    focal = torch.from_numpy(focal)
    # print(images.shape)
    plt.imshow(data["images"][100])
    plt.show()

    H, W = images.shape[1:3]
    test_poses = data["poses"][100:]
    test_poses = torch.from_numpy(test_poses).to(device)
 
    images = torch.from_numpy(images).to(device)
    test_images = torch.from_numpy(test_images).to(device)
    camera_info = (H, W, focal)
    return camera_info, images, poses, test_poses, test_images


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

    mesh_x, mesh_y = torch.meshgrid(torch.linspace(0, W-1, W).to(pose), torch.linspace(0, H-1, H).to(pose), indexing='ij')

    mesh_x = mesh_x.T
    mesh_y = mesh_y.T

    x = (mesh_x - W/2) / focal
    y = (mesh_y - H/2) / focal
    
    directions = torch.stack((x, -y, -torch.ones_like(x)), dim=-1)
    directions = directions[..., np.newaxis, :]
    
    rotation = pose[:3, :3]
    translation = pose[:3, -1].view(1, 1, 3)
    
    ray_direction = torch.sum(directions*rotation, dim=-1)
    ray_direction = ray_direction/torch.linalg.norm(ray_direction, axis=-1, keepdim=True)
    ray_origin = translation.expand(ray_direction.shape[0], ray_direction.shape[1], -1)

    return ray_direction, ray_origin

def generateBatch(ray_origins, ray_directions, gt_colors, args, train=True):
    """
    Input:
        images: all images in dataset
        poses: corresponding camera pose in world frame
        camera_info: image width, height, camera matrix
        args: get batch size related information
    Outputs:
        A set of rays origins, directions and gt colors
    """
    if train:
        indices = np.random.choice(ray_origins.shape[0], 1024, replace=False)
        sample_rays_o = ray_origins[indices].to(device)
        sample_rays_d = ray_directions[indices].to(device)
        sample_colors = gt_colors[indices].to(device)
    else:
        sample_rays_o = ray_origins.to(device)
        sample_rays_d = ray_directions.to(device)
        sample_colors = gt_colors.to(device)
    
    return sample_rays_o, sample_rays_d, sample_colors

def generateRays_and_gt(images, poses, camera_info, args):
    """
    Input:
        images: all images in dataset
        poses: corresponding camera pose in world frame
        camera_info: image width, height, camera matrix
        args: get batch size related information
    Outputs:
        A set of rays origins, directions and gt colors
    """
    all_rays_o = []
    all_rays_d = []
    all_colors = []

    H, W, focal = camera_info

    for i in range(images.shape[0]):
        img = images[i]
        pose = poses[i]

        rays_d, rays_o = PixelToRay(camera_info, pose)
        all_rays_o.append(rays_o.view(-1, 3))
        all_rays_d.append(rays_d.view(-1, 3))
        all_colors.append(img.view(-1, 3))

    all_rays_o = torch.cat(all_rays_o, dim=0)
    all_rays_d = torch.cat(all_rays_d, dim=0)
    all_colors = torch.cat(all_colors, dim=0)

    return all_rays_o, all_rays_d, all_colors

def compute_accumulated_transmittance(alphas):
    accumulated_transmittance = torch.cumprod(alphas, 1)
    return torch.cat((torch.ones((accumulated_transmittance.shape[0], 1), device=alphas.device),
                      accumulated_transmittance[:, :-1]), dim=-1)


def render_rays(nerf_model, ray_origins, ray_directions, hn=2, hf=6, nb_bins=192):
    device = ray_origins.device
    
    t = torch.linspace(hn, hf, nb_bins, device=device).expand(ray_origins.shape[0], nb_bins)
    # Perturb sampling along each ray.
    mid = (t[:, :-1] + t[:, 1:]) / 2.
    lower = torch.cat((t[:, :1], mid), -1)
    upper = torch.cat((mid, t[:, -1:]), -1)
    u = torch.rand(t.shape, device=device)
    t = lower + (upper - lower) * u  # [batch_size, nb_bins]
    delta = torch.cat((t[:, 1:] - t[:, :-1], torch.tensor([1e10], device=device).expand(ray_origins.shape[0], 1)), -1)

    # Compute the 3D points along each ray
    x = ray_origins.unsqueeze(1) + t.unsqueeze(2) * ray_directions.unsqueeze(1)   # [batch_size, nb_bins, 3]
    # Expand the ray_directions tensor to match the shape of x
    ray_directions = ray_directions.expand(nb_bins, ray_directions.shape[0], 3).transpose(0, 1) 

    colors, sigma  = nerf_model(x.reshape(-1, 3))
    # print(f"Output shape: {output.shape}")
    colors = colors.reshape(x.shape)
    sigma = sigma.reshape(x.shape[:-1])

    alpha = 1 - torch.exp(-sigma * delta)  # [batch_size, nb_bins]
    weights = compute_accumulated_transmittance(1 - alpha).unsqueeze(2) * alpha.unsqueeze(2)
    # Compute the pixel values as a weighted sum of colors along each ray
    c = (weights * colors).sum(dim=1)
    weight_sum = weights.sum(-1).sum(-1)  # Regularization for white background 
    return c + 1 - weight_sum.unsqueeze(-1)

def render(model, rays_origin, rays_direction, args):
    """
    Input:
        model: NeRF model
        rays_origin: origins of input rays
        rays_direction: direction of input rays
    Outputs:
        rgb values of input rays
    """
    n_bins = 192
    t_near = 2
    t_far = 6
    
    t = torch.linspace(t_near, t_far, n_bins)
    t = t.expand(rays_origin.shape[0], n_bins).clone().to(device)
    
    random_offsets = torch.rand(*rays_origin.shape[:-1], n_bins) * (t_far - t_near) / n_bins
    t += random_offsets.to(device)

    query_input = rays_origin.unsqueeze(1) + rays_direction.unsqueeze(1) * t.unsqueeze(-1)
    flattened_query_input = query_input.view(-1, 3)
    
    rays_direction = rays_direction.expand(n_bins, rays_direction.shape[0], 3).transpose(0, 1).reshape(-1, 3)
    
    output = model(flattened_query_input)
    colors = torch.nn.functional.sigmoid(output[:3])
    sigma = torch.nn.functional.relu(output[3])
    colors = colors.view(*query_input.shape[:-1], 3)
    sigma = sigma.view(*query_input.shape[:-1])
    
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

    # print(f" Weights shape: {weights.shape}")
    # Compute weighted sum of colors to get RGB map
    rgb_map = torch.sum(weights.unsqueeze(-1) * colors, dim=-2)
    
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

    # model = NerfModel().to(device)
    model = VeryTinyNerfModel().to(device)
    # model.load_state_dict(torch.load("Output/checkpoint/model_500.pt", map_location=device))
    # model = TinyNeRFmodel().to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=args.lrate)
    
    Loss = []
    Epochs = []
        
    # read all images, poses and generate rays and gt colors
    ray_origins, ray_directions, gt_colors = generateRays_and_gt(images, poses, camera_info, args)
    print(f" Ray Origins: {ray_origins.shape}")
    print(f" Ray Directions: {ray_directions.shape}")
    print(f" GT Colors: {gt_colors.shape}")

    best_loss = float('inf')
    for i in range(args.max_iters):
       
        model.train()
        
        batch_ray_origins, batch_ray_directions, batch_gt_colors = generateBatch(ray_origins, ray_directions, gt_colors, args)
        # print(f"gt_colors shape: {batch_gt_colors.shape}")
        # rgb_pred = render(model, batch_ray_origins, batch_ray_directions, args)
        rgb_pred = render_rays(model, batch_ray_origins, batch_ray_directions)
        # print(f" RGB shape: {rgb_pred.shape}")
        
        current_loss = loss(batch_gt_colors, rgb_pred)
        print(f" Iteration: {i}, Loss: {current_loss.item()}")
        optimiser.zero_grad()
        current_loss.backward()
        optimiser.step()
        
        Loss.append(current_loss.item())
        
        if i%300 == 0:
            
            if current_loss.item() < best_loss:
                best_loss = current_loss
                print(f"Saving model at iteration: {i}")
                torch.save(model.state_dict(), f"{args.checkpoint_path}/model_{i}.pt")
            
            
            print(f" Iteration: {i}, Loss: {current_loss}")
            
            
            model.eval()
            with torch.no_grad():
                test_ray_origins, test_ray_directions, test_gt  = generateRays_and_gt(test_images, test_poses, camera_info, args)
                
                rgb_pred_test = []
                for i in range(100):
                    index_1 = i * 100
                    index_2 = (i+1) * 100
                    test_origins, test_directions, gt = generateBatch(test_ray_origins[index_1:index_2], test_ray_directions[index_1:index_2], test_gt[index_1:index_2], args, train=False)
                    pred = render_rays(model, test_origins, test_directions)
                    rgb_pred_test.append(pred)
                    
                rgb_pred_test = torch.cat(rgb_pred_test, dim=0)
                pred_image = rgb_pred_test.view(100, 100, 3).cpu().detach().numpy()
                gt_image = test_gt[0:10000].view(100, 100, 3).cpu().detach().numpy()
             
                # Plotting the original vs predicted images
                fig, ax = plt.subplots(1, 2, figsize=(10, 5))
                
                ax[0].imshow(gt_image)
                ax[0].set_title("Original Test Image")
                ax[0].axis('off')  # Hide axes ticks
                
                ax[1].imshow(pred_image)
                ax[1].set_title("Predicted Test Image")
                ax[1].axis('off')  # Hide axes ticks
    
                plt.show()


# def test(images, poses, camera_info, args):


def main(args):
    # load data

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading data...")
    global test_poses, test_images
    camera_info, images, poses, test_poses, test_images = loadDataset(args.data_path, args.mode, device)
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
    parser.add_argument('--n_sample',default=192,help="number of sample per ray")
    parser.add_argument('--max_iters',default=10000,help="number of max iterations for training")
    parser.add_argument('--logs_path',default="./logs/",help="logs path")
    parser.add_argument('--checkpoint_path',default="Phase2/Output/checkpoint",help="checkpoints path")
    parser.add_argument('--load_checkpoint',default=True,help="whether to load checkpoint or not")
    parser.add_argument('--save_ckpt_iter',default=1000,help="num of iteration to save checkpoint")
    parser.add_argument('--images_path', default="./image/",help="folder to store images")
    parser.add_argument('--epochs', default=1000, help="number of epochs")
    return parser

if __name__ == "__main__":
    parser = configParser()
    args = parser.parse_args()
    main(args)