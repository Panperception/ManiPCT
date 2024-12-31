import torch
import torch.nn as nn
import torch.optim as optim
import os
from tqdm import tqdm
from dataset import *
from encoder import *
from ddm1 import *
from decoder1 import *
import logging
import tensorboard
from visual import *

class ModelConfig(object):
    def __init__(self):
        self.nblocks = 4  # Number of transformer blocks
        self.nneighbor = 16  # Number of neighbor points
        self.transformer_dim = 1024  # Dimensionality of the transformer

class Config(object):
    def __init__(self):
        self.num_point = 2048  # Number of points in the point cloud
        self.model = ModelConfig()  # Model configuration
        self.num_class = 3  # Number of classes
        self.input_dim = 3  # Input dimension for point cloud data

class Diffusion:
    def __init__(self, noise_steps=1000, beta_start=1e-4, beta_end=0.02, device=None, n_channels=1, n_classes=1):
        self.noise_steps = noise_steps  # Number of noise steps
        self.beta_start = beta_start  # Starting value of beta
        self.beta_end = beta_end  # Ending value of beta
        self.n_channels = n_channels  # Number of channels
        self.n_classes = n_classes  # Number of classes
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device is None else device  # Device to run on
        self.model = UNet(self.n_channels, self.n_classes)  # UNet model for denoising

        # Prepare the noise schedule
        self.beta = torch.linspace(self.beta_start, self.beta_end, self.noise_steps).to(self.device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0).to(self.device)

    def sample_timesteps(self, n):
        """
        Sample time steps for the diffusion process.
        Args:
            n (int): Number of time steps to sample.
        Returns:
            torch.Tensor: Time steps tensor of shape (n, 1024).
        """
        return torch.randint(low=1, high=self.noise_steps, size=(n, 1024)).to(self.device)

    def q_sample(self, x_start, t):
        """
        Add noise to the point cloud at a given time step.
        Args:
            x_start (torch.Tensor): Initial point cloud tensor of shape (B, 1, 1024).
            t (torch.Tensor): Time step tensor of shape (B, 1, 1024).
        Returns:
            tuple: Tuple containing the noised point cloud and the added noise.
        """
        noise = torch.randn_like(x_start).to(self.device)
        return (
            x_start * torch.sqrt(self.alpha_hat[t]) +
            noise * torch.sqrt(1 - self.alpha_hat[t])
        ), noise

    def p_sample(self, model, x_t, t):
        """
        Sample from the reverse diffusion process.
        Args:
            model (torch.nn.Module): The denoising model.
            x_t (torch.Tensor): Noised point cloud tensor at time step t.
            t (torch.Tensor): Time step tensor of shape (B, 1, 1024).
        Returns:
            torch.Tensor: Predicted point cloud at the previous time step.
        """
        with torch.no_grad():
            model.eval()
            pred_noise = model(x_t, t)
            if t > 0:
                x_pred = (x_t - self.beta[t] / torch.sqrt(1 - self.alpha_hat[t]) * pred_noise) \
                         / torch.sqrt(self.alpha[t])
            else:
                x_pred = x_t
        return x_pred

    def p_sample_loop(self, model, shape):
        """
        Generate point cloud by sampling from the reverse diffusion process.
        Args:
            model (torch.nn.Module): The denoising model.
            shape (tuple): Shape of the generated point cloud tensor.
        Returns:
            torch.Tensor: Generated point cloud tensor.
        """
        cur_x = torch.randn(shape).to(self.device)
        std = (self.beta_start + self.beta_end) / 2
        cur_x = cur_x * std
        for t in reversed(range(1, self.noise_steps)):
            cur_x = self.p_sample(model, cur_x, t)
        cur_x = cur_x.view(-1, 1024)
        return cur_x

    def sample(self, model, batch_size):
        """
        Sample a batch of point clouds.
        Args:
            model (torch.nn.Module): The denoising model.
            batch_size (int): Number of point clouds to generate.
        Returns:
            torch.Tensor: Generated point cloud tensor of shape (batch_size, 1024).
        """
        return self.p_sample_loop(model, (batch_size, 1, 1024))

class PointCloudModel(nn.Module):
    def __init__(self, cfg):
        super(PointCloudModel, self).__init__()
        self.diffusion = Diffusion()
        self.model_feature = PointTransformerSeg(cfg)
        self.pcg = PointCloudGenerator()

        # Move the model to the appropriate device
        self.to_device()

    def to_device(self):
        """
        Move the model components to the appropriate device (CPU or GPU).
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.diffusion.model.to(self.device)
        self.model_feature.to(self.device)
        self.pcg.to(self.device)

    def generate_point_cloud(self, model, batch_size):
        """
        Generate a batch of point clouds using the diffusion and point cloud generator models.
        Args:
            model (torch.nn.Module): The denoising model.
            batch_size (int): Number of point clouds to generate.
        Returns:
            torch.Tensor: Generated point cloud tensor of shape (batch_size, 1024, 3).
        """
        latent_vectors = self.diffusion.p_sample_loop(model, (batch_size, 1, 1024))
        return self.pcg(latent_vectors)

    def forward(self, points):
        """
        Forward pass of the point cloud model.
        Args:
            points (torch.Tensor): Input point cloud tensor of shape (B, 2048, 3).
        Returns:
            tuple: Tuple containing the output point cloud and the denoised point cloud.
        """
        # Encode the point cloud
        feature_points = self.model_feature(points)

        # Sample time steps
        t = self.diffusion.sample_timesteps(feature_points.size(0)).to(self.device)

        # Add noise
        noised_fp, noise = self.diffusion.q_sample(feature_points, t)

        # Adjust shape to fit the decoder model
        noised_fp = noised_fp.view(-1, 1, 1024)  # Assume batch size is 4
        t = t.view(-1, 1, 1024)

        # Denoise
        latent_output = self.diffusion.model(noised_fp, t)
        latent_output = latent_output.view(-1, 1024)  # Reshape back to original batch size
        noised_fp = noised_fp.view(-1, 1024)

        # Decode to get the output point cloud
        output = self.pcg(latent_output)
        output_noise = self.pcg(noise)
        return output, output_noise

# Model and training configuration initialization
cfg = Config()
model = PointCloudModel(cfg)

# Load pre-trained weights. Assume the weights are named 'diffusion_weights.pth', 'model_feature_weights.pth', 'pcg_weights.pth'
# diffusion_state_dict = torch.load('E:\\BaiduNetdiskDownload\\P\\v2_mod\\denoising_model.pth')
# model_feature_state_dict = torch.load('E:\\BaiduNetdiskDownload\\P\\v2_mod\\point_transformer_seg.pth')
# pcg_state_dict = torch.load('E:\\BaiduNetdiskDownload\\P\\v2_mod\\point_cloud_generator.pth')

# Apply pre-trained weights
model.diffusion.model.load_state_dict(diffusion_state_dict)
model.model_feature.load_state_dict(model_feature_state_dict)
model.pcg.load_state_dict(pcg_state_dict)

# Ensure the model is on the correct device
model.to_device()

# Generate point clouds
output_point_cloud = model.generate_point_cloud(model, 4)

# Convert the generated point clouds to numpy
output_point_cloud_np = output_point_cloud.detach().cpu().numpy()

# Write the first generated point cloud to a .ply file
pcwrite(
    filename="E:\\BaiduNetdiskDownload\\P\\chair1.ply",  # Output .ply file location
    xyz=output_point_cloud_np[0]  # Only take the first point cloud
)

# Visualize the generated point clouds
for j in range(output_point_cloud_np.shape[0]):
    xyz = output_point_cloud_np[j][:, :3]
    pcd = o3d.geometry.PointCloud()

    # Set the point cloud coordinates
    pcd.points = o3d.utility.Vector3dVector(xyz)
    o3d.visualization.draw_geometries([pcd])
