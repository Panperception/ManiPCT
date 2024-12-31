import os
import torch
import torch.nn as nn
from tqdm import tqdm
from torch import optim
from pyemd import emd_samples
from dataset import *
from encoder import *
from DDM import *
from decoder import *
import logging
import tensorboard
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
class ModelConfig(object):
    def __init__(self):
        self.nblocks = 4
        self.nneighbor = 16
        self.transformer_dim = 1024

class Config(object):
    def __init__(self):
        self.num_point = 2048
        self.model = ModelConfig()
        self.num_class = 3
        self.input_dim = 3
class Diffusion:
    def __init__(self, noise_steps=1000, beta_start=1e-4, beta_end=0.02, device=None, n_channels = 1, n_classes=1):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device is None else device
        self.model = UNet(self.n_channels, self.n_classes)
        # Prepare the noise schedule
        self.beta = torch.linspace(self.beta_start, self.beta_end, self.noise_steps).to(self.device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0).to(self.device)
    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n,1024))
    def q_sample(self, x_start, t):
        noise = torch.randn_like(x_start).to(self.device)
        return (
            x_start * torch.sqrt(self.alpha_hat[t]) +
            noise * torch.sqrt(1 - self.alpha_hat[t])
        ), noise

    def p_sample(self, model, x_t, t):
        with torch.no_grad():
            if t == 0:
                return x_t
            else:
                z = torch.randn_like(x_t)
                pred_noise = model(x_t, t)
                return (
                    torch.sqrt(self.alpha_hat[t - 1]) * (x_t - torch.sqrt(1 - self.alpha_hat[t]) * pred_noise) +
                    torch.sqrt(1 - self.alpha_hat[t - 1]) * z
                )

    def p_sample_loop(self, model, shape):
        cur_x = torch.randn(shape).to(self.device)
        for t in reversed(range(0, self.noise_steps)):
            cur_x = self.p_sample(model, cur_x, t)
        return cur_x

    def sample(self, model, batch_size):
        return self.p_sample_loop(model, (batch_size, 1024))
# Instantiate configuration. Can be passed directly to the model.
def chamfer_distance1(p, q):
    """
    Compute Chamfer Distance between two point clouds
    Args:
        p: a BxNx3 tensor, where B is batch size, N is number of points in point clouds
        q: a BxMx3 tensor, B is batch size (should be the same in p, q) and M is number of points
    Returns:
        A scalar tensor with the Chamfer Distance
    """
    # Compute pairwise distances between p and q, BxNxM
    p = p.unsqueeze(2)
    q = q.unsqueeze(1)
    diff = p - q
    dist = torch.sum(diff ** 2, dim=-1)

    # For each point in p, find the nearest point in q and compute the squared distance
    dist_pq = torch.min(dist, dim=2)[0]
    dist_qp = torch.min(dist, dim=1)[0]

    # Compute the final Chamfer Distance
    chamfer_dist = torch.mean(dist_pq, dim=1) + torch.mean(dist_qp, dim=1)

    return chamfer_dist.mean()

def chamfer_distance(p, q):
    """
    Compute Chamfer Distance between two point clouds for a single sample
    Args:
        p: an Nx3 tensor, where N is number of points in point cloud
        q: an Mx3 tensor, where M is number of points
    Returns:
        A scalar tensor with the Chamfer Distance
    """
    # Compute pairwise distances between p and q, NxM
    p = p.unsqueeze(1)  # p is now Nx1x3
    q = q.unsqueeze(0)  # q is now 1xMx3
    diff = p - q  # Broadcasting to compute pairwise distance
    dist = torch.sum(diff ** 2, dim=-1)  # Squared distance, NxM

    # For each point in p, find the nearest point in q and compute the squared distance
    dist_pq = torch.min(dist, dim=1)[0]  # Nearest point distance vector, N

    # For each point in q, find the nearest point in p and compute the squared distance
    dist_qp = torch.min(dist, dim=0)[0]  # Nearest point distance vector, M

    # Compute the final Chamfer Distance
    chamfer_dist = dist_pq.mean() + dist_qp.mean()

    return chamfer_dist

def earth_mover_distance(p, q):
    """
    Compute Earth Mover's Distance (EMD) between two point clouds

    Args:
        p: a BxNx3 tensor, where B is batch size, N is number of points in point cloud
        q: a BxMx3 tensor, B is batch size (should be the same in p, q), and M is number of points

    Returns:
        A scalar tensor containing the average EMD for all batches
    """
    # Ensure inputs are numpy arrays to be compatible with pyemd library
    p_np = p.cpu().detach().numpy()
    q_np = q.cpu().detach().numpy()

    batch_size = p_np.shape[0]
    emd_sum = 0.0

    for i in range(batch_size):
        # Compute EMD between two point clouds
        emd_i = emd_samples(p_np[i], q_np[i])
        emd_sum += emd_i

    # Compute average EMD
    avg_emd = emd_sum / batch_size

    return avg_emd

# Modify your train function to accept optional preloaded model parameters
def train(num_epochs, pcg_path=None, model_feature_path=None, denoising_model_path=None):
    """
    Train the models with optional preloaded model parameters.
    Args:
        num_epochs: Number of epochs to train.
        pcg_path: Path to the pre-saved PointCloudGenerator weights.
        model_feature_path: Path to the pre-saved PointTransformerSeg weights.
        denoising_model_path: Path to the pre-saved denoising model weights.
    """
    cfg = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    diffusion = Diffusion()
    denoising_model = diffusion.model.to(device)
    model_feature = PointTransformerSeg(cfg).to(device)
    pcg = PointCloudGenerator().to(device)

    if pcg_path:
        pcg.load_state_dict(torch.load(pcg_path, map_location=device))
    if model_feature_path:
        model_feature.load_state_dict(torch.load(model_feature_path, map_location=device))
    if denoising_model_path:
        denoising_model.load_state_dict(torch.load(denoising_model_path, map_location=device))

    criterion = torch.nn.MSELoss()
    params_to_optimize = list(denoising_model.parameters()) + list(model_feature.parameters()) + list(pcg.parameters())
    optimizer = torch.optim.Adam(params_to_optimize)

    for epoch in range(num_epochs):
        pbar = tqdm(train_loader)
        for i, data in enumerate(pbar):
            points = data['train_points'].to(device)
            optimizer.zero_grad()
            feature_points = model_feature(points)
            t = diffusion.sample_timesteps(feature_points.size(0)).to(device)
            noised_fp, noise = diffusion.q_sample(feature_points, t)
            out_noise_fp = pcg(noised_fp, t)
            noised_fp = noised_fp.view(4, 1, 1024)
            latent_output = denoising_model(noised_fp)
            latent_output = latent_output.view(4, 1024)
            output = pcg(latent_output, t)
            x = pcg(feature_points, t)
            loss1 = chamfer_distance1(points, x)

            out_noise = pcg(noise, t)
            out_points = out_noise_fp - out_noise
            loss2 = chamfer_distance1(points, out_points)

            loss = criterion(output, out_noise)
            total_loss = loss2 + loss1 + loss
            total_loss.backward()
            optimizer.step()
            pbar.set_description(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss.item()}")

    torch.save(pcg.state_dict(), 'E:\\BaiduNetdiskDownload\\P\\v2_mod\\point_cloud_generator.pth')
    torch.save(model_feature.state_dict(), 'E:\\BaiduNetdiskDownload\\P\\v2_mod\\point_transformer_seg.pth')
    torch.save(denoising_model.state_dict(), 'E:\\BaiduNetdiskDownload\\P\\v2_mod\\denoising_model.pth')

# Train the model for the first time with original parameters
# Continue training the model based on saved weights, provide the file paths to the previously saved weights


if __name__ == '__main__':
    train(2)


