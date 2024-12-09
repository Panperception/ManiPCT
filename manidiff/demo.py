import torch
import torch.nn as nn
import torch.optim as optim
import os
from tqdm import tqdm
from torch import optim
from dataset import *
from encoder import *
from ddm1 import *
from decoder1 import *
import logging
import tensorboard
from tqdm import tqdm
import open3d as o3d
from visual import *
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
            model.eval()
            pred_noise= self.model(x_t, t)
            if t > 0:
                x_pred = (x_t - self.beta[t] / torch.sqrt(1 - self.alpha_hat[t]) * pred_noise) \
                         / torch.sqrt(self.alpha[t])
            else:
                x_pred = x_t
        return x_pred
    def p_sample_loop(self, model, shape):
        cur_x = torch.randn(shape).to(self.device)
        std = (self.beta_start + self.beta_end) / 2
        cur_x = cur_x * std
        for t in reversed(range(1, self.noise_steps)):
            cur_x = self.p_sample(model, cur_x, t)
        cur_x = cur_x.view(-1, 1024)

        return cur_x

    def sample(self, model, batch_size):
        # 现在将生成数据的shape设置为(batch_size, 2500, 3)，
        return self.p_sample_loop(model, (batch_size, 1, 1024))

# 实例化配置。可以直接传递到模型中。





class PointCloudModel(nn.Module):
    def __init__(self, cfg):
        super(PointCloudModel, self).__init__()
        self.diffusion = Diffusion()
        self.model_feature = PointTransformerSeg(cfg)
        self.pcg = PointCloudGenerator()

        # 应用到设备
        self.to_device()

    def to_device(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.diffusion.model.to(self.device)
        self.model_feature.to(self.device)
        self.pcg.to(self.device)

    def generate_point_cloud(self, model, batch_size):
        latent_vectors = self.diffusion.p_sample_loop(model, (batch_size, 1, 1024))



        return self.pcg(latent_vectors)

    def forward(self, points):
        # 编码点云
        feature_points = self.model_feature(points)

        # 采样时间步长
        t = self.diffusion.sample_timesteps(feature_points.size(0)).to(self.device)

        # 添加噪声
        noised_fp, noise = self.diffusion.q_sample(feature_points, t)

        # 调整形状以适应解码器模型
        noised_fp = noised_fp.view(-1, 1, 1024)  # 假设 batch size 为 4
        t = t.view(-1,1,1024)
        # 降噪
        latent_output = self.diffusion.model(noised_fp, t)
        latent_output = latent_output.view(-1, 1024)  # 还原成原始 batch size 形状
        noised_fp = noised_fp.view(-1, 1024)
        # 解码得到输出点云
        output = self.pcg(latent_output)
        output_noise = self.pcg(noise)
        return output, output_noise





# 模型和训练配置 初始化
cfg = Config()

model = PointCloudModel(cfg)

# 载入预训练权重，假设您的权重命名为'diffusion_weights.pth', 'model_feature_weights.pth', 'pcg_weights.pth'
diffusion_state_dict = torch.load('E:\\BaiduNetdiskDownload\\P\\v2_mod\\car\\2\\denoising_model.pth')
model_feature_state_dict = torch.load('E:\\BaiduNetdiskDownload\\P\\v2_mod\\car\\2\\point_transformer_seg.pth')
pcg_state_dict = torch.load('E:\\BaiduNetdiskDownload\\P\\v2_mod\\car\\2\\point_cloud_generator.pth')

# 应用预训练权重

model.diffusion.model.load_state_dict(diffusion_state_dict)  # 可能的命名错误
model.model_feature.load_state_dict(model_feature_state_dict)
model.pcg.load_state_dict(pcg_state_dict)
# 确保模型在正确的设备上

model.to_device()

output_point_cloud = model.generate_point_cloud(model, 4)

output_point_cloud_np = output_point_cloud.detach().cpu().numpy()
pcwrite(
    filename="E:\\BaiduNetdiskDownload\\P\\chair1.ply",  # 输出的.ply文件位置
    xyz=output_point_cloud_np[0]       # 只取第一个点云
)
for j in range(output_point_cloud_np.shape[0]):
    xyz = output_point_cloud_np[j][:, :3]
    pcd = o3d.geometry.PointCloud()


    # 设置点云的坐标
    pcd.points = o3d.utility.Vector3dVector(xyz)
    o3d.visualization.draw_geometries([pcd])

# for j in range(output_point_cloud_np.shape[0]):
#     xyz = output_point_cloud_np[j][:, :3]
#     pcd = o3d.geometry.PointCloud()
#
#     # 设置点云的坐标
#     pcd.points = o3d.utility.Vector3dVector(xyz)
#
#     # 估计法线（BPA算法需要点云上的法线）
#     pcd.estimate_normals()
#
#     # 计算每个点的平均最近邻距离
#     distances = pcd.compute_nearest_neighbor_distance()
#     avg_dist = np.mean(distances)
#
#     # 使用3倍平均距离作为半径参数
#     radius =  2*avg_dist
#
#     # 创建球面枢轴三角网格
#     bpa_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
#         pcd,
#         radii=o3d.utility.DoubleVector([radius, radius * 2]))
#
#     # 可视化
#     o3d.visualization.draw_geometries([bpa_mesh])


# print(xyz)
# pcd = o3d.geometry.PointCloud()
#
# # 设置点云的坐标
# pcd.points = o3d.utility.Vector3dVector(xyz)
# o3d.visualization.draw_geometries([pcd])
