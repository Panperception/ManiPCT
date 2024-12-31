import torch
import torch.nn as nn
import torch.nn.functional as F


class PointCloudGenerator(nn.Module):
    def __init__(self, input_dim=1024, num_output_points=5000, output_dim=3):
        super(PointCloudGenerator, self).__init__()
        self.num_output_points = num_output_points
        self.output_dim = output_dim

        self.fc1 = nn.Linear(input_dim, 512)
        self.fc2 = nn.Linear(512, 1024)
        self.fc3 = nn.Linear(1024, 2048)

        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(1024)
        self.bn3 = nn.BatchNorm1d(2048)

        
        self.deconv1 = nn.ConvTranspose1d(2048, 1024, kernel_size=1)
        self.deconv2 = nn.ConvTranspose1d(1024, 512, kernel_size=1)
        self.deconv3 = nn.ConvTranspose1d(512, output_dim, kernel_size=num_output_points)

        self.bn4 = nn.BatchNorm1d(1024)
        self.bn5 = nn.BatchNorm1d(512)

    def forward(self, x, t):
        batch_size = x.size(0)

        
        x = x + t

        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = F.relu(self.bn3(self.fc3(x)))

        
        x = x.unsqueeze(2)

        x = F.relu(self.bn4(self.deconv1(x)))
        x = F.relu(self.bn5(self.deconv2(x)))

        x = self.deconv3(x)

        # output size(batch_size, num_output_points, output_dim)
        x = x[:, :, :self.num_output_points]

        x = x.transpose(2, 1).contiguous()
        x = x.view(batch_size, self.num_output_points, self.output_dim)
        return x
