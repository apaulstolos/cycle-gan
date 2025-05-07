import torch
import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self, in_channels=3, num_features=64):
        super().__init__()

        self.initial_block = nn.Sequential(
            nn.Conv2d(in_channels, num_features, kernel_size=4, stride=2, padding=1, padding_mode="reflect"),
            nn.InstanceNorm2d(num_features),
            nn.LeakyReLU(0.2),

            nn.Conv2d(num_features, num_features * 2, kernel_size=4, stride=2, padding=1),  
            nn.InstanceNorm2d(num_features * 2),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(num_features * 2, num_features * 4, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(num_features * 4),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(num_features * 4, num_features * 8, kernel_size=4, stride=1, padding=1),
            nn.InstanceNorm2d(num_features * 8),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(num_features * 8, 1, kernel_size=4, stride=1, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.initial_block(x)
    