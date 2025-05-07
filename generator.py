import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, img_channels=3, num_features=64, num_residual_blocks=9):
        super().__init__()

        self.initial_block = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(img_channels, num_features, kernel_size=7, stride=1),
            nn.InstanceNorm2d(num_features),
            nn.ReLU(inplace=True)
        )

        self.down_sampling = nn.Sequential(
            nn.Conv2d(num_features, num_features * 2, kernel_size=3, stride=2, padding=1, padding_mode="reflect"),
            nn.InstanceNorm2d(num_features * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_features * 2, num_features * 4, kernel_size=3, stride=2, padding=1, padding_mode="reflect"),
            nn.InstanceNorm2d(num_features * 4),
            nn.ReLU(inplace=True)
        )

        res_blocks = []
        for _ in range(num_residual_blocks):
            res_blocks.append(nn.Sequential(
                nn.Conv2d(num_features * 4, num_features * 4, kernel_size=3, stride=1, padding=1, padding_mode="reflect"),
                nn.InstanceNorm2d(num_features * 4),
                nn.ReLU(inplace=True),
                nn.Conv2d(num_features * 4, num_features * 4, kernel_size=3, stride=1, padding=1, padding_mode="reflect"),
                nn.InstanceNorm2d(num_features * 4)
            ))
        self.residual_blocks = nn.ModuleList(res_blocks)

        self.up_sampling = nn.Sequential(
            nn.ConvTranspose2d(num_features * 4, num_features * 2, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(num_features * 2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(num_features * 2, num_features, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(num_features),
            nn.ReLU(inplace=True)
        )

        self.final = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(num_features, img_channels, kernel_size=7, stride=1),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.initial_block(x)
        x = self.down_sampling(x)
        for res_block in self.residual_blocks:
            x = x + res_block(x)
        x = self.up_sampling(x)
        x = self.final(x)
        return x
