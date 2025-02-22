import torch
import torch.nn as nn
import torch.nn.functional as F


class Downsample2d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim, 3, 2, 1)
    
    def forward(self, x):
        return self.conv(x)
    
class Upsample2d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.ConvTranspose2d(dim, dim, 4, 2, 1)
    
    def forward(self, x):
        return self.conv(x)
    
class Conv2dBlock(nn.Module):
    '''
        Conv2d -> GroupNorm -> Mish
    '''
    def __init__(self, in_channels, out_channels, kernel_size, n_groups=8):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size // 2),
            nn.GroupNorm(n_groups, out_channels),
            nn.Mish()
        )

    def forward(self, x):
        return self.block(x)
