import torch
import torch.nn as nn

class ConditionalResidualBlock2D(nn.Module):
    def __init__(self,
        in_channels,
        out_channels,
        cond_dim,
        kernel_size=3,
        n_groups=8,
        cond_predict_scale=False):
        super().__init__()

    def forward(self, x, cond):
        """
        Args:
            x : [ B x in_channels x H x W ]
            cond : [ B x cond_dim ]
        Returns:
            out : [ B x out_channels x H x W]
        """
        pass

class ConditionalUnet2D(nn.Module):
    def __init__(self,
        input_dim,
        base_channels=16,
        global_cond_dim=None,
        step_embed_dim=256,
        down_dims=[256,512,1024],
        kernel_size=3,
        n_groups=8,
        cond_predict_scale=False
        ):
        super().__init__()

    def forward(self,
        sample: torch.Tensor,
        timestep: torch.Tensor,
        global_cond=None):
        pass
