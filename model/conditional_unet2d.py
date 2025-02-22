import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
from einops.layers.torch import Rearrange
from model.conv2d_components import Downsample2d, Upsample2d, Conv2dBlock
from model.positional_embedding import SinusoidalPosEmb

class ConditionalResidualBlock2D(nn.Module):
    def __init__(self,
            in_channels,
            out_channels,
            cond_dim,
            kernel_size=3,
            n_groups=8,
            cond_predict_scale=False):
        super().__init__()

        self.blocks = nn.ModuleList([
            Conv2dBlock(in_channels, out_channels, kernel_size, n_groups=n_groups),
            Conv2dBlock(out_channels, out_channels, kernel_size, n_groups=n_groups)
        ])

        # FiLM modulation
        # predicts per-channel scale and bias
        cond_channels = out_channels
        if cond_predict_scale:
            cond_channels = cond_channels * 2
        self.cond_predict_scale = cond_predict_scale
        self.out_channels = out_channels
        self.cond_encoder = nn.Sequential(
            nn.Mish(),
            nn.Linear(cond_dim, cond_channels),
            Rearrange('b t -> b t 1 1')
        )

        # maker sure dimensions compatible
        self.residual_conv = nn.Conv2d(in_channels, out_channels, 1) \
            if in_channels != out_channels else nn.Identity()

    def forward(self, x, cond):
        """
        Args:
            x : [ B x in_channels x H x W ]
            cond : [ B x cond_dim ]
        Returns:
            out : [ B x out_channels x H x W]
        """
        out = self.blocks[0](x)
        embed = self.cond_encoder(cond)
        if self.cond_predict_scale:
            embed = embed.reshape(
                embed.shape[0], 2, self.out_channels, 1, 1)
            scale = embed[:, 0, ...]
            bias = embed[:, 1, ...]
            out = scale * out + bias
        else:
            out = out + embed
        out = self.blocks[1](out)
        out = out + self.residual_conv(x)
        return out

class ConditionalUnet2D(nn.Module):
    def __init__(self,
        input_dim,
        base_channels=16,
        global_cond_embed_dim=None,
        timestep_embed_dim=128,
        down_dims=[128, 256, 512],
        kernel_size=3,
        n_groups=8,
        cond_predict_scale=False
        ):
        super().__init__()
        all_dims = [base_channels] + down_dims
        start_dim = down_dims[0]

        self.conv_init = nn.Conv2d(input_dim, base_channels, kernel_size, padding=1)

        tsed = timestep_embed_dim
        self.timestep_encoder = nn.Sequential(
            SinusoidalPosEmb(tsed),
            nn.Linear(tsed, tsed * 4),
            nn.Mish(),
            nn.Linear(tsed * 4, tsed)
        )
        cond_dim = tsed
        if global_cond_embed_dim is not None:
            gced = global_cond_embed_dim
            self.global_cond_encoder = nn.Sequential(
                SinusoidalPosEmb(gced),
                nn.Linear(gced, gced * 4),
                nn.Mish(),
                nn.Linear(gced * 4, gced)
            )
            cond_dim += gced
        
        in_out = list(zip(all_dims[:-1], all_dims[1:]))

        down_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)
            down_modules.append(nn.ModuleList([
                ConditionalResidualBlock2D(
                    dim_in, dim_out, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale
                ),
                ConditionalResidualBlock2D(
                    dim_out, dim_out, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale
                ),
                Downsample2d(dim_out) if not is_last else nn.Identity()
            ]))

        mid_dim = all_dims[-1]
        self.mid_modules = nn.ModuleList([
            ConditionalResidualBlock2D(
                mid_dim, mid_dim, cond_dim=cond_dim,
                kernel_size=kernel_size, n_groups=n_groups,
                cond_predict_scale=cond_predict_scale
            ),
            ConditionalResidualBlock2D(
                mid_dim, mid_dim, cond_dim=cond_dim,
                kernel_size=kernel_size, n_groups=n_groups,
                cond_predict_scale=cond_predict_scale
            )
        ])
        self.down_modules = down_modules

        up_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (len(in_out) - 1)
            up_modules.append(nn.ModuleList([
                ConditionalResidualBlock2D(
                    dim_out*2, dim_in, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale
                ),
                ConditionalResidualBlock2D(
                    dim_in, dim_in, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale
                ),
                Upsample2d(dim_in) if not is_last else nn.Identity()
            ]))
        self.up_modules = up_modules

        self.conv_final = nn.Sequential(
            Conv2dBlock(start_dim, start_dim, kernel_size=kernel_size),
            nn.Conv2d(start_dim, input_dim, 1)
        )

        # self.global_cond_embed_dim = global_cond_embed_dim


    def forward(self,
        sample: torch.Tensor,
        timestep: torch.Tensor,
        global_cond=None):
        """
        Args:
            sample: [B, C, H, W]
            timestep: [B]
            global_cond: [B]
        Returns:
            x: [B, C, H, W]
        """
        x = self.conv_init(sample)

        # 1. time
        global_feature = self.timestep_encoder(timestep)

        # 2. global condition
        if global_cond is not None:
            global_cond = self.global_cond_encoder(global_cond)
            global_feature = torch.cat([global_feature, global_cond], axis=-1)
        
        # 3. Down Sample Layers
        h = []
        for idx, (resnet, resnet2, downsample) in enumerate(self.down_modules):
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            h.append(x)
            x = downsample(x)

        # 4. Mid Layers
        for mid_module in self.mid_modules:
            x = mid_module(x, global_feature)
        
        # 5. Up Sample Layers
        for ixd, (resnet, resnet2, upsample) in enumerate(self.up_modules):
            x = torch.cat([x, h.pop()], dim=1)
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            x = upsample(x)
        
        x = self.conv_final(x)

        return x
