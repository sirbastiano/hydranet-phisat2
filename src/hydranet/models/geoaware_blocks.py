import torch
import torch.nn as nn

from .util_tools import SE_Block, get_activation, get_normalization


# class CoreCNNBlock(nn.Module):
#     def __init__(self, in_channels, out_channels, *, norm="batch", 
#                  activation="relu", residual=True, activation_out=None):
#         super(CoreCNNBlock, self).__init__()
        
#         self.activation = get_activation(activation)
#         self.activation_out = self.activation if activation_out is None else get_activation(activation_out)
#         self.residual = residual
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.squeeze = SE_Block(self.out_channels, 16)

#         self.conv1 = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1, padding=0)
#         self.norm1 = get_normalization(norm, self.out_channels)

#         self.conv2 = nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, padding=1, groups=self.out_channels)
#         self.norm2 = get_normalization(norm, self.out_channels)
        
#         self.conv3 = nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, padding=1,  groups=1)
#         self.norm3 = get_normalization(norm, self.out_channels)

#         if self.residual:
#             if in_channels != out_channels:
#                 self.match_channels = nn.Sequential(
#                     nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, bias=False),
#                     get_normalization(norm, out_channels),
#                 )
#             else:
#                 self.match_channels = nn.Sequential(
#                     nn.Identity(),
#                     get_normalization(norm, out_channels)
#                 )

#     def forward(self, x):
#         identity = x
#         x = self.activation(self.norm1(self.conv1(x)))
#         x = self.activation(self.norm2(self.conv2(x)))
#         x = self.norm3(self.conv3(x))

#         x = self.squeeze(x)

#         if self.residual:
#             x = x + self.match_channels(identity)

#         x = self.activation_out(x)
#         return x

class CoreCNNBlock(nn.Module):
    """Simple convolutional block with normalization and activation.
    
    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        norm (str): Normalization type. Defaults to 'batch'.
        activation (str): Activation function. Defaults to 'relu'.
        residual (bool): Whether to use residual connection. Defaults to True.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        norm: str = 'batch',
        activation: str = 'relu',
        residual: bool = True
    ) -> None:
        """Initialize CoreCNNBlock."""
        super().__init__()
        
        self.activation = get_activation(activation)
        self.residual = residual
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm1 = get_normalization(norm, out_channels)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = get_normalization(norm, out_channels)

        if self.residual and in_channels != out_channels:
            self.match_channels = nn.Conv2d(
                in_channels, out_channels, 
                kernel_size=1, padding=0
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W).
        
        Returns:
            torch.Tensor: Output tensor of shape (B, out_channels, H, W).
        """
        identity = x
        
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.activation(x)
        
        x = self.conv2(x)
        x = self.norm2(x)

        if self.residual:
            if self.in_channels != self.out_channels:
                identity = self.match_channels(identity)
            x = x + identity

        x = self.activation(x)
        return x


class CoreAttentionBlock(nn.Module):
    def __init__(self,
        lower_channels,
        higher_channels, *,
        norm="batch",
        activation="relu",
    ):
        super(CoreAttentionBlock, self).__init__()

        self.lower_channels = lower_channels
        self.higher_channels = higher_channels
        self.activation = get_activation(activation)
        self.norm = norm
        self.expansion = 4
        self.reduction = 4

        if self.lower_channels != self.higher_channels:
            self.match = nn.Sequential(
                nn.Conv2d(self.higher_channels, self.lower_channels, kernel_size=1, padding=0, bias=False),
                get_normalization(self.norm, self.lower_channels),
            )

        self.compress = nn.Conv2d(self.lower_channels, 1, kernel_size=1, padding=0)
        self.sigmoid = nn.Sigmoid()

        self.attn_c_pool = nn.AdaptiveAvgPool2d(self.reduction)
        self.attn_c_reduction = nn.Linear(self.lower_channels * (self.reduction ** 2), self.lower_channels * self.expansion)
        self.attn_c_extention = nn.Linear(self.lower_channels * self.expansion, self.lower_channels)

    def forward(self, x, skip):
        if x.size(1) != skip.size(1):
            x = self.match(x)
        x = x + skip
        x = self.activation(x)

        attn_spatial = self.compress(x)
        attn_spatial = self.sigmoid(attn_spatial)

        attn_channel = self.attn_c_pool(x)
        attn_channel = attn_channel.reshape(attn_channel.size(0), -1)
        attn_channel = self.attn_c_reduction(attn_channel)
        attn_channel = self.activation(attn_channel)
        attn_channel = self.attn_c_extention(attn_channel)
        attn_channel = attn_channel.reshape(x.size(0), x.size(1), 1, 1)
        attn_channel = self.sigmoid(attn_channel)

        return attn_spatial, attn_channel


