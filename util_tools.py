import numpy as np
import torch
import torch.nn as nn

def make_bilinear_upsample(in_channels: int):
    """
    Returns a ConvTranspose2d layer that does a fixed 2× bilinear upsampling
    for 'in_channels' channels without cross-channel mixing.
    """
    layer = nn.ConvTranspose2d(
        in_channels  = in_channels,
        out_channels = in_channels,
        kernel_size  = 4,
        stride       = 2,
        padding      = 1,
        groups       = in_channels,
        bias         = False
    )

    def bilinear_filter_2d(kernel_size: int) -> np.ndarray:
        factor = (kernel_size + 1) // 2
        if kernel_size % 2 == 1:
            center = factor - 1
        else:
            center = factor - 0.5

        og = np.ogrid[:kernel_size, :kernel_size]
        filt = (1 - abs(og[0] - center)/factor) * (1 - abs(og[1] - center)/factor)
        return filt

    kernel_size = layer.kernel_size[0]
    filt_2d = bilinear_filter_2d(kernel_size)

    w = np.zeros((in_channels, 1, kernel_size, kernel_size), dtype=np.float32)
    for c in range(in_channels):
        w[c, 0, :, :] = filt_2d

    with torch.no_grad():
        layer.weight.copy_(torch.from_numpy(w))

    return layer


# class SE_Block(nn.Module):
#     "credits: https://github.com/moskomule/senet.pytorch/blob/master/senet/se_module.py#L4"
#     def __init__(self, channels, reduction=16):
#         super().__init__()
#         self.reduction = reduction
#         self.squeeze = nn.AdaptiveAvgPool2d(1)
#         self.excitation = nn.Sequential(
#             nn.Linear(channels, max(1, channels // self.reduction), bias=False),
#             nn.GELU(),
#             nn.Linear(max(1, channels // self.reduction), channels, bias=False),
#             nn.Sigmoid()
#         )

#     def forward(self, x):
#         bs, c, _, _ = x.shape
#         y = self.squeeze(x).view(bs, c)

#         if not torch.isfinite(y).all():
#             print("Found NaNs or Infs in squeeze output")

#         y = self.excitation(y).view(bs, c, 1, 1)

#         return x * y.expand_as(x)

class SE_Block(nn.Module):
    """Squeeze-and-Excitation block for channel attention.
    
    Credits: https://github.com/moskomule/senet.pytorch/blob/master/senet/se_module.py#L4
    
    Args:
        channels (int): Number of input channels.
        reduction (int): Reduction ratio for the bottleneck. Defaults to 16.
    """
    
    def __init__(self, channels: int, reduction: int = 16) -> None:
        """Initialize the SE block.
        
        Args:
            channels (int): Number of input channels.
            reduction (int): Reduction ratio for the bottleneck. Defaults to 16.
        """
        super().__init__()
        reduced_channels = max(1, channels // reduction)
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, reduced_channels, bias=False),
            nn.GELU(),
            nn.Linear(reduced_channels, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the SE block.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W).
        
        Returns:
            torch.Tensor: Output tensor with channel attention applied.
        """
        bs, c, h, w = x.shape
        y = self.squeeze(x)
        y = y.view(bs, c)
        y = self.excitation(y)
        y = y.view(bs, c, 1, 1)
        return x * y


def get_activation(activation_name):
    if activation_name == "relu":
        return nn.ReLU(inplace=False)
    elif isinstance(activation_name, torch.nn.modules.activation.ReLU6):
        return activation_name

    elif activation_name == "gelu":
        return nn.GELU()
    elif isinstance(activation_name, torch.nn.modules.activation.GELU):
        return activation_name

    elif activation_name == "leaky_relu":
        return nn.LeakyReLU(inplace=False)
    elif isinstance(activation_name, torch.nn.modules.activation.LeakyReLU):
        return activation_name

    elif activation_name == "prelu":
        return nn.PReLU()
    elif isinstance(activation_name, torch.nn.modules.activation.PReLU):
        return activation_name

    elif activation_name == "selu":
        return nn.SELU(inplace=False)
    elif isinstance(activation_name, torch.nn.modules.activation.SELU):
        return activation_name

    elif activation_name == "sigmoid":
        return nn.Sigmoid()
    elif isinstance(activation_name, torch.nn.modules.activation.Sigmoid):
        return activation_name

    elif activation_name == "tanh":
        return nn.Tanh()
    elif isinstance(activation_name, torch.nn.modules.activation.Tanh):
        return activation_name

    elif activation_name == "mish":
        return nn.Mish()
    elif isinstance(activation_name, torch.nn.modules.activation.Mish):
        return activation_name
    else:
        raise ValueError(f"activation must be one of leaky_relu, prelu, selu, gelu, sigmoid, tanh, relu. Got: {activation_name}")


def get_normalization(normalization_name, num_channels, img_size=None, num_groups=32, dims=2):
    if normalization_name == "batch":
        if dims == 1:
            return nn.BatchNorm1d(num_channels)
        elif dims == 2:
            return nn.BatchNorm2d(num_channels)
        elif dims == 3:
            return nn.BatchNorm3d(num_channels)
    elif normalization_name == "instance":
        if dims == 1:
            return nn.InstanceNorm1d(num_channels)
        elif dims == 2:
            return nn.InstanceNorm2d(num_channels)
        elif dims == 3:
            return nn.InstanceNorm3d(num_channels)
    elif normalization_name == "layer":
        return nn.LayerNorm(num_channels, img_size[0], img_size[1])
    elif normalization_name == "group":
        num_groups = min(num_groups, num_channels)
        while num_channels % num_groups != 0:
            num_groups -= 1
        return nn.GroupNorm(num_groups=num_groups, num_channels=num_channels)
    elif normalization_name == "bcn":
        if dims == 1:
            return nn.Sequential(
                nn.BatchNorm1d(num_channels),
                nn.GroupNorm(1, num_channels)
            )
        elif dims == 2:
            return nn.Sequential(
                nn.BatchNorm2d(num_channels),
                nn.GroupNorm(1, num_channels)
            )
        elif dims == 3:
            return nn.Sequential(
                nn.BatchNorm3d(num_channels),
                nn.GroupNorm(1, num_channels)
            )    
    elif normalization_name == "none":
        return nn.Identity()
    else:
        raise ValueError(f"normalization must be one of batch, instance, layer, group, none. Got: {normalization_name}")
