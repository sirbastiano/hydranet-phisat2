#!/usr/bin/env python3
"""
PhisatNet - U-Net architecture optimized for Myriad 2 VPU inference
Designed for satellite imagery processing with configurable depth and channel multipliers.
"""

import torch
import torch.nn as nn
from typing import List, Dict, Any


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample when applied in main path of residual blocks."""
    
    def __init__(self, drop_prob: float = 0.0):
        """
        Initialize DropPath.
        
        Args:
            drop_prob (float): Drop probability. Defaults to 0.0.
        """
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x (torch.Tensor): Input tensor.
            
        Returns:
            torch.Tensor: Output tensor.
        """
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # binarize
        output = x.div(keep_prob) * random_tensor
        return output


class ConvNeXtBlock(nn.Module):
    """ConvNeXt block with depthwise convolution, batch norm, and inverted bottleneck."""
    
    def __init__(self, dim: int, drop_path: float = 0.0, layer_scale_init_value: float = 1e-6):
        """
        Initialize ConvNeXtBlock.
        
        Args:
            dim (int): Number of input/output channels.
            drop_path (float): Drop path rate. Defaults to 0.0.
            layer_scale_init_value (float): Layer scale initialization value. Defaults to 1e-6.
        """
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)  # depthwise conv
        self.norm = nn.BatchNorm2d(dim)
        self.pwconv1 = nn.Conv2d(dim, 4 * dim, kernel_size=1)  # pointwise/1x1 convs
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv2d(4 * dim, dim, kernel_size=1)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), 
                                requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x (torch.Tensor): Input tensor.
            
        Returns:
            torch.Tensor: Output tensor.
        """
        input_tensor = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma.view(1, -1, 1, 1) * x
        
        x = input_tensor + self.drop_path(x)
        return x


class ConvBlock(nn.Module):
    """ConvNeXt-style block with BatchNorm for improved performance."""
    
    def __init__(self, in_channels: int, out_channels: int, drop_path: float = 0.0):
        """
        Initialize ConvBlock with ConvNeXt architecture using BatchNorm.
        
        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            drop_path (float): Drop path rate. Defaults to 0.0.
        """
        super().__init__()
        
        # If input and output channels differ, use a 1x1 conv to match dimensions
        if in_channels != out_channels:
            self.channel_proj = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.channel_proj = nn.Identity()
        
        # ConvNeXt block
        self.convnext_block = ConvNeXtBlock(out_channels, drop_path=drop_path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x (torch.Tensor): Input tensor.
            
        Returns:
            torch.Tensor: Output tensor.
        """
        x = self.channel_proj(x)
        x = self.convnext_block(x)
        return x


class PhisatNet(nn.Module):
    """
    PhisatNet - U-Net architecture optimized for Myriad 2 VPU compatibility.
    
    Features:
    - Encoder-decoder architecture with skip connections
    - ConvNeXt-style blocks for efficient feature extraction
    - Configurable depth and channel multipliers
    - BatchNorm for VPU compatibility
    - Designed for satellite imagery segmentation
    """
    
    def __init__(self, 
                 n_channels: int = 8, 
                 n_classes: int = 3, 
                 base_filters: int = 32,
                 depth: int = 3,
                 channel_multipliers: List[int] = None):
        """
        Initialize PhisatNet.
        
        Args:
            n_channels (int): Number of input channels. Defaults to 8 (multi-spectral).
            n_classes (int): Number of output classes. Defaults to 3.
            base_filters (int): Base number of filters. Defaults to 32.
            depth (int): Number of encoder/decoder levels. Defaults to 3.
            channel_multipliers (List[int]): List of multipliers for each level. 
                                            If None, uses [1, 2, 4, 8, ...]. 
                                            Length should be depth + 1 (including bottleneck).
        """
        super().__init__()
        
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.base_filters = base_filters
        self.depth = depth
        
        # Set default channel multipliers if not provided
        if channel_multipliers is None:
            channel_multipliers = [2**i for i in range(depth + 1)]
        
        if len(channel_multipliers) != depth + 1:
            raise ValueError(f'channel_multipliers length ({len(channel_multipliers)}) must equal depth + 1 ({depth + 1})')
        
        self.channel_multipliers = channel_multipliers
        
        # Calculate channel dimensions for each level
        self.channels = [base_filters * mult for mult in channel_multipliers]
        
        # Encoder
        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()
        
        # First encoder block
        self.encoders.append(ConvBlock(n_channels, self.channels[0]))
        
        # Remaining encoder blocks with pooling
        for i in range(depth - 1):
            self.pools.append(nn.MaxPool2d(2))
            self.encoders.append(ConvBlock(self.channels[i], self.channels[i + 1]))
        
        # Final pooling before bottleneck
        self.pools.append(nn.MaxPool2d(2))
        
        # Bottleneck
        self.bottleneck = ConvBlock(self.channels[depth - 1], self.channels[depth])
        
        # Decoder
        self.upsamplers = nn.ModuleList()
        self.decoders = nn.ModuleList()
        
        # Create decoder blocks
        for i in range(depth):
            # Upsampler from higher level
            up_in_channels = self.channels[depth - i]
            up_out_channels = self.channels[depth - i]
            self.upsamplers.append(
                nn.ConvTranspose2d(up_in_channels, up_out_channels, kernel_size=2, stride=2)
            )
            
            # Decoder block (concatenated channels + encoder skip connection)
            dec_in_channels = self.channels[depth - i] + self.channels[depth - i - 1]
            dec_out_channels = self.channels[depth - i - 1]
            self.decoders.append(ConvBlock(dec_in_channels, dec_out_channels))
        
        # Output head
        self.final_conv = nn.Conv2d(self.channels[0], n_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch, n_channels, height, width).
            
        Returns:
            torch.Tensor: Output tensor of shape (batch, n_classes, height, width).
        """
        # Store encoder outputs for skip connections
        encoder_outputs = []
        
        # Encoder path
        current = x
        for i in range(self.depth):
            current = self.encoders[i](current)
            encoder_outputs.append(current)
            if i < self.depth - 1:  # Don't pool after last encoder
                current = self.pools[i](current)
        
        # Final pooling and bottleneck
        current = self.pools[-1](current)
        current = self.bottleneck(current)
        
        # Decoder path
        for i in range(self.depth):
            # Upsample
            current = self.upsamplers[i](current)
            
            # Concatenate with skip connection
            skip_connection = encoder_outputs[self.depth - 1 - i]
            current = torch.cat([current, skip_connection], dim=1)
            
            # Decoder block
            current = self.decoders[i](current)
        
        # Final output
        out = self.final_conv(current)
        return out

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the model architecture.
        
        Returns:
            dict: Model information including depth, channels, and parameters.
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'depth': self.depth,
            'base_filters': self.base_filters,
            'channels_per_level': self.channels,
            'channel_multipliers': self.channel_multipliers,
            'n_channels': self.n_channels,
            'n_classes': self.n_classes,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_size_mb': sum(p.numel() * p.element_size() for p in self.parameters()) / 1024**2
        }

    def __repr__(self) -> str:
        """String representation of the model."""
        info = self.get_model_info()
        return (f"PhisatNet(\n"
                f"  depth={info['depth']},\n"
                f"  base_filters={info['base_filters']},\n"
                f"  channels={info['channels_per_level']},\n"
                f"  parameters={info['total_parameters']:,},\n"
                f"  size={info['model_size_mb']:.2f} MB\n"
                f")")


def create_phisatnet(config: str = 'default') -> PhisatNet:
    """
    Factory function to create PhisatNet with predefined configurations.
    
    Args:
        config (str): Configuration preset. Options:
            - 'default': Standard configuration
            - 'small': Smaller model for fast inference
            - 'large': Larger model for better accuracy
            - 'myriad_optimized': Optimized for Myriad 2 VPU
    
    Returns:
        PhisatNet: Configured model instance.
    """
    configs = {
        'default': {
            'n_channels': 8,
            'n_classes': 3,
            'base_filters': 32,
            'depth': 3,
            'channel_multipliers': [1, 2, 2, 3]
        },
        'small': {
            'n_channels': 8,
            'n_classes': 3,
            'base_filters': 16,
            'depth': 3,
            'channel_multipliers': [1, 1, 1, 1]
        },
        'large': {
            'n_channels': 8,
            'n_classes': 3,
            'base_filters': 64,
            'depth': 3,
            'channel_multipliers': [1, 2, 3, 4]
        },
        'myriad_optimized': {
            'n_channels': 8,
            'n_classes': 3,
            'base_filters': 48,
            'depth': 3,
            'channel_multipliers': [1, 2, 2, 3]
        }
    }
    
    if config not in configs:
        raise ValueError(f"Unknown config: {config}. Available: {list(configs.keys())}")
    
    return PhisatNet(**configs[config])


if __name__ == '__main__':
    # Example usage
    print("=== PhisatNet Model Examples ===\n")
    
    # Create different model configurations
    configs_to_test = ['default', 'small', 'large', 'myriad_optimized']
    
    for config_name in configs_to_test:
        print(f"Configuration: {config_name}")
        model = create_phisatnet(config_name)
        print(model)
        print()
    
    # Test forward pass
    print("=== Testing Forward Pass ===")
    model = create_phisatnet('default')
    model.eval()
    
    # Create dummy input (batch_size=1, channels=8, height=256, width=256)
    dummy_input = torch.randn(1, 8, 256, 256)
    
    with torch.no_grad():
        output = model(dummy_input)
    
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"\nModel ready for inference!")
