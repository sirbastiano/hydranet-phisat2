#!/usr/bin/env python3
"""
Model Analysis Tools for HydraNet
Provides utilities to decompose and analyze model architectures.
"""

import torch.nn as nn
from typing import Dict, List, Tuple
from collections import OrderedDict


class ModelComponents:
    """Container for model components (encoder, bottleneck, decoder)."""
    
    def __init__(self, encoder: nn.ModuleDict, bottleneck: nn.ModuleDict, decoder: nn.ModuleDict):
        """
        Initialize ModelComponents.
        
        Args:
            encoder: Dictionary of encoder layers
            bottleneck: Dictionary of bottleneck layers
            decoder: Dictionary of decoder layers
        """
        self.encoder = encoder
        self.bottleneck = bottleneck
        self.decoder = decoder
    
    def __repr__(self) -> str:
        """String representation of model components."""
        encoder_params = sum(p.numel() for module in self.encoder.values() for p in module.parameters())
        bottleneck_params = sum(p.numel() for module in self.bottleneck.values() for p in module.parameters())
        decoder_params = sum(p.numel() for module in self.decoder.values() for p in module.parameters())
        total = encoder_params + bottleneck_params + decoder_params
        
        return (
            f"ModelComponents(\n"
            f"  ENCODER: {len(self.encoder)} layer groups, {encoder_params:,} parameters ({encoder_params/total*100:.1f}%)\n"
            f"  BOTTLENECK: {len(self.bottleneck)} layer groups, {bottleneck_params:,} parameters ({bottleneck_params/total*100:.1f}%)\n"
            f"  DECODER: {len(self.decoder)} layer groups, {decoder_params:,} parameters ({decoder_params/total*100:.1f}%)\n"
            f"  TOTAL: {total:,} parameters\n"
            f")"
        )
    
    def get_summary(self) -> Dict[str, Dict]:
        """
        Get detailed summary of each component.
        
        Returns:
            Dictionary with detailed statistics for each component
        """
        def component_stats(module_dict):
            stats = {
                'num_layer_groups': len(module_dict),
                'layer_names': list(module_dict.keys()),
                'parameters': sum(p.numel() for module in module_dict.values() for p in module.parameters()),
                'trainable_parameters': sum(p.numel() for module in module_dict.values() for p in module.parameters() if p.requires_grad)
            }
            return stats
        
        return {
            'encoder': component_stats(self.encoder),
            'bottleneck': component_stats(self.bottleneck),
            'decoder': component_stats(self.decoder)
        }


def split_model(model: nn.Module) -> ModelComponents:
    """
    Split a PhisatNet/HydraNet model into ENCODER, BOTTLENECK, and DECODER components.
    
    The function analyzes layer names and groups them into three categories:
    - ENCODER: All encoder layers and pooling layers before bottleneck
    - BOTTLENECK: The central bottleneck layer
    - DECODER: All upsampling, decoder layers, and final convolution
    
    Args:
        model: A PhisatNet or compatible model instance
        
    Returns:
        ModelComponents: Object containing encoder, bottleneck, and decoder as ModuleDicts
        
    Example:
        >>> model = hydranet.load_student(preset='checkpoint')
        >>> components = hydranet.split_model(model)
        >>> print(components)
        >>> print(f"Encoder layers: {list(components.encoder.keys())}")
    """
    encoder_layers = OrderedDict()
    bottleneck_layers = OrderedDict()
    decoder_layers = OrderedDict()
    
    # Iterate through all named modules in the model
    for name, module in model.named_children():
        # Skip empty modules
        if not list(module.children()) and not list(module.parameters()):
            continue
        
        # Categorize based on layer name
        if 'encoder' in name.lower():
            encoder_layers[name] = module
        elif 'pool' in name.lower() and 'encoder' not in name.lower():
            # Pools that are part of encoder structure
            encoder_layers[name] = module
        elif 'bottleneck' in name.lower():
            bottleneck_layers[name] = module
        elif any(keyword in name.lower() for keyword in ['decoder', 'upsampl', 'final']):
            decoder_layers[name] = module
        else:
            # For any ambiguous layers, try to infer from position
            # This handles edge cases where naming might be non-standard
            if not bottleneck_layers:
                # If we haven't seen bottleneck yet, it's likely encoder
                encoder_layers[name] = module
            else:
                # After bottleneck, it's decoder
                decoder_layers[name] = module
    
    return ModelComponents(
        encoder=nn.ModuleDict(encoder_layers),
        bottleneck=nn.ModuleDict(bottleneck_layers),
        decoder=nn.ModuleDict(decoder_layers)
    )


def print_model_structure(model: nn.Module, max_depth: int = 2) -> None:
    """
    Print the hierarchical structure of the model.
    
    Args:
        model: PyTorch model
        max_depth: Maximum depth to print (default: 2)
    """
    def print_recursive(module, prefix="", depth=0):
        if depth > max_depth:
            return
        
        for name, child in module.named_children():
            num_params = sum(p.numel() for p in child.parameters())
            trainable = sum(p.numel() for p in child.parameters() if p.requires_grad)
            
            print(f"{prefix}├── {name}: {child.__class__.__name__} ({num_params:,} params, {trainable:,} trainable)")
            
            if depth < max_depth:
                print_recursive(child, prefix + "│   ", depth + 1)
    
    print(f"\nModel Structure (depth={max_depth}):")
    print(f"{model.__class__.__name__}")
    print_recursive(model)


def get_layer_names(model: nn.Module) -> Dict[str, List[str]]:
    """
    Get all layer names categorized by their type.
    
    Args:
        model: PyTorch model
        
    Returns:
        Dictionary with layer names grouped by category
    """
    components = split_model(model)
    
    return {
        'encoder': list(components.encoder.keys()),
        'bottleneck': list(components.bottleneck.keys()),
        'decoder': list(components.decoder.keys())
    }


if __name__ == '__main__':
    import torch
    from .loading import load_student
    
    print("=== Model Splitting Example ===\n")
    
    # Create a simple test model
    print("Loading model...")
    model = load_student(preset='checkpoint', auto_load_weights=False)
    
    print("\n1. Splitting model into components...")
    components = split_model(model)
    print(components)
    
    print("\n2. Detailed summary:")
    summary = components.get_summary()
    for component_name, stats in summary.items():
        print(f"\n{component_name.upper()}:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
    
    print("\n3. Layer names:")
    layer_names = get_layer_names(model)
    for category, names in layer_names.items():
        print(f"\n{category.upper()}: {names}")
    
    print("\n4. Model structure:")
    print_model_structure(model, max_depth=1)
