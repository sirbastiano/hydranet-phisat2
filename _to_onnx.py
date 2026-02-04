#!/usr/bin/env python3
"""
Minimal script to generate PhiSatNet model with latest ONNX compatibility improvements.

This script creates an ONNX-compatible model with the following fixes:
- TracerWarnings resolved (dynamic conditionals)
- MaxPool dilations errors fixed
- Opset 11 compatibility for OpenVINO/Myriad deployment
"""

import sys
import os
import torch
import numpy as np
import logging
from datetime import datetime
from pathlib import Path

from hydranet_teacher import PhiSatNetDownstream


def setup_logging(log_file_path: str) -> logging.Logger:
    """
    Setup logging to both console and file.
    
    Args:
        log_file_path (str): Path to the log file
        
    Returns:
        logging.Logger: Configured logger instance
    """
    logger = logging.getLogger('phisatnet_generator')
    logger.setLevel(logging.INFO)
    
    # Clear any existing handlers
    logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler
    file_handler = logging.FileHandler(log_file_path, mode='w')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger


def clear_module_cache(modules_to_reload: list[str]) -> None:
    """Clear module cache to force reload of updated modules."""
    for module_name in modules_to_reload:
        if module_name in sys.modules:
            del sys.modules[module_name]
    
    # Remove any modules containing these names
    modules_to_remove = [
        name for name in sys.modules 
        if any(mod in name for mod in modules_to_reload)
    ]
    
    for module_name in modules_to_remove:
        del sys.modules[module_name]


def export_model_to_onnx(model: torch.nn.Module, 
                        output_path: str, 
                        dummy_input: torch.Tensor,
                        opset_version: int = 11,
                        logger: logging.Logger = None) -> None:
    """
    Export PyTorch model to ONNX format with diagnostics.
    
    Args:
        model (torch.nn.Module): PyTorch model to export
        output_path (str): Path to save the ONNX model
        dummy_input (torch.Tensor): Sample input tensor
        opset_version (int): ONNX opset version
        logger (logging.Logger): Logger instance for output
    """
    if logger is None:
        logger = logging.getLogger(__name__)
        
    logger.info(f'Starting ONNX export with opset version {opset_version}')
    logger.info(f'Model input shape: {dummy_input.shape}')
    
    # Set model to evaluation mode
    model.eval()
    
    # Test forward pass first
    with torch.no_grad():
        test_output = model(dummy_input)
    logger.info(f'Forward pass successful. Output shape: {test_output.shape}')
    
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names=['input'],
        output_names=['output'],
        opset_version=opset_version,
        do_constant_folding=True,
        keep_initializers_as_inputs=False,
        export_params=True,
        verbose=False,
        dynamic_axes=None
    )
    
    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    logger.info(f'✅ Model exported to {output_path}')
    logger.info(f'✅ ONNX model size: {file_size_mb:.2f} MB')


def create_sample_input(shape: tuple, output_path: str, logger: logging.Logger = None) -> np.ndarray:
    """
    Create normalized sample input for testing.
    
    Args:
        shape (tuple): Shape of the input tensor
        output_path (str): Path to save the sample input
        logger (logging.Logger): Logger instance for output
        
    Returns:
        np.ndarray: Generated sample input array
    """
    if logger is None:
        logger = logging.getLogger(__name__)
        
    sample_input = torch.randn(*shape).numpy().astype(np.float32)
    
    # Normalize to satellite imagery range
    sample_input = (sample_input - sample_input.mean()) / sample_input.std()
    sample_input = sample_input * 0.2 + 0.1  # Scale to [~-0.5, ~0.7] range
    
    np.save(output_path, sample_input)
    logger.info(f'✅ Sample input saved to: {output_path}')
    return sample_input


def main():
    """Main function to generate the ONNX-compatible model."""
    # Create output directories first
    output_dir = Path('/home/philab/Desktop/hydranet/onnx')
    output_dir.mkdir(exist_ok=True)
    
    # Setup logging
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file_path = output_dir / f'phisatnet_generation_{timestamp}.log'
    logger = setup_logging(str(log_file_path))
    
    logger.info("=== PhiSatNet Model Generator with Latest Improvements ===")
    logger.info(f"Log file: {log_file_path}")
    
    # Model configuration
    config = {
        'input_dim': 8,
        'output_dim': 3,
        'img_size': 128,
        'depths': [2, 2, 2, 2],  # Optimized for small model
        'dims': [8, 16, 24, 48],
        'activation': 'gelu',
        'task': 'segmentation'
    }
    
    logger.info("Model Configuration:")
    for key, value in config.items():
        logger.info(f"  {key}: {value}")
    
    # Clear module cache to ensure latest fixes are loaded
    modules_to_reload = [
        'geoaware_blocks',
        'geoaware_foundation', 
        'util_tools',
        'phisatnet'
    ]
    
    logger.info("Clearing module cache for latest improvements...")
    clear_module_cache(modules_to_reload)
    
    # Re-import to get latest fixes
    from hydranet_teacher import PhiSatNetDownstream
    
    # Create ONNX-compatible model
    logger.info("Creating ONNX-compatible model...")
    model = PhiSatNetDownstream(
        pretrained_path=None,
        task=config['task'],
        input_dim=config['input_dim'],
        output_dim=config['output_dim'],
        depths=config['depths'],
        dims=config['dims'],
        img_size=config['img_size'],
        freeze_body=False, 
        activation=config['activation']
    )
    
    model.eval()
    logger.info("✅ Model created successfully with latest improvements")
    
    # Model statistics
    total_params = sum(p.numel() for p in model.parameters())
    param_size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024**2
    
    logger.info("Model Statistics:")
    logger.info(f"  Total parameters: {total_params:,}")
    logger.info(f"  Parameter memory: {param_size_mb:.2f} MB")
    
    # Create dummy input for ONNX export
    dummy_input = torch.randn(1, config['input_dim'], config['img_size'], config['img_size'])
    
    # Export to ONNX (final compatible version)
    onnx_output_path = output_dir / 'model.onnx'
    
    logger.info("Exporting ONNX-compatible model...")
    export_model_to_onnx(
        model=model,
        dummy_input=dummy_input, 
        output_path=str(onnx_output_path), 
        opset_version=11,
        logger=logger
    )
    
    # Create sample input for testing
    sample_input_path = output_dir / 'sample_input.npy'
    sample_input = create_sample_input(
        shape=(1, config['input_dim'], config['img_size'], config['img_size']),
        output_path=str(sample_input_path),
        logger=logger
    )
    
    # Test model with sample input
    logger.info("Testing model with sample input...")
    test_input_torch = torch.from_numpy(sample_input)
    with torch.no_grad():
        test_output = model(test_input_torch)
    
    logger.info(f"  Input shape: {test_input_torch.shape}")
    logger.info(f"  Output shape: {test_output.shape}")
    logger.info(f"  Output range: [{test_output.min():.4f}, {test_output.max():.4f}]")
    
    logger.info("=== Generation Complete ===")
    logger.info(f"✅ ONNX model: {onnx_output_path}")
    logger.info(f"✅ Sample input: {sample_input_path}")
    logger.info(f"✅ Log file: {log_file_path}")
    logger.info("✅ Ready for OpenVINO conversion and Myriad deployment!")
    
    return str(onnx_output_path), str(sample_input_path)


if __name__ == '__main__':
    try:
        onnx_path, sample_path = main()
        print(f"\nSuccess! Files generated:")
        print(f"  ONNX model: {onnx_path}")
        print(f"  Sample input: {sample_path}")
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)