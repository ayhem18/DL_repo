"""
This script contains sanity checks for the MNIST dataset:

1. Visualize the original MNIST images from the dataset
2. Visualize noisy MNIST images at different timesteps
"""

import os
import torch
import numpy as np
import albumentations as A

from pathlib import Path
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from torch.utils.data import Dataset, DataLoader

from mypt.code_utils import pytorch_utils as pu
from mypt.visualization.general import visualize, visualize_grid

from dataset.mnist import MnistDSWrapper
from config import ModelConfig

SCRIPT_DIR = Path(__file__).parent


def visualize_original_mnist(num_samples=5):
    """
    Visualize original MNIST images from the dataset
    """
    print("Visualizing original MNIST images...")
    
    # Load a small subset of the dataset
    data_path = os.path.join(SCRIPT_DIR, 'data', 'train')
    model_config = ModelConfig()

    # Define transforms
    transforms = A.Compose([
        A.Resize(height=model_config.input_shape[1], width=model_config.input_shape[2]),
        A.ToTensorV2()
    ])
    
    # Create dataset wrapper
    wrapped_ds = MnistDSWrapper(root=data_path,
                                train=True,
                                transforms=transforms,
                                output_shape=model_config.input_shape[1:],
                                unconditional=True
                                )
    
    # Create dataloader
    dataloader = DataLoader(wrapped_ds, batch_size=num_samples, shuffle=True)
    
    # Get a batch of images
    batch = next(iter(dataloader))
    
    # Visualize each image
    for i, img in enumerate(batch):
        visualize(img, window_name=f"Original MNIST Image {i+1}")


def visualize_noisy_mnist(num_samples=5, timesteps=[25, 100, 200, 400, 800, 995]):
    """
    Visualize noisy MNIST images at different timesteps
    """
    print("Visualizing noisy MNIST images...")
    
    # Set random seed for reproducibility
    pu.seed_everything(42)
    
    # Load a small subset of the dataset
    data_path = os.path.join(SCRIPT_DIR, 'data', 'train')
    model_config = ModelConfig()
    
    # Define transforms
    transforms = A.Compose([
        A.Resize(height=model_config.input_shape[1], width=model_config.input_shape[2]),
        A.ToTensorV2()
    ])
    
    # Create dataset wrapper
    wrapped_ds = MnistDSWrapper(root=data_path,
                                train=True,
                                transforms=transforms,
                                output_shape=model_config.input_shape[1:],
                                unconditional=True
                                )
    
    # Create dataloader
    dataloader = DataLoader(wrapped_ds, batch_size=num_samples, shuffle=True)
    
    # Initialize noise scheduler
    noise_scheduler = DDPMScheduler(num_train_timesteps=1000)

    # Get device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Get a batch of images
    batch = next(iter(dataloader)).to(device)
    
    # Normalize images to [-1, 1] range as expected by diffusion models
    org_batch = batch.clone()

    norm_batch = (batch / 127.5) - 1.0
    
    if not (torch.all(norm_batch >= -1) and torch.all(norm_batch <= 1)):
        raise ValueError("Batch is not normalized to [-1, 1] range")
    
    # Generate noise
    noise = torch.randn(norm_batch.shape, device=device)
    
    # Visualize noisy images at different timesteps
    for t in timesteps:
        # Create tensor of timesteps
        timestep_tensor = torch.tensor([t] * batch.shape[0], device=device, dtype=torch.long)
        
        # Add noise to images
        noisy_images = noise_scheduler.add_noise(norm_batch, noise, timestep_tensor)

        # convert to the [0, 255] range.
        noisy_images = (noisy_images + 1) * 127.5

        # Visualize original and noisy images side-by-side
        for i in range(num_samples):
            original_image = org_batch[i]
            noisy_image = noisy_images[i]

            visualize_grid(
                [original_image.cpu(), noisy_image.cpu()], 
                title=f"Image {i+1}: Original and Noisy at Timestep {t}"
            )


if __name__ == "__main__":
    print("Running sanity checks for MNIST dataset...")
    # visualize_original_mnist()
    visualize_noisy_mnist() 