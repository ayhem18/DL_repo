import os
import torch
import numpy as np
import albumentations as A
import matplotlib.pyplot as plt

from pathlib import Path
from typing import Tuple, List, Optional
from torch.utils.data import DataLoader

# Import the dataset class
from mnist import MnistDSWrapper

def visualize_mnist_samples(
    root: str = "./data",
    num_samples: int = 5,
    output_shape: Tuple[int, int] = (48, 48),
    batch_size: int = 1,
    apply_transforms: bool = True,
    save_path: Optional[str] = None
):
    """
    Visualize samples from the MnistDSWrapper dataset.
    
    Args:
        root: Path to the MNIST dataset
        num_samples: Number of samples to visualize
        output_shape: Shape to pad images to
        batch_size: Batch size for the dataloader
        apply_transforms: Whether to apply transformations
        save_path: If provided, save the visualization to this path
    """
    # Create a directory for the dataset if it doesn't exist
    os.makedirs(root, exist_ok=True)
    
    # Define transformations (if requested)
    transforms = None
    if apply_transforms:
        transforms = [
            A.RandomResizedCrop(size=(128, 128), scale=(0.4, 1))

            # A.RandomRotate90(p=0.5),
            # A.HorizontalFlip(p=0.5),
            # A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            # A.RandomResizedCrop(size=output_shape, scale=(0.7, 1), p=0.8),
            # A.Normalize(mean=0.5, std=0.5),
        ]
    
    # Initialize the dataset
    dataset = MnistDSWrapper(
        root=root,
        train=True,
        transforms=transforms,
        output_shape=output_shape
    )
    
    # Create a dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False
    )
    
    # Get samples to visualize
    samples = []
    for i, (image, label, label_map) in enumerate(dataloader):
        if i >= num_samples:
            break
        samples.append((
            image.squeeze(0) if batch_size == 1 else image[0],
            label.item() if batch_size == 1 else label[0].item(),
            label_map.squeeze(0) if batch_size == 1 else label_map[0]
        ))
    
    # Create a figure for visualization
    fig, axs = plt.subplots(len(samples), 2, figsize=(12, 4 * len(samples)))
    if len(samples) == 1:
        axs = [axs]
    

    for i, (image, label, label_map) in enumerate(samples):
        image = image.squeeze(0).numpy()
        label_map = label_map.squeeze(0).numpy() 
        # Plot the original image
        axs[i][0].imshow(image, cmap='gray')
        axs[i][0].set_title(f"Image (Label: {label})")
        axs[i][0].axis('off')
        
        # Plot the label map
        axs[i][1].imshow(label_map, cmap='viridis')
        axs[i][1].set_title(f"Label Map (Value: {label})")
        axs[i][1].axis('off')
        
    plt.tight_layout()
    
    # Save if a path is provided
    if save_path:
        plt.savefig(save_path)
        print(f"Visualization saved to {save_path}")
    
    plt.show()
    
    return samples



if __name__ == "__main__":
    # Set the root path for the dataset
    data_root = "./data"
    
    # Visualize samples
    for _ in range(10):
        samples = visualize_mnist_samples(
            root=data_root,
            num_samples=3,
            output_shape=(64, 64),
            apply_transforms=True,
            save_path="mnist_visualization.png"
        )
    