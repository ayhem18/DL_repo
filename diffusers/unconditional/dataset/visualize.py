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


def analyze_dataset_statistics(
    root: str = "./data",
    output_shape: Tuple[int, int] = (64, 64),
    num_samples: int = 1000
):
    """
    Analyze statistics of the dataset to ensure it's behaving as expected.
    
    Args:
        root: Path to the MNIST dataset
        output_shape: Shape to pad images to
        num_samples: Number of samples to analyze
    """
    # Initialize the dataset without transformations for clean analysis
    dataset = MnistDSWrapper(
        root=root,
        train=True,
        transforms=None,
        output_shape=output_shape
    )
    
    # Collect statistics
    stats = {
        "num_samples": min(num_samples, len(dataset)),
        "class_distribution": {},
        "label_map_stats": {
            "min": float('inf'),
            "max": float('-inf'),
            "mean_nonzero": 0,
            "total_nonzero_pixels": 0
        }
    }
    
    for i in range(min(num_samples, len(dataset))):
        image, label, label_map = dataset[i]
        
        # Update class distribution
        if label not in stats["class_distribution"]:
            stats["class_distribution"][label] = 0
        stats["class_distribution"][label] += 1
        
        # Update label map statistics
        label_map_np = label_map.numpy()
        stats["label_map_stats"]["min"] = min(stats["label_map_stats"]["min"], label_map_np.min())
        stats["label_map_stats"]["max"] = max(stats["label_map_stats"]["max"], label_map_np.max())
        
        # Count non-zero pixels and their mean value
        nonzero_mask = label_map_np > 0
        num_nonzero = nonzero_mask.sum()
        if num_nonzero > 0:
            stats["label_map_stats"]["total_nonzero_pixels"] += num_nonzero
            stats["label_map_stats"]["mean_nonzero"] += label_map_np[nonzero_mask].sum()
    
    # Calculate final mean of non-zero pixels
    if stats["label_map_stats"]["total_nonzero_pixels"] > 0:
        stats["label_map_stats"]["mean_nonzero"] /= stats["label_map_stats"]["total_nonzero_pixels"]
    
    # Pretty print the statistics
    print("\n===== Dataset Statistics =====")
    print(f"Analyzed {stats['num_samples']} samples with output shape {output_shape}")
    
    print("\nClass Distribution:")
    for label, count in sorted(stats["class_distribution"].items()):
        print(f"  Class {label}: {count} samples ({count/stats['num_samples']*100:.1f}%)")
    
    print("\nLabel Map Statistics:")
    print(f"  Min value: {stats['label_map_stats']['min']}")
    print(f"  Max value: {stats['label_map_stats']['max']}")
    print(f"  Mean of non-zero pixels: {stats['label_map_stats']['mean_nonzero']:.2f}")
    print(f"  Total non-zero pixels: {stats['label_map_stats']['total_nonzero_pixels']}")
    
    return stats


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
    
    # # Analyze dataset statistics
    # print("\nAnalyzing dataset statistics...")
    # stats = analyze_dataset_statistics(
    #     root=data_root,
    #     output_shape=(64, 64),
    #     num_samples=1000
    # )
    
    # print("\nVisualization and analysis complete!") 
