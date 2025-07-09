"""
This script contains sanity checks for the butterfly dataset:

1. Visualize the original butterfly images from the dataset
2. Visualize noisy butterfly images at different timesteps
"""

import torch
import numpy as np
import albumentations as A


from datasets import load_dataset
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from torch.utils.data import Dataset, DataLoader

from mypt.code_utils import pytorch_utils as pu
from mypt.visualization.general import visualize, visualize_grid


class HuggingFaceDatasetWrapper(Dataset):
    def __init__(self, dataset, transforms):
        self.dataset = dataset
        self.transforms = transforms
        
    def __len__(self):
        return len(self.dataset)
        
    def __getitem__(self, idx):
        image = self.dataset[idx]["image"].convert("RGB")
        transformed = self.transforms(image=np.asarray(image))
        return transformed["image"]


def visualize_original_butterflies(num_samples=5):
    """
    Visualize original butterfly images from the dataset
    """
    print("Visualizing original butterfly images...")
    
    # Load a small subset of the dataset
    dataset_name = "huggan/smithsonian_butterflies_subset"
    butterfly_ds = load_dataset(dataset_name, split="train[:5%]")
    
    # Define transforms
    transforms = A.Compose([
        A.Resize(height=128, width=128),
        A.ToTensorV2()
    ])
    
    # Create dataset wrapper
    wrapped_ds = HuggingFaceDatasetWrapper(butterfly_ds, transforms)
    
    # Create dataloader
    dataloader = DataLoader(wrapped_ds, batch_size=num_samples, shuffle=True)
    
    # Get a batch of images
    batch = next(iter(dataloader))
    
    # Visualize each image
    for i, img in enumerate(batch):
        visualize(img, window_name=f"Original Butterfly Image {i+1}")


def visualize_noisy_butterflies(num_samples=5, timesteps=[25, 100, 200, 400, 800, 995]):
    """
    Visualize noisy butterfly images at different timesteps
    """
    print("Visualizing noisy butterfly images...")
    
    # Set random seed for reproducibility
    pu.seed_everything(42)
    
    # Load a small subset of the dataset
    dataset_name = "huggan/smithsonian_butterflies_subset"
    butterfly_ds = load_dataset(dataset_name, split="train[:5%]")
    
    # Define transforms
    transforms = A.Compose([
        A.Resize(height=128, width=128),
        A.ToTensorV2()
    ])
    
    # Create dataset wrapper
    wrapped_ds = HuggingFaceDatasetWrapper(butterfly_ds, transforms)
    
    # Create dataloader
    dataloader = DataLoader(wrapped_ds, batch_size=num_samples, shuffle=True)
    
    # Initialize noise scheduler
    # noise_scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule="linear", beta_end=0.0015)

    noise_scheduler = DDPMScheduler(num_train_timesteps=1000)

    # Get device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Get a batch of images
    batch = next(iter(dataloader)).to(device)
    
    # Normalize images to [-1, 1] range as expected by diffusion models
    org_batch = batch.clone()

    norm_batch = (batch / 255.0) * 2 - 1
    
    if not (torch.all(norm_batch >= -1) and torch.all(norm_batch <= 1)):
        raise ValueError("Batch is not normalized to [-1, 1] range")
    
    # # Visualize original images
    # original_images = (batch + 1) * 127.5
    # for i, img in enumerate(original_images):
    #     visualize(img.cpu(), window_name=f"Original Image {i+1}")
    
    # Generate noise
    noise = torch.randn(norm_batch.shape, device=device)
    
    # Visualize noisy images at different timesteps
    for t in timesteps:
        # Create tensor of timesteps
        timestep_tensor = torch.tensor([t] * batch.shape[0], device=device, dtype=torch.long)
        
        # Add noise to images
        noisy_images = noise_scheduler.add_noise(norm_batch, noise, timestep_tensor)

        # convert to the [0, 255] range.
        # the noisy images values can have values less than -1 and greater than 1. (normal noise...)
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
    print("Running sanity checks for butterfly dataset...")
    # visualize_original_butterflies()
    visualize_noisy_butterflies() 