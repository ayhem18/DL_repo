import os
import torch

import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from PIL import Image
from typing import Optional
from diffusers import DDPMScheduler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


from mypt.shortcuts import P
from mypt.visualization.general import visualize
from mypt.code_utils import directories_and_files as dirf
from mypt.nets.conv_nets.diffusion_unet.wrapper.diffusion_unet import DiffusionUNet


def train_diffusion_epoch(model: DiffusionUNet, 
                noise_scheduler: DDPMScheduler,
               train_loader: DataLoader, 
               criterion: torch.nn.Module, 
               optimizer: torch.optim.Optimizer, 
               device: torch.device,
               writer: SummaryWriter = None,
               epoch: int = None):
    """
    Run one epoch of training.
    
    Args:
        model: The model to train
        train_loader: DataLoader for training data
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on
        writer: TensorBoard writer for logging batch losses
        epoch: Current epoch number
    
    Returns:
        Average training loss for the epoch and list of batch losses
    """
    model = model.to(device)
    model.train()
    train_loss = 0.0
    batch_losses = [None for _ in range(len(train_loader))]
    
    train_loop = tqdm(train_loader, desc="Training")

    # batch_size = batch.shape[0]
    # # Sample noise to add to the images
    # noise = torch.randn(batch.shape, device=batch.device)

    # # Sample a random timestep for each image
    # timesteps = torch.randint(
    #     0, noise_scheduler.config.num_train_timesteps, (batch_size,), device=device,
    #     dtype=torch.int64
    # )

    # # Add noise to the clean images according to the noise magnitude at each timestep
    # # (this is the forward diffusion process)
    # noisy_images = noise_scheduler.add_noise(batch, noise, timesteps)


    for batch_idx, (images, labels, mask_bbx) in enumerate(train_loop):
        if batch_idx > 15:
            break

        # move all data to the device
        images = images.to(device)
        labels = labels.to(device)
        mask_bbx = mask_bbx.to(device)
        
        # Zero the parameter gradients
        optimizer.zero_grad()
        

        # create noise
        noise = torch.randn(images.shape, device=images.device)

        # assign a random timestep (between 0 and the maximum number of timesteps) 
        # to each image in the batch
        timesteps = torch.randint(
            0, 
            noise_scheduler.config.num_train_timesteps, 
            
            (images.shape[0],), device=images.device,
            dtype=torch.int64
        )

        # add noise to the images
        noisy_images = noise_scheduler.add_noise(images, noise, timesteps).to(torch.float32)

        outputs = model.forward(noisy_images, timesteps, labels, mask_bbx)

        # Forward pass
        # outputs = model(noisy_images)
        loss = criterion(outputs, noise)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        # Get the loss value
        batch_loss = loss.item()
        batch_losses[batch_idx] = batch_loss
        
        # Update statistics
        train_loss += batch_loss
        train_loop.set_postfix(loss=batch_loss)
        
        # Log batch loss to TensorBoard if writer is provided
        if writer is not None and epoch is not None:
            global_step = epoch * len(train_loader) + batch_idx
            writer.add_scalar('Loss/train_batch', batch_loss, global_step)
    
    avg_loss = train_loss / len(train_loader)
    return avg_loss, batch_losses


def val_diffusion_epoch(model: DiffusionUNet, 
                        noise_scheduler: DDPMScheduler,
                        val_loader: DataLoader, 
                        criterion: torch.nn.Module, 
                        device: torch.device,
                        writer: Optional[SummaryWriter] = None,
                        epoch: Optional[int] = None):
    
    model = model.to(device)
    model.eval()

    val_loop = tqdm(val_loader, desc="Validation")

    val_batch_losses = [None for _ in range(len(val_loader))]

    for batch_idx, images in enumerate(val_loop):
        noise = torch.randn(images.shape, device=images.device)
        images = images.to(device)

        # convert images to the [-1, 1] range
        images = (images * 2) - 1

        # make sure the range is correct
        if torch.max(images) > 1 or torch.min(images) < -1:
            raise ValueError("the images must be in the [-1, 1]. Otherwise, it might affect the training process.")

        timesteps = torch.randint(
            0, 
            noise_scheduler.config.num_train_timesteps, 
            
            (images.shape[0],), device=images.device,
            dtype=torch.int64
        )   

        noisy_images = noise_scheduler.add_noise(images, noise, timesteps).to(torch.float32)

        outputs = model.forward(noisy_images, timesteps)

        loss = criterion(outputs, noise)

        loss_value = loss.item()
        val_batch_losses[batch_idx] = loss_value
        val_loop.set_postfix(loss=loss_value)
        
        if writer is not None and epoch is not None:
            global_step = epoch * len(val_loader) + batch_idx
            writer.add_scalar('Loss/val_batch', loss_value, global_step)

    avg_loss = sum(val_batch_losses) / len(val_loader)
    return avg_loss, val_batch_losses


def sample_from_diffusion_model(model: DiffusionUNet, 
                                images: torch.Tensor,
                                labels: torch.Tensor, 
                                masks: torch.Tensor,
                                noise_scheduler: DDPMScheduler,
                                device: torch.device):
    
    model = model.to(device)
    model.eval()

    noise_scheduler.set_timesteps(num_inference_steps=100)

    samples = torch.randn(*images.shape, device=device)

    for _, t in enumerate(tqdm(noise_scheduler.timesteps, desc="Sampling")):
    
        with torch.no_grad():
            # 1. predict noise residual
            residual = model.forward(samples, torch.ones(samples.shape[0], device=device) * t, labels, masks) # generate images with the same label and the same masks as the validation batch
            # 2. compute previous image and set x_t -> x_t-1
            samples = noise_scheduler.step(residual, t, samples).prev_sample

    return samples


def val_sample_diffusion_epoch(model: DiffusionUNet, 
                        noise_scheduler: DDPMScheduler,
                        val_loader: DataLoader, 
                        device: torch.device,
                        val_dir: P,
                        epoch: int,
                        debug: bool = False):
    """
    Run one epoch of validation.
    
    Args:
        model: The model to validate
        val_loader: DataLoader for validation data
        criterion: Loss function
        device: Device to validate on
    
    Returns:
        Average validation loss for the epoch
    """

    with torch.no_grad():
        val_loop = tqdm(val_loader, desc="Validation")

        for batch_idx, images in enumerate(val_loop):
            
            if batch_idx > 2:
                break

            images = images.to(device)
            
            diffusion_samples = sample_from_diffusion_model(model, images, noise_scheduler, device=device)

            epoch_dir = os.path.join(val_dir, f"epoch_{epoch+1}")
            dirf.process_path(epoch_dir, dir_ok=True, file_ok=False, must_exist=False)

            for i, s in enumerate(diffusion_samples):
                # convert to numpy
                if s.shape[0] > 3:
                    s = s.cpu().permute(1, 2, 0).numpy().astype(np.uint8)
                else:
                    s = s.cpu().squeeze(0).numpy().astype(np.uint8)

                if debug:
                    visualize(s, title=f"sample_{batch_idx}_{i}")

                Image.fromarray(s).save(os.path.join(epoch_dir, f"sample_{batch_idx}_{i}.png"))



def log_predictions(model: torch.nn.Module, 
                    val_loader: DataLoader, 
                    writer: SummaryWriter, 
                    epoch: int, 
                    device: torch.device):
    """Log sample predictions to TensorBoard"""
    model.eval()

    images, masks = next(iter(val_loader))
    
    images = images.to(device)
    masks = masks.to(device)
    
    with torch.no_grad():
        outputs = model(images)
        predictions = (torch.sigmoid(outputs) > 0.5)

    images = images.cpu().permute(0, 2, 3, 1).numpy().astype(np.uint8)
    # convert the masks to the [0, 255] range and the type uint8
    masks = (masks.cpu().permute(0, 2, 3, 1).numpy() * 255).astype(np.uint8) 
    predictions = (predictions.cpu().permute(0, 2, 3, 1).numpy() * 255).clip(0, 255).astype(np.uint8)

    # Log a few sample images
    for i in range(min(3, len(images))):
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original image
        axes[0].imshow(images[i])
        axes[0].set_title("Input Image")
        axes[0].axis('off')
        
        # Ground truth mask
        axes[1].imshow(masks[i])
        axes[1].set_title("Ground Truth")
        axes[1].axis('off')
        
        # Predicted mask
        axes[2].imshow(predictions[i])
        axes[2].set_title("Prediction")
        axes[2].axis('off')
        
        plt.tight_layout()
        writer.add_figure(f'Predictions/sample_{i}', fig, epoch)
        plt.close(fig)

def train_diffusion_model(model: DiffusionUNet, 
               noise_scheduler: DDPMScheduler,
               train_loader: DataLoader, 
               val_loader: DataLoader, 
               criterion: torch.nn.Module, 
               optimizer: torch.optim.Optimizer, 
            #    lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
               num_epochs: int, 
               device: torch.device, 
               writer: SummaryWriter,
               log_dir: P):
    """
    Train the model and validate it periodically.
    
    Args:
        model: The model to train
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        criterion: Loss function
        optimizer: Optimizer
        num_epochs: Number of epochs to train for
        device: Device to train on
        writer: TensorBoard writer
        log_dir: Directory for saving model and logs
    
    Returns:
        Trained model
    """
    best_val_loss = float('inf')
    all_train_losses = []
    all_val_losses = []
    
    for epoch in range(num_epochs):
        
        # Training phase
        train_loss, batch_losses = train_diffusion_epoch(
            model=model,
            noise_scheduler=noise_scheduler,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            writer=writer,
            epoch=epoch
        )
        
        # lr_scheduler.step()
        
        writer.add_scalar('Loss/train_epoch', train_loss, epoch)
        all_train_losses.append(train_loss)



        # val_diffusion_epoch(model, noise_scheduler, val_loader, criterion, device, os.path.join(log_dir, 'samples'), epoch)

        # Validation phase
        val_loss, val_batch_losses = val_diffusion_epoch(
            model=model,
            noise_scheduler=noise_scheduler,
            val_loader=val_loader,
            criterion=criterion,
            device=device,
            writer=writer,
            epoch=epoch
        )

        writer.add_scalar('Loss/val_epoch', val_loss, epoch)
        all_val_losses.append(val_loss)
        
        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}")        

        # Save the model if validation loss improves
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(log_dir, 'best_model.pth'))
            print(f"Model saved with validation loss: {val_loss:.4f}")

        if epoch % val_per_epoch == 0:
            val_sample_diffusion_epoch(model, noise_scheduler, val_loader, device, os.path.join(log_dir, 'samples'), epoch)

        # Log sample predictions
        # if epoch % 5 == 0:
        #     log_predictions(model, val_loader, writer, epoch, device)
            
    return model

