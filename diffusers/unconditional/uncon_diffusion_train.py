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


def train_epoch(model: DiffusionUNet, 
                noise_scheduler: DDPMScheduler,
                train_loader: DataLoader, 
                criterion: torch.nn.Module, 
                optimizer: torch.optim.Optimizer, 
                device: torch.device,
                writer: SummaryWriter = None,
                epoch: int = None,
                debug: bool = False):
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


    debug_samples = []
    debug_samples_timesteps = []

    for batch_idx, images in enumerate(train_loop):
        # move all data to the device
        images = images.to(device)        
        
        # scale the image to the range [-1, 1]
        images = images * 2.0 - 1.0

        if not torch.all(images >= -1.0) or not torch.all(images <= 1.0):
            raise ValueError("Images must be scaled to the [-1, 1] range. Otherwise, it might affect the training process.")
        
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # create noise
        noise = torch.randn(images.shape, device=images.device)

        # assign a random timestep (between 0 and the maximum number of timesteps) 
        # to each image in the batch
        timesteps = torch.randint(
            0, # between 0 and the maximum number of timesteps     
            noise_scheduler.config.num_train_timesteps, 
            (images.shape[0],), device=images.device,
            dtype=torch.int64
        )

        # add noise to the images
        noisy_images = noise_scheduler.add_noise(images, noise, timesteps).to(torch.float32)

        if debug:
            debug_ni = noisy_images[0].detach().cpu()
            debug_samples.append(debug_ni)
            debug_samples_timesteps.append(timesteps[0].detach().cpu())

        outputs = model.forward(noisy_images, timesteps)

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
    

    if debug:
        for i, (s, t) in enumerate(zip(debug_samples[:10], debug_samples_timesteps[:10])):
            # rescale s 
            s = (s + 1) * 127.5
            visualize(s, window_name=f"Debug Sample {i} - timestep: {t.item()}")


    avg_loss = train_loss / len(train_loader)
    return avg_loss, batch_losses


def val_epoch(model: DiffusionUNet, 
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

    with torch.no_grad():

        for batch_idx, images in enumerate(val_loop):
            images = images.to(device)
            
            noise = torch.randn(images.shape, device=device)

            # convert images to the [-1, 1] range
            images = (images * 2) - 1

            # make sure the range is correct
            if torch.max(images) > 1 or torch.min(images) < -1:
                raise ValueError("the images must be in the [-1, 1]. Otherwise, it might affect the training process.")

            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, # between 0 and the num_train_timesteps 
                (images.shape[0],), device=device,
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
                                noise_scheduler: DDPMScheduler,
                                device: torch.device) -> torch.Tensor:
    
    model = model.to(device)
    model.eval()

    noise_scheduler.set_timesteps(num_inference_steps=100) # 1000 is too slow

    samples = torch.randn(*images.shape, device=device)

    for _, t in enumerate(tqdm(noise_scheduler.timesteps, desc="Sampling")):
    
        with torch.no_grad():
            # 1. predict noise residual
            residual = model.forward(samples, torch.ones(samples.shape[0], device=device) * t) # generate images with the same label and the same masks as the validation batch
            
            if not isinstance(residual, torch.Tensor):
                residual = residual.sample 
            
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
        val_loop = tqdm(val_loader, desc="Sampling validation")

        for batch_idx, images in enumerate(val_loop):
            
            if batch_idx > 2:
                break

            images = images.to(device)
            
            diffusion_samples = sample_from_diffusion_model(model, images, noise_scheduler, device=device)

            epoch_dir = os.path.join(val_dir, f"epoch_{epoch+1}")
            dirf.process_path(epoch_dir, dir_ok=True, file_ok=False, must_exist=False)

            for i, s in enumerate(diffusion_samples):
                # rescale s to the [0, 255] range
                s = (s + 1) * 127.5

                if debug:
                    visualize(s, window_name=f"sampled_image_{batch_idx}_{i}")

                # convert to numpy
                if s.shape[0] > 3:
                    s = s.cpu().permute(1, 2, 0)
                else:
                    s = s.cpu().squeeze(0)
                    
                # convert to numpy and to uint8
                s = s.numpy().astype(np.uint8)
                Image.fromarray(s).save(os.path.join(epoch_dir, f"sample_{batch_idx}_{i}.png"))


def train_diffusion_model(model: DiffusionUNet, 
               noise_scheduler: DDPMScheduler,
               train_loader: DataLoader, 
               val_loader: DataLoader, 
               criterion: torch.nn.Module, 
               optimizer: torch.optim.Optimizer, 
               lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
               num_epochs: int, 
               device: torch.device, 
               writer: SummaryWriter,
               log_dir: P,
               val_per_epoch: int = 5,
               debug: bool = False):
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
        train_loss, batch_losses = train_epoch(
            model=model,
            noise_scheduler=noise_scheduler,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            writer=writer,
            epoch=epoch,
            debug=debug,
        )
        
        lr_scheduler.step()
        
        writer.add_scalar('Loss/train_epoch', train_loss, epoch)
        all_train_losses.append(train_loss)

        # Validation phase
        val_loss, val_batch_losses = val_epoch(
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

        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}")        


        if epoch % val_per_epoch == 0:
            val_sample_diffusion_epoch(model, noise_scheduler, val_loader, device, os.path.join(log_dir, 'samples'), epoch, debug=debug)

        # Save the model if validation loss improves
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(log_dir, 'best_model.pth'))
            print(f"Model saved with validation loss: {val_loss:.4f}")

            
    return model

