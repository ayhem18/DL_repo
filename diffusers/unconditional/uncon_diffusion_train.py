import os
import torch

import numpy as np

from tqdm import tqdm
from PIL import Image
from typing import Optional, Tuple, Union
from torch.utils.data import DataLoader
from diffusers import DDPMScheduler, UNet2DModel, DiffusionPipeline, DDPMPipeline


from mypt.shortcuts import P
from mypt.loggers import BaseLogger
from mypt.visualization.general import visualize
from mypt.code_utils import directories_and_files as dirf
from mypt.nets.conv_nets.diffusion_unet.wrapper.diffusion_unet import DiffusionUNet


def train_epoch(model: Union[DiffusionUNet, UNet2DModel], 
                noise_scheduler: DDPMScheduler,
                train_loader: DataLoader, 
                criterion: torch.nn.Module, 
                optimizer: torch.optim.Optimizer, 
                device: torch.device,
                logger: BaseLogger = None,
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
        logger: Logger for logging batch losses
        epoch: Current epoch number
    
    Returns:
        Average training loss for the epoch and list of batch losses
    """
    model = model.to(device)
    model.train()
    train_loss = 0.0
    num_train_samples = 0
    batch_losses = [None for _ in range(len(train_loader))]
    
    train_loop = tqdm(train_loader, desc=f"Training epoch {epoch or ''}")


    debug_samples = []
    debug_samples_timesteps = []
    loss_on_debug_samples = []

    for batch_idx, images in enumerate(train_loop):
        # move all data to the device
        images = images.to(device)        

        num_train_samples += images.shape[0]

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

        outputs = model.forward(noisy_images, timesteps)

        if isinstance(model, UNet2DModel):
            # for whatever reason, the outputs are a wrapper around a tensor.
            outputs = outputs.sample

        loss = criterion(outputs, noise)

        if debug:
            debug_ni = noisy_images[0].detach().cpu()
            debug_samples.append(debug_ni)
            debug_samples_timesteps.append(timesteps[0].detach().cpu())
            loss_on_debug_samples.append(loss.item())

        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        # Get the loss value
        batch_loss = loss.item()
        batch_losses[batch_idx] = batch_loss
        
        # Update statistics
        train_loss += images.shape[0] * batch_loss # multiply by the number of samples in the batch (since batch_loss is the averaged loss per sample... the averaging for the training loss is done at the end)
        train_loop.set_postfix(loss=batch_loss)
        
        # Log batch loss to TensorBoard if logger is provided
        if logger is not None and epoch is not None:
            global_step = epoch * len(train_loader) + batch_idx
            logger.log_scalar('Loss/train_batch', batch_loss, global_step)
    

    if debug:
        for i, (s, t, l) in enumerate(zip(debug_samples[:10], debug_samples_timesteps[:10], loss_on_debug_samples[:10])):
            # rescale s 
            s = (s + 1) * 127.5
            visualize(s, window_name=f"Debug Sample {i} - timestep: {t.item()} - loss: {l:.4f}")


    avg_loss = round(train_loss / num_train_samples, 4)
    return avg_loss, batch_losses


def val_epoch(model: DiffusionUNet, 
                noise_scheduler: DDPMScheduler,
                val_loader: DataLoader, 
                criterion: torch.nn.Module, 
                device: torch.device,
                logger: Optional[BaseLogger] = None,
                epoch: Optional[int] = None):
    
    model = model.to(device)
    model.eval()

    epoch_val_loss = 0.0
    val_num_samples = 0

    val_loop = tqdm(val_loader, desc="Validation")

    val_batch_losses = [None for _ in range(len(val_loader))]

    with torch.no_grad():

        for batch_idx, images in enumerate(val_loop):
            images = images.to(device)

            val_num_samples += images.shape[0]
            
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

            if isinstance(model, UNet2DModel):
                # for whatever reason, the outputs are a wrapper around a tensor.
                outputs = outputs.sample

            loss = criterion(outputs, noise)

            loss_value = loss.item()
            val_batch_losses[batch_idx] = loss_value 
            val_loop.set_postfix(loss=loss_value)

            # Update statistics
            epoch_val_loss += images.shape[0] * loss_value # multiply by the number of samples in the batch (since batch_loss is the averaged loss per sample... the averaging for the epoch-level loss is done at the end)
            
            if logger is not None and epoch is not None:
                global_step = epoch * len(val_loader) + batch_idx
                logger.log_scalar('Loss/val_batch', loss_value, global_step)

        avg_loss = round(epoch_val_loss / val_num_samples, 4)
        return avg_loss, val_batch_losses



def sample_with_diffusers(model: UNet2DModel, 
                          noise_scheduler: DDPMScheduler,
                          device: torch.device,
                          num_samples: int = 10,
                          num_inference_steps: int = 10) -> torch.Tensor:
    
    
    # set the number of inference time steps
    noise_scheduler.set_timesteps(num_inference_steps=num_inference_steps)
    
    pipeline = DDPMPipeline(unet=model, scheduler=noise_scheduler)
    pipeline.to(device)

    images: np.ndarray = pipeline(
        batch_size=num_samples,
        generator=torch.Generator(device='cpu').manual_seed(42),
        output_type="tensor", # if output_type is "pil", it will return a list of PIL images, any other value it will return a numpy array
        num_inference_steps=num_inference_steps
    ).images

    # convert to tensor and return
    return torch.from_numpy(images.transpose(0, 3, 1, 2))


def sample_manual(model: DiffusionUNet, 
                    images: torch.Tensor,
                    noise_scheduler: DDPMScheduler,
                    device: torch.device,
                    num_inference_steps: int = 1000,
                    ) -> torch.Tensor:
    
    model = model.to(device)
    model.eval()

    noise_scheduler.set_timesteps(num_inference_steps=num_inference_steps)

    samples = torch.randn(*images.shape, device=device)

    for _, t in enumerate(tqdm(noise_scheduler.timesteps, desc="Sampling")):
    
        with torch.no_grad():
            # 1. predict noise residual
            residual = model.forward(samples, torch.ones(samples.shape[0], device=device) * t) # generate images with the same label and the same masks as the validation batch
            
            if isinstance(model, UNet2DModel):
                # for whatever reason, the outputs are a wrapper around a tensor.
                residual = residual.sample

            # 2. compute previous image and set x_t -> x_t-1
            samples = noise_scheduler.step(residual, t, samples).prev_sample

    return samples


def sample_from_diffusion_model(model: Union[DiffusionUNet, UNet2DModel], 
                          noise_scheduler: DDPMScheduler,
                          device: torch.device,
                          images: torch.Tensor,
                          num_samples: int = 10,
                          num_inference_steps: int = 10) -> Tuple[torch.Tensor, bool]:
    
    if isinstance(model, UNet2DModel):
        return sample_with_diffusers(model, noise_scheduler, device, num_samples, num_inference_steps), False
    else:
        return sample_manual(model, images, noise_scheduler, device, num_inference_steps), True


def val_sample_diffusion_epoch(model: DiffusionUNet, 
                        noise_scheduler: DDPMScheduler,
                        val_loader: DataLoader, 
                        device: torch.device,
                        logger: BaseLogger,
                        log_dir: P,
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
            
            if batch_idx >= 2:
                break

            images = images.to(device)
            
            diffusion_samples, scale_images = sample_from_diffusion_model(model, noise_scheduler, device, images[:10], num_samples=10, num_inference_steps=250)

            epoch_dir = os.path.join(log_dir, 'samples', f"epoch_{epoch+1}")
            dirf.process_path(epoch_dir, dir_ok=True, file_ok=False, must_exist=False)

            for i, s in enumerate(diffusion_samples):
                # rescale to [0, 255] for saving and visualization

                if scale_images:
                    vis_s = (s + 1) * 127.5
                else:
                    vis_s = s

                logger.log_image(f"sample_{batch_idx}_{i}", vis_s, epoch)


                if debug:
                    visualize(vis_s, window_name=f"sampled_image_{batch_idx}_{i}")

                # convert to numpy for saving
                if vis_s.shape[0] > 3: # (C, H, W) with C > 3
                    save_s = vis_s.cpu().permute(1, 2, 0)
                else: # (C, H, W) with C <= 3
                    save_s = vis_s.cpu().squeeze(0)
                    
                save_s = save_s.numpy().astype(np.uint8)
                Image.fromarray(save_s).save(os.path.join(epoch_dir, f"sample_{batch_idx}_{i}.png"))


def train_diffusion_model(model: DiffusionUNet, 
               noise_scheduler: DDPMScheduler,
               train_loader: DataLoader, 
               val_loader: DataLoader, 
               criterion: torch.nn.Module, 
               optimizer: torch.optim.Optimizer, 
               lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
               num_epochs: int, 
               device: torch.device, 
               logger: BaseLogger,
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
        logger: Logger for logging batch losses
        log_dir: Directory for saving model and logs
    
    Returns:
        Trained model
    """
    best_val_loss = float('inf')
    all_train_losses = []
    all_val_losses = []
    
    for epoch in tqdm(range(num_epochs), desc="Training loop"):
        
        # Training phase
        train_loss, batch_losses = train_epoch(
            model=model,
            noise_scheduler=noise_scheduler,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            logger=logger,
            epoch=epoch,
            debug=debug,
        )
        
        lr_scheduler.step()
        
        logger.log_scalar('Loss/train_epoch', train_loss, epoch)
        all_train_losses.append(train_loss)

        # Validation phase
        val_loss, val_batch_losses = val_epoch(
            model=model,
            noise_scheduler=noise_scheduler,
            val_loader=val_loader,
            criterion=criterion,
            device=device,
            logger=logger,
            epoch=epoch
        )

        logger.log_scalar('Loss/val_epoch', val_loss, epoch)
        all_val_losses.append(val_loss)

        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        # Log epoch losses
        logger.log_scalar('Loss/train_epoch', train_loss, epoch)
        logger.log_scalar('Loss/val_epoch', val_loss, epoch)

        if (epoch + 1) % val_per_epoch == 0:
            val_sample_diffusion_epoch(model, noise_scheduler, val_loader, device, logger, log_dir, epoch, debug)

        # Save the model if validation loss improves
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(log_dir, 'best_model.pth'))
            print(f"Model saved with validation loss: {val_loss:.4f}")

            
    return model

