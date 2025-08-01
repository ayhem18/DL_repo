import torch

from tqdm import tqdm
from torch.utils.data import DataLoader
from typing import Any, Dict, List, Optional, Tuple, Union
from diffusers.models.unets.unet_2d import UNet2DModel
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler


from mypt.shortcuts import P
from mypt.loggers import BaseLogger
from mypt.visualization.general import visualize
from mypt.nets.conv_nets.diffusion_unet.wrapper.diffusion_unet1d import DiffusionUNetOneDim

from training.sample import sample_diffusion_epoch       
from training.time_steps import AbstractTimeStepsSampler, set_timesteps_sampler


def _normalize_images(images: torch.Tensor) -> torch.Tensor:
    """
    Normalizing images is so important !!! (I am not sure if having [0, 1] is significantly different from [-1, 1] (probably depends on the scheduler I am using))
    """
    # make sure the images are either in the [0, 1] or [0, 255] ranges
    if (not (images.min() >= 0 and images.max() <= 1)) and (not (images.min() >= 0 and images.max() <= 255)):
        raise ValueError("Images must be in the [0, 1] or [0, 255] range")  

    # at this point, the images are either in the [0, 1] or [0, 255] ranges     
    if images.max() > 1:
        images = images / 255.0

    # convert to the [-1, 1] range
    return (images * 2.0) - 1.0


def _prepare_bins(timestep_bins: List[int], num_train_steps: int) -> Tuple[torch.Tensor, List[str]]:
    bin_boundaries = torch.tensor([0] + timestep_bins, device='cpu')
    bin_labels = [f'{bin_boundaries[i]}-{bin_boundaries[i+1]-1}' for i in range(len(bin_boundaries)-1)]
    
    # the last value of the bin boundaries must be equal to the number of training steps
    if bin_boundaries[-1] != num_train_steps:
        raise ValueError(f"The last bin boundary must be equal to the number of training steps, but it is {bin_boundaries[-1]} and the number of training steps is {num_train_steps}")

    return bin_boundaries, bin_labels


def _compute_bin_losses(bin_boundaries: torch.Tensor, 
                        bin_labels: List[str], 
                        timesteps: torch.Tensor,
                        per_sample_loss: torch.Tensor,
                        train_loss_per_bin: Dict[str, float],
                        train_count_per_bin: Dict[str, int]) -> None:
    """keeps track of the loss per bin and the number of samples per bin for the training loss

    Args:
        bin_boundaries (torch.Tensor): the boundaries of the bins
        bin_labels (List[str]): the labels of the bins
        timesteps (torch.Tensor): the timesteps of the samples
        per_sample_loss (torch.Tensor): the loss per sample
        train_loss_per_bin (Dict[str, float]): the accumulated loss for samples that belong to a bin (across the entire epoch)
        train_count_per_bin (Dict[str, int]): the accumulated number of samples that belong to a bin
    """

    bin_indices = torch.bucketize(timesteps.cpu(), bin_boundaries, right=False)
    for i, b_idx in enumerate(bin_indices):
        label_idx = b_idx.item()

        if label_idx >= len(bin_labels):
            continue

        label = bin_labels[label_idx]
        train_loss_per_bin[label] += per_sample_loss[i].item()
        train_count_per_bin[label] += 1

    
def train_epoch(model: Union[DiffusionUNetOneDim, UNet2DModel], 
                noise_scheduler: DDPMScheduler,
                train_loader: DataLoader, 
                criterion: torch.nn.Module, 
                optimizer: torch.optim.Optimizer, 
                device: torch.device,
                timestep_bins: List[int],
                timesteps_sampler: AbstractTimeStepsSampler,
                logger: Optional[BaseLogger] = None,
                epoch: Optional[int] = None,
                max_grad_norm: float = 1.0,
                debug: bool = False,
                lr_scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None, 
                epoch_lr: Optional[float] = None) -> Tuple[float, List[float], Dict[str, float]]:
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

    if not hasattr(criterion, 'reduction'):
        raise ValueError("criterion must have a 'reduction' attribute")
    
    criterion.reduction = 'none'

    # at least one of "epoch_lr" or "lr_scheduler" should be provided
    if epoch_lr is not None and lr_scheduler is not None:
        raise ValueError("either 'epoch_lr' or 'lr_scheduler' should be provided, not both")

    model = model.to(device)
    model.train()
    
    train_loss = 0.0
    num_train_samples = 0
    batch_losses = [None for _ in range(len(train_loader))]
    
    # prepare the bins for the loss per timestep
    bin_boundaries, bin_labels = _prepare_bins(timestep_bins, noise_scheduler.config.num_train_timesteps)

    train_loss_per_bin = {label: 0.0 for label in bin_labels}
    train_count_per_bin = {label: 0 for label in bin_labels}

    train_loop = tqdm(train_loader, desc=f"Training epoch {epoch or ''}")


    debug_samples = []
    debug_samples_timesteps = []
    loss_on_debug_samples = []

    for batch_idx, images in enumerate(train_loop):
        # normalize the images to the [-1, 1] range.
        # TODO: think of a more efficient way to do this
        images = _normalize_images(images).to(device)
        num_train_samples += images.shape[0]

        # Zero the parameter gradients
        optimizer.zero_grad()

        # create noise
        noise = torch.randn(images.shape, device=images.device)

        # assign a random timestep (between 0 and the maximum number of timesteps) 
        # to each image in the batch
        timesteps = timesteps_sampler.sample(batch_size=images.shape[0], device=images.device)

        # add noise to the images
        noisy_images = noise_scheduler.add_noise(images, noise, timesteps).to(torch.float32)

        outputs = model.forward(noisy_images, timesteps)
        
        if hasattr(outputs, 'sample'):
            outputs = outputs.sample

        per_sample_loss = criterion(outputs, noise).mean(dim=tuple(range(1, len(outputs.shape)))) 
        loss = per_sample_loss.mean()

        if debug:
            debug_ni = noisy_images[0].detach().cpu()
            debug_samples.append(debug_ni)
            debug_samples_timesteps.append(timesteps[0].detach().cpu())
            loss_on_debug_samples.append(loss.item())

        # compute the gradients
        loss.backward()

        # clip the gradients to prevent them from exploding, which is a common issue in training deep neural networks.
        # this helps stabilize the training process. A max_norm of 1.0 is a common default.
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

        optimizer.step()
        
        if lr_scheduler is not None:
            lr_scheduler.step()

        # batch loss
        batch_loss = loss.item()
        batch_losses[batch_idx] = batch_loss

        train_loss += images.shape[0] * batch_loss # multiply by the number of samples in the batch (since batch_loss is the averaged loss per sample... the averaging for the training loss is done at the end)
        train_loop.set_postfix(loss=batch_loss)

        _compute_bin_losses(bin_boundaries, bin_labels, timesteps, per_sample_loss, train_loss_per_bin, train_count_per_bin)
        
        # Log batch loss to TensorBoard if logger is provided
        if logger is not None and epoch is not None:
            global_step = epoch * len(train_loader) + batch_idx
            logger.log_scalar('Loss/train_batch', batch_loss, global_step)

        # log the loss and the learning rate to the tqdm progress bar
        # prioritize the lr_scheduler if it is provided
        epoch_lr = lr_scheduler.get_last_lr()[0] if lr_scheduler is not None else epoch_lr
        train_loop.set_postfix(**{'batch_loss': batch_loss, 'lr': epoch_lr})

    if debug:
        for i, (s, t, l) in enumerate(zip(debug_samples[:10], debug_samples_timesteps[:10], loss_on_debug_samples[:10])):
            # rescale s 
            s = (s + 1) * 127.5
            visualize(s, window_name=f"Debug Sample {i} - timestep: {t.item()} - loss: {l:.4f}")


    avg_loss = round(train_loss / num_train_samples, 8)

    # keep only the bins that have at least one sample
    avg_train_loss_per_bin = dict([(label, train_loss_per_bin[label] / train_count_per_bin[label]) for label in bin_labels if train_count_per_bin[label]])

    return avg_loss, batch_losses, avg_train_loss_per_bin


def train_epoch_logging(       
                        # metrics from the training epoch
                        train_loss: float,
                        epoch_lr: float,
                        train_loss_per_bin: Dict[str, float],
                        epoch: int, 

                        # variables used to track the training metrics : make sure they are passed by reference !!!!
                        logger: BaseLogger, 
                        train_loop: tqdm,
                        all_train_losses: List[float]):
    """
    This function groups all the logging and tracking logic for the training epoch
    """

    # log the loss and the learning rate to the tqdm progress bar
    train_loop.set_postfix(**{'epoch_loss': train_loss, 'lr': epoch_lr})
    
    logger.log_scalar('Loss/train_epoch', train_loss, epoch)

    for bin_label, loss in train_loss_per_bin.items():
        logger.log_scalar(f'Loss/train_epoch_bin_{bin_label}', loss, epoch)

    all_train_losses.append(train_loss)


def val_epoch(model: Union[DiffusionUNetOneDim, UNet2DModel], 
            noise_scheduler: DDPMScheduler,
            val_loader: DataLoader, 
            criterion: torch.nn.Module, 
            validation_timesteps: List[int],
            device: torch.device,
            logger: Optional[BaseLogger] = None,
            epoch: Optional[int] = None) -> Tuple[float, List[float], Dict[int, float]]:
    
    if not hasattr(criterion, 'reduction'):
        raise ValueError("criterion must have a 'reduction' attribute")

    criterion.reduction = 'none'
  
    model = model.to(device)
    model.eval()

    epoch_val_loss = 0.0
    val_num_samples = 0
    epoch_val_loss_per_ts = {ts: 0.0 for ts in validation_timesteps}

    val_loop = tqdm(val_loader, desc="Validation")

    val_batch_losses = [None for _ in range(len(val_loader))]

    with torch.no_grad():

        for batch_idx, images in enumerate(val_loop):
            #normalize the images to the [-1, 1] range
            images = _normalize_images(images)
            images = images.to(device)

            val_num_samples += images.shape[0]
            batch_size = images.shape[0]
            
            if not validation_timesteps or len(validation_timesteps) == 0:
                raise ValueError("validation_timesteps must be provided for validation")

            n_ts = len(validation_timesteps)
            
            # Expand images: [img1, ..., img_N] -> [img1, ..., img_N, img1, ..., img_N, ...]
            # This repeats the whole batch of images for each timestep.
            images_expanded = images.repeat(n_ts, 1, 1, 1)

            # Create timesteps: [ts1, ts2] -> [ts1, ... ts1 (N times), ts2, ... ts2 (N times)]
            # Each timestep is repeated N=batch_size times.
            ts_tensor = torch.tensor(validation_timesteps, device=device, dtype=torch.int64)
            timesteps = ts_tensor.repeat_interleave(batch_size)

            noise = torch.randn(images_expanded.shape, device=device)
            
            noisy_images = noise_scheduler.add_noise(images_expanded, noise, timesteps).to(torch.float32)
            outputs = model.forward(noisy_images, timesteps)


            if hasattr(outputs, 'sample'):
                # for whatever reason, the outputs of UNet2DModel are a wrapper around a tensor.
                outputs = outputs.sample

            loss = criterion(outputs, noise)

            # compute the mean across all dimensions expect the batch dimension
            loss = loss.mean(dim=tuple(range(1, len(loss.shape)))) 
            # at this point the loss must be a tensor of shape (bs * n_ts,)
            # The layout is [loss_img1_ts1, ..., loss_imgN_ts1, loss_img1_ts2, ..., loss_imgN_ts2, ...]

            if loss.shape != (batch_size * n_ts,):
                # sanity check !!
                raise ValueError(f"the loss must be a tensor of shape (bs * n_ts,), but it is {loss.shape}")

            # Reshape to (n_ts, batch_size) to group losses by timestep
            loss_per_ts_view = loss.view(n_ts, batch_size)

            # Update total loss for each timestep
            for i, ts in enumerate(validation_timesteps):
                epoch_val_loss_per_ts[ts] += loss_per_ts_view[i].sum().item()

            loss_value = loss.mean().item()
            val_batch_losses[batch_idx] = loss_value 
            val_loop.set_postfix(loss=loss_value)

            # Update statistics
            epoch_val_loss += loss.sum().item() 
            
            if logger is not None and epoch is not None:
                global_step = epoch * len(val_loader) + batch_idx
                logger.log_scalar('Loss/val_batch', loss_value, global_step)

        avg_loss = round(epoch_val_loss / (val_num_samples * n_ts), 8)

        avg_loss_per_ts = {ts: round(total_loss / val_num_samples, 8) for ts, total_loss in epoch_val_loss_per_ts.items()}

        return avg_loss, val_batch_losses, avg_loss_per_ts


def val_epoch_logging(
                        # metrics from the validation epoch
                        val_loss: float,
                        val_loss_per_ts: Dict[int, float],
                        # variables used to track the validation metrics : make sure they are passed by reference !!!!
                        epoch: int,
                        logger: BaseLogger, 
                        all_val_losses: List[float]):
    """
    This function groups all the logging and tracking logic for the validation epoch
    """
    all_val_losses.append(val_loss)
    # Log epoch losses
    logger.log_scalar('Loss/val_epoch', val_loss, epoch)
    # save the validation epoch loss per timestep        
    for ts, loss in val_loss_per_ts.items():
        logger.log_scalar(f'Loss/val_epoch_ts_{ts}', loss, epoch)



def train_diffusion_model(model: Union[DiffusionUNetOneDim, UNet2DModel], 
               noise_scheduler: DDPMScheduler,
               train_loader: DataLoader, 
               val_loader: Optional[DataLoader], 
               criterion: torch.nn.Module, 
               optimizer: torch.optim.Optimizer, 
               lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
               num_epochs: int, 
               device: torch.device, 
               logger: BaseLogger,
               log_dir: P,
               timestep_bins: List[int],
               validation_timesteps: Optional[list],
               timesteps_sampler_type: str,
               val_per_epoch: int = 5,
               max_grad_norm: float = 1.0,
               debug: bool = False,
               time_steps_kwargs: Dict[str, Any] = {}
               ) -> Union[DiffusionUNetOneDim, UNet2DModel]:
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
    all_train_losses = []
    all_val_losses = []
    
    train_loop = tqdm(range(num_epochs), desc="Training loop")

    timesteps_sampler = set_timesteps_sampler(timesteps_sampler_type, noise_scheduler.config.num_train_timesteps, **time_steps_kwargs)

    for epoch in train_loop:

        # Training phase
        train_loss, batch_losses, train_loss_per_bin = train_epoch(
            model=model,
            noise_scheduler=noise_scheduler,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            timestep_bins=timestep_bins,
            timesteps_sampler=timesteps_sampler,
            logger=logger,
            epoch=epoch,
            max_grad_norm=max_grad_norm,
            debug=debug,
            lr_scheduler=lr_scheduler,
        )

        # get the learning rate of the current epoch:
        # use the optimizer to get the learning rate
        epoch_lr = optimizer.param_groups[0]['lr'] 

        # update the timestep sampler with the loss per bin
        timesteps_sampler.update(train_loss_per_bin)

        # track the metrics
        train_epoch_logging(
            train_loss=train_loss,
            epoch_lr=epoch_lr,
            train_loss_per_bin=train_loss_per_bin,
            epoch=epoch,
            logger=logger,
            train_loop=train_loop,
            all_train_losses=all_train_losses
        )

        # Validation phase
        if val_loader is None:
            print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.8f}")
            continue 

        if validation_timesteps is None:
            raise ValueError("validation_timesteps must be provided when a validation loader is present.")

        val_loss, val_batch_losses, val_loss_per_ts = val_epoch(
            model=model,
            noise_scheduler=noise_scheduler,
            val_loader=val_loader,
            criterion=criterion,
            validation_timesteps=validation_timesteps,
            device=device,
            logger=logger,
            epoch=epoch
        )
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.8f}, Val Loss: {val_loss:.8f}")

        # track the metrics
        val_epoch_logging(
            val_loss=val_loss,
            val_loss_per_ts=val_loss_per_ts,
            epoch=epoch,
            logger=logger,
            all_val_losses=all_val_losses
        )

        if (epoch + 1) % val_per_epoch == 0:
            sample_diffusion_epoch(model, noise_scheduler, val_loader, device, log_dir, epoch, debug)

        # TODO: add FID evaluation, general checkpointing code 

            
    return model

