import os
import torch

import numpy as np

from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import matplotlib.pyplot as plt

from mypt.shortcuts import P


def train_loop(model: torch.nn.Module, 
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
    model.train()
    train_loss = 0.0
    batch_losses = [None for _ in range(len(train_loader))]
    
    train_loop = tqdm(train_loader, desc="Training")

    for batch_idx, (images, masks) in enumerate(train_loop):
        images = images.to(device)
        masks = masks.to(device)
        
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, masks)
        
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

def val_loop(model, val_loader, criterion, device):
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
    model.eval()
    val_loss = 0.0
    
    with torch.no_grad():
        val_loop = tqdm(val_loader, desc="Validation")
        for images, masks in val_loop:
            images = images.to(device)
            masks = masks.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            val_loss += loss.item()
            val_loop.set_postfix(loss=loss.item())
    
    return val_loss / len(val_loader)

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

def train_model(model: torch.nn.Module, 
               train_loader: DataLoader, 
               val_loader: DataLoader, 
               criterion: torch.nn.Module, 
               optimizer: torch.optim.Optimizer, 
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
        train_loss, batch_losses = train_loop(
            model=model,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            writer=writer,
            epoch=epoch
        )
        writer.add_scalar('Loss/train_epoch', train_loss, epoch)
        all_train_losses.append(train_loss)
        
        # Validation phase
        val_loss = val_loop(
            model=model,
            val_loader=val_loader,
            criterion=criterion,
            device=device
        )
        writer.add_scalar('Loss/val_epoch', val_loss, epoch)
        all_val_losses.append(val_loss)
        
        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")        

        # Save the model if validation loss improves
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(log_dir, 'best_model.pth'))
            print(f"Model saved with validation loss: {val_loss:.4f}")
        
        # Log sample predictions
        if epoch % 5 == 0:
            log_predictions(model, val_loader, writer, epoch, device)
            
    return model

