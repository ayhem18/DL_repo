import os
import torch
from tqdm import tqdm
import torch.nn as nn
import albumentations as A
import torch.optim as optim
from torch.utils.data import DataLoader

import numpy as np
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

from mypt.nets.conv_nets.adaptive_unet import AdaptiveUNet
from mypt.data.datasets.segmentation.semantic_seg import SemanticSegmentationDataset
from mypt.shortcuts import P

def train_loop(model: torch.nn.Module, 
               train_loader: DataLoader, 
               criterion: torch.nn.Module, 
               optimizer: torch.optim.Optimizer, 
               device: torch.device):
    """
    Run one epoch of training.
    
    Args:
        model: The model to train
        train_loader: DataLoader for training data
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on
    
    Returns:
        Average training loss for the epoch
    """
    model.train()
    train_loss = 0.0
    
    train_loop = tqdm(train_loader, desc="Training")
    for images, masks in train_loop:
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
        
        # Update statistics
        train_loss += loss.item() * images.size(0)
        train_loop.set_postfix(loss=loss.item())
    
    return train_loss / len(train_loader)

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
            
            val_loss += loss.item() * images.size(0)
            val_loop.set_postfix(loss=loss.item())
    
    return val_loss / len(val_loader.dataset)

def log_predictions(model, val_loader, writer, epoch, device):
    """Log sample predictions to TensorBoard"""
    model.eval()
    images, masks = next(iter(val_loader))
    images = images.to(device)
    masks = masks.to(device)
    
    with torch.no_grad():
        outputs = model(images)
        predictions = torch.sigmoid(outputs) > 0.5
    
    # Log a few sample images
    for i in range(min(3, len(images))):
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original image
        img = images[i].cpu().permute(1, 2, 0).numpy()
        axes[0].imshow(img)
        axes[0].set_title("Input Image")
        axes[0].axis('off')
        
        # Ground truth mask
        mask = masks[i].cpu().squeeze().numpy()
        axes[1].imshow(mask, cmap='gray')
        axes[1].set_title("Ground Truth")
        axes[1].axis('off')
        
        # Predicted mask
        pred = predictions[i].cpu().squeeze().numpy()
        axes[2].imshow(pred, cmap='gray')
        axes[2].set_title("Prediction")
        axes[2].axis('off')
        
        plt.tight_layout()
        writer.add_figure(f'Predictions/sample_{i}', fig, epoch)
        plt.close(fig)

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device, writer):
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
    
    Returns:
        Trained model
    """
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # Training phase
        train_loss = train_loop(
            model=model,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device
        )
        writer.add_scalar('Loss/train', train_loss, epoch)
        
        # Validation phase
        val_loss = val_loop(
            model=model,
            val_loader=val_loader,
            criterion=criterion,
            device=device
        )
        writer.add_scalar('Loss/val', val_loss, epoch)
        
        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # Save the model if validation loss improves
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pth')
            print(f"Model saved with validation loss: {val_loss:.4f}")
        
        # Log sample predictions
        if epoch % 5 == 0:
            log_predictions(model, val_loader, writer, epoch, device)
    
    return model

