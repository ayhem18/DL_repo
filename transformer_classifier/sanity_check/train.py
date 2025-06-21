import os
import shutil
import time
import json

import torch
import torchmetrics
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from typing import Dict, Callable, Tuple

from mypt.shortcuts import P
from mypt.code_utils import pytorch_utils as pu 
from mypt.code_utils import directories_and_files as dirf
from mypt.nets.transformers.transformer_classifier import TransformerClassifier

from config import TransformerConfig
from data import get_dataloaders, prepare_batch


class EarlyStopping:
    """Early stopping to prevent overfitting."""
    
    def __init__(self, path_dir: P, patience=5, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float('inf')
        self.delta = delta
        self.path_dir = dirf.process_path(path_dir, dir_ok=True, file_ok=False, must_exist=True)
        self.path = os.path.join(self.path_dir, 'best_model.pt')

    def __call__(self, val_loss, model):
        score = -val_loss
        
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0
            
    def save_checkpoint(self, val_loss, model):
        """Save model when validation loss decreases."""
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


def train_epoch(model: TransformerClassifier, 
                train_loader: DataLoader, 
                criterion: nn.Module, 
                optimizer: optim.Optimizer, 
                device: torch.device, 
                epoch: int, 
                writer: SummaryWriter, 
                config: TransformerConfig,
                metrics: Dict[str, torchmetrics.Metric]) -> Tuple[float, Dict[str, float]]:
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    total_samples = 0

    epoch_predictions = [None for _ in range(len(train_loader))]
    epoch_labels = [None for _ in range(len(train_loader))]

    for batch_idx, batch in enumerate(train_loader):
        # Get data
        sequences, labels, padding_mask = prepare_batch(batch, device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(sequences, padding_mask)
        
        # Convert to binary classification (0/1)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        batch_size = labels.size(0)
        running_loss += loss.item() * batch_size
        total_samples += batch_size
        
        # Calculate predictions
        predictions = (outputs > 0.0).float()  # Threshold at 0.0
        
        # save the predictions and labels
        epoch_predictions[batch_idx] = predictions
        epoch_labels[batch_idx] = labels

        if batch_idx % config.log_interval != 0:
            continue

        # Log batch results        
        batch_loss = loss.item()
        step = epoch * len(train_loader) + batch_idx

        # Log to tensorboard
        writer.add_scalar('train/batch_loss', batch_loss, step)

        # compute the metrics for the batch
        for name, metric in metrics.items():
            batch_metric_value = metric(predictions, labels)
            writer.add_scalar(f"train/batch_{name}", batch_metric_value.cpu().item(), step)
            
            
    
    # Epoch statistics
    epoch_loss = running_loss / total_samples
    
    # Compute final metrics for the epoch
    epoch_metrics = {}
    for name, metric in metrics.items():
        epoch_metrics[f"train_epoch_{name}"] = metric(epoch_predictions, epoch_labels).cpu().item()
    
    return epoch_loss, epoch_metrics


def validation_epoch(model: TransformerClassifier, 
             val_loader: DataLoader, 
             criterion: nn.Module, 
             device: torch.device,
             metrics: Dict[str, torchmetrics.Metric]) -> Tuple[float, Dict[str, float]]:
    """Validate the model."""
    model.eval()
    val_loss = 0.0
    total_samples = 0
    
    epoch_predictions = [None for _ in range(len(val_loader))]
    epoch_labels = [None for _ in range(len(val_loader))]

    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            # Get data
            sequences, labels, padding_mask = prepare_batch(batch, device)
            
            # Forward pass
            outputs = model(sequences, padding_mask)
            
            # Calculate loss
            loss = criterion(outputs, labels)
            
            # Statistics
            batch_size = labels.size(0)
            val_loss += loss.item() * batch_size
            total_samples += batch_size
            
            # Calculate predictions
            predictions = (outputs > 0.0).float()  # Threshold at 0.0
            
            # save the predictions and labels
            epoch_predictions[batch_idx] = predictions
            epoch_labels[batch_idx] = labels
    
    # Compute final metrics
    val_metrics = {}
    for name, metric in metrics.items():
        val_metrics[f"val_{name}"] = metric(epoch_predictions, epoch_labels).cpu().item()
    
    # Average loss
    val_loss /= total_samples
    
    return val_loss, val_metrics


def test(model: TransformerClassifier, 
         test_loader: DataLoader, 
         criterion: nn.Module, 
         device: torch.device,
         metrics: Dict[str, torchmetrics.Metric]) -> Tuple[float, Dict[str, float]] :
    """Test the model."""
    model.eval()
    test_loss = 0.0
    total_samples = 0
    
    epoch_predictions = [None for _ in range(len(test_loader))]
    epoch_labels = [None for _ in range(len(test_loader))]
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            # Get data
            sequences, labels, padding_mask = prepare_batch(batch, device)
            
            # Forward pass
            outputs = model(sequences, padding_mask)
            
            # Calculate loss
            loss = criterion(outputs, labels)
            
            # Statistics
            batch_size = labels.size(0)
            test_loss += loss.item() * batch_size
            total_samples += batch_size
            
            # Calculate predictions
            predictions = (outputs > 0.0).float()  # Threshold at 0.0
            
            # save the predictions and labels
            epoch_predictions[batch_idx] = predictions
            epoch_labels[batch_idx] = labels
    
    # Compute final metrics
    test_metrics = {}
    for name, metric in metrics.items():
        test_metrics[f"test_{name}"] = metric(epoch_predictions, epoch_labels).cpu().item()
    
    # Average loss
    test_loss /= total_samples
    
    return test_loss, test_metrics


def prepare_log_directory(log_parent_dir: P) -> Tuple[P, P, P]:
    # iterate through each "run_*" directory and remove any folder that does not contain a '.json' file
    for r in os.listdir(log_parent_dir):
        run_dir = False
        if os.path.isdir(os.path.join(log_parent_dir, r)):
            for file in os.listdir(os.path.join(log_parent_dir, r)):
                if os.path.splitext(file)[-1] == '.json':
                    run_dir = True
                    break
            
            if not run_dir:
                shutil.rmtree(os.path.join(log_parent_dir, r))

    exp_dir = dirf.process_path(os.path.join(log_parent_dir, f'run_{len(os.listdir(log_parent_dir)) + 1}'), dir_ok=True, file_ok=False)
    log_dir = dirf.process_path(os.path.join(exp_dir, "logs"), dir_ok=True, file_ok=False, must_exist=False)
    checkpoints_dir = dirf.process_path(os.path.join(exp_dir, "checkpoints"), dir_ok=True, file_ok=False, must_exist=False)

    return exp_dir, log_dir, checkpoints_dir



def initialize_metrics() -> Dict[str, torchmetrics.Metric]:
    """Initialize metrics for binary classification."""
    metrics = {
        "accuracy": torchmetrics.Accuracy(task="binary"),
        "f1": torchmetrics.F1Score(task="binary"),
        "precision": torchmetrics.Precision(task="binary"),
        "recall": torchmetrics.Recall(task="binary")
    }
    return metrics




def train_model(config: TransformerConfig):
    """Main training function."""
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    pu.seed_everything(config.model_seed)

    # prepare the log directory
    exp_dir, log_dir, checkpoints_dir = prepare_log_directory(config.log_parent_dir)

    # Create tensorboard writer
    writer = SummaryWriter(log_dir=log_dir)
    
    
    # Get data loaders
    train_loader, val_loader, test_loader = get_dataloaders(config)
    
    # Create model
    model = TransformerClassifier(
        d_model=config.d_model,
        num_transformer_blocks=config.num_transformer_blocks,
        num_classification_layers=config.num_classification_layers,
        num_heads=config.num_heads,
        value_dim=config.value_dim,
        key_dim=config.key_dim,
        num_classes=1,  # Binary classification
        pooling=config.pooling,
        dropout=config.dropout
    )
    model = model.to(device)
    
    # Define loss and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    
    # Initialize metrics
    metrics = initialize_metrics()
    
    early_stopping = EarlyStopping(path_dir=checkpoints_dir, patience=config.early_stopping_patience, verbose=True)
    
    # Training loop
    start_time = time.time()
    for epoch in range(1, config.epochs + 1):
        
        # Train
        train_loss, train_metrics = train_epoch(model, train_loader, criterion, optimizer, device, epoch, writer, config, metrics)
        
        # Validate
        val_loss, val_metrics = validation_epoch(model, val_loader, criterion, device, metrics)
        
        
        # Log to tensorboard
        writer.add_scalar('train/epoch_loss', train_loss, epoch)
        
        # Log all metrics
        for name, value in train_metrics.items():
            writer.add_scalar(f'train/{name}', value, epoch)
            
        for name, value in val_metrics.items():
            writer.add_scalar(f'val/{name}', value, epoch)
        
        # Early stopping
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping triggered")
            break
    
    # Load best model
    model.load_state_dict(torch.load(early_stopping.path))
    
    # Save the config when the training is completed
    config_path = os.path.join(exp_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(config.to_dict(), f, indent=4)

    # Test
    test_loss, test_metrics = test(model, test_loader, criterion, device, metrics)
    print(f'Test Loss: {test_loss:.4f} | Test Acc: {test_metrics["test_accuracy"]:.2f}')
    
    # Log test results
    writer.add_scalar('test/loss', test_loss, 0)
    
    # Log all test metrics
    for name, value in test_metrics.items():
        writer.add_scalar(f'test/{name}', value, 0)
    
    # Save final results
    results = {
        'train_loss': train_loss,
        'val_loss': val_loss,
        'test_loss': test_loss,
        'total_time': time.time() - start_time,
        **train_metrics,
        **val_metrics,
        **test_metrics
    }
    
    results_path = os.path.join(exp_dir, 'results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4)
    
    writer.close()
    
    return model, results


if __name__ == "__main__":
    config = TransformerConfig()
    model, results = train_model(config)
    print("Training completed!")
