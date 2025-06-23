import os
import time

import torch
import torchmetrics
import torch.nn as nn
from tqdm import tqdm
import torch.optim as optim
from collections import defaultdict
from typing import Dict, List, Tuple    
from torch.utils.data import DataLoader

from mypt.shortcuts import P
from mypt.loggers.base import BaseLogger
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
                logger: BaseLogger, 
                config: TransformerConfig,
                metrics: Dict[str, torchmetrics.Metric]) -> Tuple[float, Dict[str, float]]:
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    total_samples = 0

    # move the metrics to the device
    for key, metric in metrics.items():
        metrics[key] = metric.to(device)

    epoch_predictions = [None for _ in range(len(train_loader))]
    epoch_labels = [None for _ in range(len(train_loader))]

    for batch_idx, batch in enumerate(train_loader):
        # Get data
        sequences, labels, padding_mask = prepare_batch(batch, device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model.forward(sequences, padding_mask).squeeze(-1)

        if torch.any(torch.isnan(outputs)):
            print(f"NaN outputs at epoch {epoch}")
            print(f"NaN outputs at batch {batch_idx}")
            print(f"Outputs: {outputs}")
            print(f"Labels: {labels}")
            print(f"Padding mask: {padding_mask}")
            raise ValueError("NaN outputs")

        # Convert to binary classification (0/1)
        loss = criterion.forward(outputs, labels.float())
        
        if torch.any(torch.isnan(loss)):
            print(f"Nan loss at epoch {epoch}")
            print(f"NaN loss at batch {batch_idx}")
            print(f"Outputs: {outputs}")
            print(f"Labels: {labels}")
            print(f"Padding mask: {padding_mask}")
            raise ValueError("NaN loss")

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
        log_batch_metrics = {f"train/batch_{name}": metric(predictions, labels).cpu().item() for name, metric in metrics.items()}
        
        logger.log_scalar('train/batch_loss', batch_loss, step)
        logger.log_dict(log_batch_metrics, step)

            
    
    # Epoch statistics
    epoch_loss = running_loss / total_samples
    
    # Compute final metrics for the epoch
    epoch_metrics = {}
    for name, metric in metrics.items():
        epoch_metrics[f"train_epoch_{name}"] = metric(torch.cat(epoch_predictions), torch.cat(epoch_labels)).cpu().item()
    
    return epoch_loss, epoch_metrics


def validation_epoch(model: TransformerClassifier, 
             val_loader: DataLoader, 
             criterion: nn.Module, 
             device: torch.device,
             logger: BaseLogger,
             config: TransformerConfig,
             metrics: Dict[str, torchmetrics.Metric],
             epoch: int) -> Tuple[float, Dict[str, float]]:
    """Validate the model."""
    model.eval()
    val_loss = 0.0
    total_samples = 0
    
    # move the metrics to the device
    for key, metric in metrics.items():
        metrics[key] = metric.to(device)

    epoch_predictions = [None for _ in range(len(val_loader))]
    epoch_labels = [None for _ in range(len(val_loader))]

    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            # Get data
            sequences, labels, padding_mask = prepare_batch(batch, device)
            
            # Forward pass
            outputs = model(sequences, padding_mask).squeeze(-1)
            
            # Calculate loss
            loss = criterion.forward(outputs, labels.float())
            
            # Statistics
            batch_size = labels.size(0)
            val_loss += loss.item() * batch_size
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
            step = epoch * len(val_loader) + batch_idx
            
            # Log to tensorboard
            log_batch_metrics = {f"val/batch_{name}": metric(predictions, labels).cpu().item() for name, metric in metrics.items()}

            logger.log_scalar('val/batch_loss', batch_loss, step)
            logger.log_dict(log_batch_metrics, step)

    
    # Compute final metrics
    val_metrics = {}
    for name, metric in metrics.items():
        val_metrics[f"val_{name}"] = metric(torch.cat(epoch_predictions), torch.cat(epoch_labels)).cpu().item()
    
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
    
    for key, metric in metrics.items():
        metrics[key] = metric.to(device)

    epoch_predictions = [None for _ in range(len(test_loader))]
    epoch_labels = [None for _ in range(len(test_loader))]
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            # Get data
            sequences, labels, padding_mask = prepare_batch(batch, device)
            
            # Forward pass
            outputs = model(sequences, padding_mask).squeeze(-1)
            
            # Calculate loss
            loss = criterion.forward(outputs, labels.float())
            
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
        test_metrics[f"test_{name}"] = metric(torch.cat(epoch_predictions), torch.cat(epoch_labels)).cpu().item()
    
    # Average loss
    test_loss /= total_samples
    
    return test_loss, test_metrics



def initialize_metrics() -> Dict[str, torchmetrics.Metric]:
    """Initialize metrics for binary classification."""
    metrics = {
        "accuracy": torchmetrics.Accuracy(task="binary"),
        "f1": torchmetrics.F1Score(task="binary"),
        "precision": torchmetrics.Precision(task="binary"),
        "recall": torchmetrics.Recall(task="binary")
    }
    return metrics



def train_model(
                # model arguments   
                model: TransformerClassifier, 
                config: TransformerConfig, 
                metrics: Dict[str, torchmetrics.Metric],
                
                # data arguments
                train_loader: DataLoader, 
                val_loader: DataLoader, 
                
                # optimization arguments
                criterion: nn.Module, 
                optimizer: optim.Optimizer, 
                early_stopping: EarlyStopping,

                logger: BaseLogger,
                device: torch.device,
                log_checkpoints_dir: P
                ) -> Tuple[List[float], Dict[str, float]]: 
    
    run_train_losses = [None for _ in range(config.epochs)]
    run_train_metrics = defaultdict(list)

    run_val_losses = [None for _ in range(config.epochs)]
    run_val_metrics = defaultdict(list)

    # save the starting time
    start_time = time.time()

    for epoch in tqdm(range(1, config.epochs + 1), desc="Training"):
        
        # Train
        epoch_train_loss, epoch_train_metrics = train_epoch(model, train_loader, criterion, optimizer, device, epoch, logger, config, metrics)

        # Log
        logger.log_scalar('train/epoch_loss', epoch_train_loss, epoch)        
        train_metrics_log_dict = {f"train/{name}": value for name, value in epoch_train_metrics.items()}
        logger.log_dict(train_metrics_log_dict, epoch)

        # save the metrics for the epoch
        run_train_losses[epoch - 1] = epoch_train_loss

        for name, value in epoch_train_metrics.items():
            run_train_metrics[name].append(value)

        # Validate
        epoch_val_loss, epoch_val_metrics = validation_epoch(model, val_loader, criterion, device, logger, config, metrics, epoch)

        # Log
        logger.log_scalar('val/epoch_loss', epoch_val_loss, epoch)        
        val_metrics_log_dict = {f"val/{name}": value for name, value in epoch_val_metrics.items()}
        logger.log_dict(val_metrics_log_dict, epoch)

        # save the metrics for the epoch
        run_val_losses[epoch - 1] = epoch_val_loss

        for name, value in epoch_val_metrics.items():
            run_val_metrics[name].append(value)


        # Early stopping
        early_stopping(epoch_val_loss, model)
        
        if early_stopping.early_stop:
            print("Early stopping triggered")
            break

    # save the last model regardless 
    torch.save(model.state_dict(), os.path.join(log_checkpoints_dir, 'last_model.pt'))

    # compute the finish time
    print(f"Training time: {time.time() - start_time:.2f} seconds")


    # remove the None values in the lists
    run_train_losses = [loss for loss in run_train_losses if loss is not None]
    run_val_losses = [loss for loss in run_val_losses if loss is not None]

    # return metrics and losses
    return run_train_losses, run_train_metrics, run_val_losses, run_val_metrics

    
def test_model(model: TransformerClassifier, 
               test_loader: DataLoader, 
               criterion: nn.Module, 
               device: torch.device, 
               metrics: Dict[str, torchmetrics.Metric],
               logger: BaseLogger) -> Tuple[float, Dict[str, float]]:
    """Test the model."""
    # Test
    test_loss, test_metrics = test(model, test_loader, criterion, device, metrics)

    log_test_metrics = {f"test/{name}": value for name, value in test_metrics.items()}
    logger.log_scalar('test/loss', test_loss, 0)
    logger.log_dict(log_test_metrics, 0)

    return test_loss, test_metrics




def run_experiment(config: TransformerConfig, 
                   configs_logger: BaseLogger,
                   metrics_logger: BaseLogger, 
                   log_checkpoints_dir: P):
    """Main training function."""
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    pu.seed_everything(config.model_seed)

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
        num_classes=2,
        pooling=config.pooling,
        dropout=config.dropout
    )
    model = model.to(device)
    
    # Define loss and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    
    # Initialize metrics
    metrics = initialize_metrics()
    
    early_stopping = EarlyStopping(path_dir=log_checkpoints_dir, patience=config.early_stopping_patience, verbose=True)
    
    # Training loop
    run_train_losses, run_train_metrics, run_val_losses, run_val_metrics = train_model(
        model, config, metrics, train_loader, val_loader, criterion, optimizer, early_stopping, metrics_logger, device, log_checkpoints_dir
    )

    # Load best model
    model.load_state_dict(torch.load(early_stopping.path))
    
    configs_logger.log_config(config.to_dict(), 'config')

    # run it on the test set
    test_loss, test_metrics = test_model(model, test_loader, criterion, device, metrics, metrics_logger)

    # Save final results
    results = {
        'train_losses': run_train_losses,
        'val_losses': run_val_losses,
        'test_loss': test_loss,
        **run_train_metrics,
        **run_val_metrics,
        **test_metrics
    }
    
    configs_logger.save(results, 'results')

    return model, results