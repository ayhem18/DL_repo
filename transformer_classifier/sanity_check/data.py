import torch


# Add the path to the pytorch_modular package
# sys.path.append(os.path.join(os.path.dirname(__file__), '../../../ML/pytorch_modular/src'))

from config import TransformerConfig
from mypt.data.datasets.synthetic.sequence.sequence_cls import SyntheticSequenceClsDataset
from mypt.data.dataloaders.standard_dataloaders import initialize_train_dataloader, initialize_val_dataloader

def get_dataloaders(config: TransformerConfig):
    """
    Create train, validation, and test dataloaders from synthetic sequence data.
    
    Args:
        config: Configuration object with dataset parameters
        
    Returns:
        train_loader, val_loader, test_loader: DataLoader objects for each split
    """

    # Create datasets with different seeds to ensure independence
    train_dataset = SyntheticSequenceClsDataset(
        max_len=config.max_len,
        num_samples=config.train_samples,
        dim=config.dim,
        seed=config.data_seed,
        max_mean=config.max_mean,
        all_same_length=config.all_same_length
    )
    
    val_dataset = SyntheticSequenceClsDataset(
        max_len=config.max_len,
        num_samples=config.val_samples,
        dim=config.dim,
        seed=config.data_seed + 1,  # Different seed for validation
        max_mean=config.max_mean,
        all_same_length=config.all_same_length
    )
    
    test_dataset = SyntheticSequenceClsDataset(
        max_len=config.max_len,
        num_samples=config.test_samples,
        dim=config.dim,
        seed=config.data_seed + 2,  # Different seed for test
        max_mean=config.max_mean,
        all_same_length=config.all_same_length
    )
    


    # Create dataloaders
    train_loader = initialize_train_dataloader(train_dataset, 
                                               seed=config.data_seed, 
                                               batch_size=config.batch_size,
                                               num_workers=2)
    
    val_loader = initialize_val_dataloader(val_dataset, 
                                           seed=config.data_seed + 1, 
                                           batch_size=config.batch_size,
                                           num_workers=2)
    
    test_loader = initialize_val_dataloader(test_dataset, 
                                            seed=config.data_seed + 2, 
                                            batch_size=config.batch_size,
                                            num_workers=2)
    
    return train_loader, val_loader, test_loader


def prepare_batch(batch: tuple[torch.Tensor, torch.Tensor], 
                  device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """
    Move batch data to device and create padding mask if needed.
    
    Args:
        batch: Tuple of (sequences, labels)
        device: Device to move data to
        
    Returns:
        sequences, labels, padding_mask (None for fixed-length data)
    """
    sequences, labels = batch
    sequences = sequences.to(device)
    labels = labels.to(device)
    
    # For this synthetic dataset with all_same_length=True, we don't need padding masks
    # But we include this for completeness in case variable lengths are used
    padding_mask = None
    
    return sequences, labels, padding_mask


if __name__ == "__main__":
    # Test data loading
    config = TransformerConfig()
    train_loader, val_loader, test_loader = get_dataloaders(config)
    
    # Check a few batches
    for i, batch in enumerate(train_loader):
        sequences, labels = batch
        print(f"Batch {i+1}:")
        print(f"  Sequences shape: {sequences.shape}")
        print(f"  Labels shape: {labels.shape}")
        print(f"  Labels distribution: {torch.bincount(labels)}")
        
        if i >= 2:  # Just check a few batches
            break
