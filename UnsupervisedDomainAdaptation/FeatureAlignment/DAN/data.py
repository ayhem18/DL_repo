
import torchvision.transforms as tr
import src.utilities.directories_and_files as dirf

from torch.utils.data import DataLoader
from pathlib import Path
from torchvision.datasets import ImageFolder
from typing import Union
from src.data.dataloaders.standard_dataloaders import initialize_train_dataloader

def get_dataloader(root: Union[str, Path],
                   image_transform:tr, 
                   batch_size: int,
                   seed: int, 
                   num_workers: int = 2) -> DataLoader:
    
    dirf.process_path(root, dir_ok=True, 
                      file_ok=False, 
                      condition=dirf.image_dataset_directory, 
                      error_message=dirf.image_dataset_directory_error(root))
    
    # initialize the dataloader
    ds = ImageFolder(root,transform=image_transform)
    
    return initialize_train_dataloader(ds, 
                                       seed=seed, 
                                       batch_size=batch_size,
                                       num_workers=num_workers)
