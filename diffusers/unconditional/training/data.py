import os
import shutil
import numpy as np
import albumentations as A

from pathlib import Path
from typing import Tuple, Union
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader


from mypt.shortcuts import P
from mypt.code_utils import directories_and_files as dirf
from mypt.data.dataloaders.standard_dataloaders import initialize_train_dataloader, initialize_val_dataloader


from mnist_dataset.mnist import MnistDSWrapper
from training.config import ModelConfig, TrainingConfig



SCRIPT_DIR = Path(__file__).parent
current_dir = SCRIPT_DIR
while 'data' not in os.listdir(current_dir):
    current_dir = current_dir.parent

DATA_DIR = os.path.join(current_dir, 'data') 


class HuggingFaceDatasetWrapper(Dataset):
    def __init__(self, dataset, transforms):
        self.dataset = dataset
        self.transforms = transforms
        
    def __len__(self):
        return len(self.dataset)
        
    def __getitem__(self, idx):
        image = self.dataset[idx]["image"].convert("RGB")
        transformed = self.transforms(image=np.asarray(image))
        res = transformed["image"]

        return res


def set_butterfiles_dataset(model_config: ModelConfig, train_config: TrainingConfig) -> Tuple[DataLoader, DataLoader]:

    dataset_name = "huggan/smithsonian_butterflies_subset"
    train_ds = load_dataset(dataset_name, split="train[:95%]")    
    val_ds = load_dataset(dataset_name, split="train[-5%:]")     # load the last 10% of the training data as validation data

    train_transforms = A.Compose([
        A.Resize(height=model_config.input_shape[1], width=model_config.input_shape[2]),
        A.ToTensorV2()
    ])

    val_transforms = A.Compose([
        A.Resize(height=model_config.input_shape[1], width=model_config.input_shape[2]),
        A.ToTensorV2()
    ])
    
            
    # Apply transformations
    train_ds_transformed = HuggingFaceDatasetWrapper(train_ds, train_transforms)
    val_ds_transformed = HuggingFaceDatasetWrapper(val_ds, val_transforms)
    
    train_loader = initialize_train_dataloader(train_ds_transformed, seed=42, batch_size=train_config.train_batch_size, num_workers=3, drop_last=True)
    val_loader = initialize_val_dataloader(val_ds_transformed, seed=42, batch_size=train_config.val_batch_size, num_workers=3)

    return train_loader, val_loader



def set_mnist_dataset(model_config: ModelConfig, 
             train_config: TrainingConfig, 
            #  return_datasets: bool = False
             ) -> Tuple[DataLoader, DataLoader]:
 
    train_data_path = os.path.join(SCRIPT_DIR, 'data', 'train')
    val_data_path = os.path.join(SCRIPT_DIR, 'data', 'val')

    train_transforms = [
        # A.RandomResizedCrop(size=model_config.input_shape[1:], scale=(0.8, 1)),
        A.ToTensorV2()
    ]
    
    val_transforms = [
        # A.RandomResizedCrop(size=model_config.input_shape[1:], scale=(0.8, 1)),
        A.ToTensorV2()
    ]

    # create the datasets
    train_ds = MnistDSWrapper(root=train_data_path, 
                              train=True, 
                              transforms=train_transforms, 
                              output_shape=model_config.input_shape[1:],
                              unconditional=True
                              )
    
    val_ds = MnistDSWrapper(root=val_data_path, 
                            train=False, 
                            transforms=val_transforms, 
                            output_shape=model_config.input_shape[1:],
                            unconditional=True
                            )


    # Create data loaders
    train_loader = initialize_train_dataloader(train_ds, seed=42, batch_size=train_config.train_batch_size, num_workers=3, drop_last=True)
    val_loader = initialize_val_dataloader(val_ds, seed=42, batch_size=train_config.val_batch_size, num_workers=3)

    # if return_datasets:
    #     return train_loader, val_loader, train_ds, val_ds

    return train_loader, val_loader


def set_data(model_config: ModelConfig, train_config: TrainingConfig) -> Tuple[DataLoader, DataLoader]:
    if train_config.dataset == "butterflies":
        return set_butterfiles_dataset(model_config, train_config)
    elif train_config.dataset == "mnist":
        return set_mnist_dataset(model_config, train_config)
    else:
        raise ValueError(f"Invalid dataset type: {train_config.dataset}") 


def prepare_log_directory(checkpoint_tolerant: bool) -> Path:
    logs_dir = dirf.process_path(os.path.join(SCRIPT_DIR, 'runs'), dir_ok=True, file_ok=False)

    # iterate through each "run_*" directory and remove any folder that does not contain a '.json' file
    for r in os.listdir(logs_dir):
        run_dir = False
        if os.path.isdir(os.path.join(logs_dir, r)):
            for file in os.listdir(os.path.join(logs_dir, r)):
                if os.path.splitext(file)[-1] == '.json':
                    run_dir = True
                    break
            
            if not run_dir and not checkpoint_tolerant:
                shutil.rmtree(os.path.join(logs_dir, r))

    exp_log_dir = dirf.process_path(os.path.join(logs_dir, f'run_{len(os.listdir(logs_dir)) + 1}'), dir_ok=True, file_ok=False)
    return exp_log_dir
