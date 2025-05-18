import os
import torch

import albumentations as A

from pathlib import Path
from typing import Tuple
from dataclasses import dataclass

import mypt.code_utils.pytorch_utils as pu

from mypt.code_utils import directories_and_files as dirf
from mypt.nets.conv_nets.adaptive_unet import AdaptiveUNet
from mypt.data.datasets.segmentation.semantic_seg import SemanticSegmentationDS
from mypt.data.dataloaders.standard_dataloaders import initialize_train_dataloader, initialize_val_dataloader

from train import train_model

SCRIPT_DIR = Path(__file__).parent
current_dir = SCRIPT_DIR
while 'data' not in os.listdir(current_dir):
    current_dir = current_dir.parent

DATA_DIR = os.path.join(current_dir, 'data') 

from torch.utils.tensorboard import SummaryWriter

@dataclass
class TrainConfig:
    train_batch_size: int = 8
    val_batch_size: int = 8
    num_epochs: int = 2
    learning_rate: float = 1e-4
    
    input_shape: Tuple[int, int, int] = (3, 256, 256)
    output_shape: Tuple[int, int, int] = (1, 256, 256)

    bottleneck_shape: Tuple[int, int, int] = (256, 256, 256)
    bottleneck_out_channels: int = 256
    
    input_channels: int = 3
    output_channels: int = 1

    seed: int = 42



def _set_data(config: TrainConfig):
    train_data = os.path.join(DATA_DIR, 'working_version', 'train')
    train_masks = os.path.join(DATA_DIR, 'working_version', 'train_masks')

    val_data = os.path.join(DATA_DIR, 'working_version', 'val')
    val_masks = os.path.join(DATA_DIR, 'working_version', 'val_masks')

    if any(not os.path.exists(dir) for dir in [train_data, train_masks, val_data, val_masks]):
        raise FileNotFoundError(f"Please follow the instructions in the README.md file to prepare the data !!!")

    train_transforms = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.pytorch.ToTensorV2(),
    ])
    
    val_transforms = A.Compose([
        A.pytorch.ToTensorV2(),
    ])

    # Create datasets
    train_dataset = SemanticSegmentationDS(
        data_dir=train_data,
        mask_dir=train_masks,
        transforms=train_transforms
    )
    
    val_dataset = SemanticSegmentationDS(
        data_dir=val_data,
        mask_dir=val_masks,
        transforms=val_transforms
    )
    
    # Create data loaders
    train_loader = initialize_train_dataloader(train_dataset, seed=42, batch_size=config.train_batch_size, num_workers=2, drop_last=True)
    val_loader = initialize_val_dataloader(val_dataset, seed=42, batch_size=config.val_batch_size, num_workers=2)

    return train_loader, val_loader




def main():
    # Set device
    config = TrainConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pu.seed_everything(config.seed)

    train_loader, val_loader = _set_data(config)    

    
    # Initialize model
    model = AdaptiveUNet(
        input_shape=(config.input_channels, config.input_shape[1], config.input_shape[2]),
        output_shape=(config.output_channels, config.output_shape[1], config.output_shape[2]),
        bottleneck_shape=(config.bottleneck_shape[0], config.input_shape[1] // 8, config.input_shape[2] // 8),
        bottleneck_out_channels=config.bottleneck_out_channels
    )
    
    # Build the model architecture
    model.build_contracting_path(max_conv_layers_per_block=3, min_conv_layers_per_block=2)
    model.build_bottleneck(kernel_sizes=3, num_blocks=3, conv_layers_per_block=2)
    model.build_expanding_path(max_conv_layers_per_block=3, min_conv_layers_per_block=2)
    model.build()
    
    # Move model to device
    model = model.to(device)
    
    # Define loss function and optimizer
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    logs_dir = dirf.process_path(os.path.join(SCRIPT_DIR, 'runs'), dir_ok=True, file_ok=False)

    # Initialize TensorBoard writer
    writer = SummaryWriter(dirf.process_path(os.path.join(logs_dir, f'run_{len(os.listdir(logs_dir)) + 1}'), dir_ok=True, file_ok=False))
    
    # Train the model
    trained_model = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=config.num_epochs,
        device=device,
        writer=writer
    )
    
    # Save the final model
    torch.save(trained_model.state_dict(), 'final_model.pth')
    
    # Clean up temporary directories
    import shutil
    shutil.rmtree('temp')
    
    print("Training completed!")

if __name__ == '__main__':
    main()




