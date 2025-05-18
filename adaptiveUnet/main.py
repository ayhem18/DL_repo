import os
import json
import numpy as np
import torch
import shutil

import albumentations as A


from pathlib import Path
from typing import List, Optional, Tuple
from dataclasses import dataclass, asdict
from torch.utils.tensorboard import SummaryWriter

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


@dataclass
class TrainConfig:
    train_batch_size: int = 32
    val_batch_size: int = 64
    num_epochs: int = 2
    learning_rate: float = 1e-4
    
    input_shape: Tuple[int, int, int] = (3, 200, 200)
    output_shape: Tuple[int, int, int] = (3, 200, 200)

    bottleneck_shape: Tuple[int, int, int] = (128, 32, 32)
    bottleneck_out_channels: int = 256
    
    seed: int = 42

    def save(self, filepath: str) -> None:
        """Save the config to a JSON file"""
        # Convert the config to a dictionary
        config_dict = asdict(self)
        
        # Convert tuples to lists for JSON serialization
        for key, value in config_dict.items():
            if isinstance(value, tuple):
                config_dict[key] = list(value)
        
        # Save to JSON file
        with open(filepath, 'w') as f:
            json.dump(config_dict, f, indent=4)

    @classmethod
    def load(cls, filepath: str) -> 'TrainConfig':
        """Load the config from a JSON file"""
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        
        # Convert lists back to tuples
        for key, value in config_dict.items():
            if isinstance(value, list):
                config_dict[key] = tuple(value)
        
        # Create a new instance of TrainConfig with the loaded values
        return cls(**config_dict)



def _set_data(config: TrainConfig):
    train_data = os.path.join(DATA_DIR, 'working_version', 'train')
    train_masks = os.path.join(DATA_DIR, 'working_version', 'train_masks')

    val_data = os.path.join(DATA_DIR, 'working_version', 'val')
    val_masks = os.path.join(DATA_DIR, 'working_version', 'val_masks')

    if any(not os.path.exists(dir) for dir in [train_data, train_masks, val_data, val_masks]):
        raise FileNotFoundError(f"Please follow the instructions in the README.md file to prepare the data !!!")

    train_transforms = A.Compose([
        A.Resize(height=config.input_shape[1], width=config.input_shape[2]),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.ToTensorV2(),
    ])
    
    val_transforms = A.Compose([
        A.Resize(height=config.output_shape[1], width=config.output_shape[2]),
        A.ToTensorV2(),
    ])

    # Create datasets
    train_dataset = SemanticSegmentationDS(
        data_dir=train_data,
        mask_dir=train_masks,
        transforms=train_transforms,
        binary_mask=True
    )
    
    val_dataset = SemanticSegmentationDS(
        data_dir=val_data,
        mask_dir=val_masks,
        transforms=val_transforms,
        binary_mask=True
    )
    
    # Create data loaders
    train_loader = initialize_train_dataloader(train_dataset, seed=42, batch_size=config.train_batch_size, num_workers=2, drop_last=True)
    val_loader = initialize_val_dataloader(val_dataset, seed=42, batch_size=config.val_batch_size, num_workers=2)

    return train_loader, val_loader




def visualize_data(images: List):
    import cv2
    for i, img in enumerate(images):
        if isinstance(img, torch.Tensor):
            img = img.transpose(0, 2, 3).cpu().numpy()
        
        img = img * 255.0
        img = img.astype(np.uint8)

        cv2.imshow('image', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()




def sanity_check():
    config = TrainConfig()
    train_loader, _ = _set_data(config)

    for img, mask in train_loader:
        visualize_data([img, mask])


def main(checkpoint_path: Optional[str]=None):
    # Set device
    if checkpoint_path is not None:
        # find the path to the model
        # and the path to the config
        model_path = os.path.join(checkpoint_path, [f for f in os.listdir(checkpoint_path) if os.path.splitext(f)[-1] == '.pth'][0])
        config_path = os.path.join(checkpoint_path, [f for f in os.listdir(checkpoint_path) if os.path.splitext(f)[-1] == '.json'][0])

        config = TrainConfig.load(config_path)
        model = AdaptiveUNet(
            input_shape=config.input_shape,
            output_shape=config.output_shape,
            bottleneck_shape=config.bottleneck_shape,
            bottleneck_out_channels=config.bottleneck_out_channels
        )

    else:
        config = TrainConfig()
        # Initialize model
        model = AdaptiveUNet(
            input_shape=config.input_shape,
            output_shape=config.output_shape,
            bottleneck_shape=config.bottleneck_shape,
            bottleneck_out_channels=config.bottleneck_out_channels
        )

    
    # Build the model architecture
    model.build_contracting_path(max_conv_layers_per_block=5, min_conv_layers_per_block=1)
    model.build_bottleneck(kernel_sizes=3, num_blocks=3, conv_layers_per_block=3)
    model.build_expanding_path(max_conv_layers_per_block=5, min_conv_layers_per_block=1)
    model.build()


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")   
    train_loader, val_loader = _set_data(config)    

    if checkpoint_path is not None:
        model.load_state_dict(torch.load(model_path))

    # Move model to device
    model = model.to(device)
    
    # Define loss function and optimizer
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    logs_dir = dirf.process_path(os.path.join(SCRIPT_DIR, 'runs'), dir_ok=True, file_ok=False)

    # iterate through each "run_*" directory and remove any folder that does not contain a '.json' file
    for r in os.listdir(logs_dir):
        run_dir = False
        if os.path.isdir(os.path.join(logs_dir, r)):
            for file in os.listdir(os.path.join(logs_dir, r)):
                if os.path.splitext(file)[-1] == '.json':
                    run_dir = True
                    break
            
            if not run_dir:
                shutil.rmtree(os.path.join(logs_dir, r))

    exp_log_dir = dirf.process_path(os.path.join(logs_dir, f'run_{len(os.listdir(logs_dir)) + 1}'), dir_ok=True, file_ok=False)

    # Initialize TensorBoard writer
    writer = SummaryWriter(os.path.join(exp_log_dir, 'logs'))
    
    # Train the model
    trained_model = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=config.num_epochs,
        device=device, 
        writer=writer,
        log_dir=exp_log_dir
    )
    
    # Save the final model
    model_path = os.path.join(exp_log_dir, 'final_model.pth')
    torch.save(trained_model.state_dict(), model_path)
    
    # Save the config
    config_path = os.path.join(exp_log_dir, 'config.json')
    config.save(config_path)
    
    # Clean up temporary directories    
    print("Training completed!")

if __name__ == '__main__':
    # try to make it work with a checkpoint path
    checkpoint_path = os.path.join(SCRIPT_DIR, 'runs', 'run_1')
    # main(checkpoint_path)
    main()
        

