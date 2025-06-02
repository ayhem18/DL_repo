import os
import torch
import shutil

import albumentations as A


from pathlib import Path
from typing import Optional, Tuple, Union
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, Dataset


from mypt.shortcuts import P
from mypt.code_utils import directories_and_files as dirf
from mypt.nets.conv_nets.adaptive_unet.adaptive_unet import AdaptiveUNet
from mypt.sanity_checks.loss_functions.bce import dataset_sanity_check
from mypt.data.datasets.segmentation.semantic_seg import SemanticSegmentationDS
from mypt.data.dataloaders.standard_dataloaders import initialize_train_dataloader, initialize_val_dataloader


from train import train_model
from config import TrainConfig

SCRIPT_DIR = Path(__file__).parent
current_dir = SCRIPT_DIR
while 'data' not in os.listdir(current_dir):
    current_dir = current_dir.parent

DATA_DIR = os.path.join(current_dir, 'data') 




def _set_data(config: TrainConfig, return_datasets: bool = False) -> Union[Tuple[DataLoader, DataLoader], 
                                                                           Tuple[DataLoader, DataLoader, Dataset, Dataset]]:
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

    if return_datasets:
        return train_loader, val_loader, train_dataset, val_dataset

    return train_loader, val_loader




def prepare_log_directory() -> P:
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
    return exp_log_dir



def set_model(checkpoint_path: Optional[str]=None) -> Tuple[AdaptiveUNet, TrainConfig]:
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

    if checkpoint_path is not None:
        model.load_state_dict(torch.load(model_path))

    return model, config



def main(checkpoint_path: Optional[str]=None, sanity_check: bool = False):

    model, config = set_model(checkpoint_path=checkpoint_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")   
    
    if sanity_check:
        train_loader, val_loader, train_dataset, val_dataset = _set_data(config, return_datasets=True)
        dataset_sanity_check(train_dataset, lambda x: x[1])
        dataset_sanity_check(val_dataset, lambda x: x[1])

    else:
        train_loader, val_loader = _set_data(config)    

    # Define loss function and optimizer
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)


    # Initialize TensorBoard writer
    exp_log_dir = prepare_log_directory()
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
        

