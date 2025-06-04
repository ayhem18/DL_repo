import os
from mypt.data.datasets.genericFolderDs import MnistGenericWrapper
import torch
import shutil

import albumentations as A


from pathlib import Path
from diffusers import DDPMScheduler

from typing import Optional, Tuple, Union
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, Dataset


from mypt.shortcuts import P
from mypt.code_utils import directories_and_files as dirf
from mypt.nets.conv_nets.diffusion_unet.wrapper.diffusion_unet import DiffusionUNet
from mypt.data.dataloaders.standard_dataloaders import initialize_train_dataloader, initialize_val_dataloader


from DL_repo.diffusers.dataset.mnist import MnistDSWrapper
from config import ModelConfig, OptimizerConfig, TrainingConfig
from DL_repo.diffusers.unconditional.uncon_diffusion_train import train_diffusion_model



SCRIPT_DIR = Path(__file__).parent
current_dir = SCRIPT_DIR
while 'data' not in os.listdir(current_dir):
    current_dir = current_dir.parent

DATA_DIR = os.path.join(current_dir, 'data') 


def set_data(model_config: ModelConfig, train_config: TrainingConfig, return_datasets: bool = False) -> Union[Tuple[DataLoader, DataLoader], 
                                                                           Tuple[DataLoader, DataLoader, Dataset, Dataset]]:
    train_data_path = os.path.join(SCRIPT_DIR, 'data', 'train')
    val_data_path = os.path.join(SCRIPT_DIR, 'data', 'val')


    train_transforms = A.Compose([
        # A.Resize(height=config.input_shape[1], width=config.input_shape[2]),
        # A.HorizontalFlip(p=0.5),
        # A.VerticalFlip(p=0.5),
        # A.RandomBrightnessContrast(p=0.2),
        A.ToTensorV2(),
    ])
    
    val_transforms = A.Compose([
        # A.Resize(height=config.output_shape[1], width=config.output_shape[2]),
        A.ToTensorV2(),
    ])

    # Create datasets
    train_dataset = MnistGenericWrapper(
        root=train_data_path,
        train=True,
        download=not os.path.exists(train_data_path),
        transforms=train_transforms,
        output_shape=model_config.input_shape[1:]
    )
    
    val_dataset = MnistGenericWrapper(
        root=val_data_path,
        train=False,
        download=not os.path.exists(val_data_path),
        transforms=val_transforms,
        output_shape=model_config.input_shape[1:]
    )
    
    # Create data loaders
    train_loader = initialize_train_dataloader(train_dataset, seed=42, batch_size=train_config.train_batch_size, num_workers=2, drop_last=True)
    val_loader = initialize_val_dataloader(val_dataset, seed=42, batch_size=train_config.val_batch_size, num_workers=2)

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



def set_model(config: ModelConfig) -> DiffusionUNet:
    model = DiffusionUNet(input_channels=config.input_shape[0],
                          output_channels=config.input_shape[0],
                          cond_dimension=128,                          
                          )
    
    model.build_down_block(num_down_layers=2, num_res_blocks=2, out_channels=[16, 32], downsample_types="conv")
    model.build_middle_block(num_res_blocks=2, inner_dim=256)
    model.build_up_block(num_res_blocks=2, upsample_types="transpose_conv", inner_dim=256)

    return model


def main(checkpoint_path: Optional[str]=None, sanity_check: bool = False):

    model_config = ModelConfig()
    opt_config = OptimizerConfig()
    train_config = TrainingConfig() 

    model = set_model(model_config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")   

    train_loader, val_loader = set_data(model_config, train_config)

    # Define loss function and optimizer
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=opt_config.learning_rate)


    # Initialize TensorBoard writer
    exp_log_dir = prepare_log_directory()
    writer = SummaryWriter(os.path.join(exp_log_dir, 'logs'))
    
    noise_scheduler = DDPMScheduler(num_train_timesteps=250)

    # Train the model
    trained_model = train_diffusion_model(
        model=model,
        noise_scheduler=noise_scheduler,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=train_config.num_epochs,
        device=device, 
        writer=writer,
        log_dir=exp_log_dir
    )
    
    # Save the final model
    model_path = os.path.join(exp_log_dir, 'final_model.pth')
    torch.save(trained_model.state_dict(), model_path)
    
    # # Save the config
    # config_path = os.path.join(exp_log_dir, 'config.json')
    # train_config.save(config_path)
    
    # Clean up temporary directories    
    print("Training completed!")

if __name__ == '__main__':
    # try to make it work with a checkpoint path
    checkpoint_path = os.path.join(SCRIPT_DIR, 'runs', 'run_1')
    # main(checkpoint_path)
    main()
        

