import os
import torch
import shutil

import albumentations as A


from pathlib import Path
from typing import Tuple, Union
from diffusers import DDPMScheduler, DDPMPipeline

from torch.utils.data import DataLoader, Dataset


from mypt.shortcuts import P
from mypt.loggers import get_logger
from mypt.code_utils import directories_and_files as dirf
from mypt.data.dataloaders.standard_dataloaders import initialize_train_dataloader, initialize_val_dataloader


from dataset.mnist import MnistDSWrapper
from uncon_diffusion_train import train_diffusion_model
from config import ModelConfig, OptimizerConfig, TrainingConfig




SCRIPT_DIR = Path(__file__).parent
current_dir = SCRIPT_DIR
while 'data' not in os.listdir(current_dir):
    current_dir = current_dir.parent

DATA_DIR = os.path.join(current_dir, 'data') 


def set_data(model_config: ModelConfig, 
             train_config: TrainingConfig, 
             return_datasets: bool = False) -> Union[Tuple[DataLoader, DataLoader],Tuple[DataLoader, DataLoader, Dataset, Dataset]]:

    train_data_path = os.path.join(SCRIPT_DIR, 'data', 'train')
    val_data_path = os.path.join(SCRIPT_DIR, 'data', 'val')

    train_transforms = [
        A.RandomResizedCrop(size=model_config.input_shape[1:], scale=(0.4, 1)),
        A.ToTensorV2()
    ]
    
    val_transforms = [
        A.RandomResizedCrop(size=model_config.input_shape[1:], scale=(0.4, 1)),
        A.ToTensorV2()
    ]

    # create the datasets
    train_ds = MnistDSWrapper(root=train_data_path, 
                              train=True, 
                              transforms=train_transforms, 
                              output_shape=(48, 48),
                              unconditional=True
                              )
    
    val_ds = MnistDSWrapper(root=val_data_path, 
                            train=False, 
                            transforms=val_transforms, 
                            output_shape=(48, 48),
                            unconditional=True
                            )


    # Create data loaders
    train_loader = initialize_train_dataloader(train_ds, seed=42, batch_size=train_config.train_batch_size, num_workers=2, drop_last=True)
    val_loader = initialize_val_dataloader(val_ds, seed=42, batch_size=train_config.val_batch_size, num_workers=2)

    if return_datasets:
        return train_loader, val_loader, train_ds, val_ds

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


from diffusers import UNet2DModel

# one that uses the diffusers implementation of the diffusion unet
def set_model(config: ModelConfig) -> UNet2DModel:
    
    model = UNet2DModel(
        sample_size=config.input_shape[1:],
        in_channels=config.input_shape[0],
        out_channels=config.input_shape[0],
        block_out_channels=(128, 128),
        down_block_types=("DownBlock2D", "DownBlock2D"), # let's see how this turns out without attention blocks
        up_block_types=("UpBlock2D", "UpBlock2D"), # let's see how this turns out without attention blocks
        time_embedding_dim=128,
    )

    # # let's use the default model for now.
    # model = UNet2DModel(
    #     sample_size=config.input_shape[1:],
    #     in_channels=config.input_shape[0],
    #     out_channels=config.input_shape[0],

        # layers_per_block=3,
        # block_out_channels=(32, 64, 128),
        # down_block_types=("DownBlock2D", "DownBlock2D", "DownBlock2D"),
        # up_block_types=("UpBlock2D", "UpBlock2D", "UpBlock2D"),
        # block_out_channels=(32, 64, 128),
    # )



    return model


# a function using my own implementation of the diffusion unet

# def set_model(config: ModelConfig) -> DiffusionUNet:

#     model = DiffusionUNet(input_channels=config.input_shape[0],
#                           output_channels=config.input_shape[0],
#                           cond_dimension=256,                          
#                           )
    
#     # the default model reaches a training loss of 0.01 in less than 10 epochs
#     # let's try to use a model with a similar capacity: it uses attention blocks which aren't currently implemented 
#     # however, let's match the number of blocks + the number of channels. 

#     model.build_down_block(num_down_layers=4, num_res_blocks=3, out_channels=[256, 512, 1024, 1024], downsample_types="conv")
#     model.build_middle_block(num_res_blocks=3)
#     model.build_up_block(num_res_blocks=3, upsample_types="transpose_conv")

#     return model


from diffusers.optimization import get_cosine_schedule_with_warmup



def main():
    from mypt.code_utils import pytorch_utils as pu
    pu.seed_everything(42)
    
    model_config = ModelConfig()
    opt_config = OptimizerConfig()
    train_config = TrainingConfig() 

    model = set_model(model_config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")   

    train_loader, val_loader = set_data(model_config, train_config)

    # Define loss function and optimizer
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=opt_config.learning_rate)


    # Initialize TensorBoard writer
    exp_log_dir = prepare_log_directory()
    logger = get_logger('tensorboard', log_dir=os.path.join(exp_log_dir, 'logs'))

    # the parameters of the noise scheduler were set by playing around with the noise scheduler and visualizing the results.
    # check the sanity_checks.py file for more details.
    noise_scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule="linear", beta_end=0.0015) 


    # why ? no good reason, just copied from 
    # https://colab.research.google.com/github/huggingface/notebooks/blob/main/diffusers/training_example.ipynb#scrollTo=my90vVcmxU5V
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=len(train_loader) * 2, # 2 epochs to warm up
        num_training_steps=(len(train_loader) * train_config.num_epochs),
    )


    # Train the model
    trained_model = train_diffusion_model(
        model=model,
        noise_scheduler=noise_scheduler,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        num_epochs=train_config.num_epochs,
        device=device, 
        logger=logger,
        log_dir=exp_log_dir,
        debug=False,
        val_per_epoch=10 
    )
    
    # save the model in a way that can be loaded by the diffusers library
    if isinstance (model, UNet2DModel):
        pipeline = DDPMPipeline(unet=trained_model, scheduler=noise_scheduler)
        pipeline.save_pretrained(os.path.join(exp_log_dir, 'model')) 
        return 

    # save custom model 
    # Save the config
    model_config.save(os.path.join(exp_log_dir, 'model_config.json'))
    opt_config.save(os.path.join(exp_log_dir, 'opt_config.json'))
    print("Training completed!")


from mypt.visualization.general import visualize


def inference(folder_path: P, num_samples: int = 20, num_inference_steps: int = 250):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipeline = DDPMPipeline.from_pretrained(folder_path)
    pipeline.to(device)
    images = pipeline(batch_size=num_samples, num_inference_steps=num_inference_steps).images
    
    for i, im in enumerate(images):
        visualize(im, window_name=f"sampled_image_{i}")
    
    return images


if __name__ == '__main__':
    # try to make it work with a checkpoint path
    # checkpoint_path = os.path.join(SCRIPT_DIR, 'runs', 'run_1')
    # # main(checkpoint_path)
    # main()
    checkpoint_path = os.path.join(SCRIPT_DIR, 'runs', 'run_2', 'model')
    inference(checkpoint_path)
