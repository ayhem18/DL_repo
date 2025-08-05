import os, torch, shutil

from PIL import Image
from typing import Union, Dict
from torch.utils.data import DataLoader
from torch_fidelity import calculate_metrics
from diffusers.models.unets.unet_2d import UNet2DModel
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from mypt.shortcuts import P
from mypt.code_utils import directories_and_files as dirf
from mypt.data.datasets.genericFolderDs import GenericFolderDS
from mypt.code_utils.image_processing.to_numpy import to_displayable_np
from mypt.nets.conv_nets.diffusion_unet.wrapper.diffusion_unet1d import DiffusionUNetOneDim

from training.data import prepare_tensor_for_metrics    
from training.sample import sample_from_diffusion_model



def evaluate_diffusion_model(
    model: Union[UNet2DModel, DiffusionUNetOneDim],
    noise_scheduler: DDPMScheduler,
    val_loader: DataLoader,
    sampled_dir: P,
    num_inference_steps: int,
    device: torch.device,
    remove_sampled_dir: bool = True,
) -> Dict[str, float]: 

    sampled_dir = dirf.process_path(sampled_dir, dir_ok=True, file_ok=False, must_exist=False)
    
    # Get properties from the validation set
    num_total_samples = len(val_loader.dataset)
    image_shape = val_loader.dataset[0].shape

    samples_generated = 0
    with torch.no_grad():
        for i in range(0, num_total_samples, val_loader.batch_size):
            batch_size = min(val_loader.batch_size, num_total_samples - i)
            
            shape_for_batch = (batch_size, *image_shape[1:])
            
            samples, scale_images = sample_from_diffusion_model(
                model, noise_scheduler, device, shape_for_batch, batch_size, num_inference_steps
            )

            samples = samples.cpu()

            for j, s in enumerate(samples):
                if scale_images:
                    s = (s + 1) * 127.5
                
                s = to_displayable_np(s)
                Image.fromarray(s).save(os.path.join(sampled_dir, f"sample_{samples_generated + j}.png"))
            
            samples_generated += batch_size
            
    sampled_ds = GenericFolderDS(sampled_dir, [], item_transforms=prepare_tensor_for_metrics)

    metrics_dict = calculate_metrics(
        input1=sampled_ds,
        input2=val_loader.dataset,
        cuda=True,
        isc=True,
        fid=True,
        kid=True,
        verbose=False,
        kid_subset_size=len(sampled_ds)
    )

    # round the results
    metrics_dict = {k: round(v, 6) for k, v in metrics_dict.items()}
    
    # add the number of inference steps to the metrics dict
    metrics_dict["num_inference_steps"] = num_inference_steps

    if remove_sampled_dir:
        shutil.rmtree(sampled_dir)

    return metrics_dict




