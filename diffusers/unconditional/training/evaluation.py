import os, torch, shutil

from PIL import Image
from typing import Union
from torch.utils.data import DataLoader
from torch_fidelity import calculate_metrics
from diffusers.models.unets.unet_2d import UNet2DModel
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from mypt.shortcuts import P
from mypt.code_utils import directories_and_files as dirf
from mypt.code_utils.image_processing.to_numpy import to_displayable_np
from mypt.nets.conv_nets.diffusion_unet.wrapper.diffusion_unet1d import DiffusionUNetOneDim

from training.sample import sample_from_diffusion_model

def evaluate_diffusion_model(
    model: Union[UNet2DModel, DiffusionUNetOneDim],
    noise_scheduler: DDPMScheduler,
    val_loader: DataLoader,
    sampled_dir: P,
    real_dir: P,
    num_inference_steps: int,
    device: torch.device,
    remove_sampled_dir: bool = True,
) -> float: 

    sampled_dir = dirf.process_path(sampled_dir, dir_ok=True, file_ok=False, must_exist=False)
    real_dir = dirf.process_path(real_dir, dir_ok=True, file_ok=False, must_exist=False)

    # scale if necessary
    real_images = next(iter(val_loader)) 

    # move to cpu
    real_images = real_images.cpu()

    for i, r in enumerate(real_images):
        if r.max() <= 1:
            r = (r * 255).to(torch.uint8) 
    
        # convert to a proper numpy
        r = to_displayable_np(r)

        # save the image
        Image.fromarray(r).save(os.path.join(real_dir, f"sample_{i}.png"))

    # extract the shape and the number of samples from the real images
    shape = real_images.shape
    num_samples = real_images.shape[0]

    # sample from the model
    samples, scale_images = sample_from_diffusion_model(model, noise_scheduler, device, shape, num_samples, num_inference_steps)

    # move to cpu
    samples = samples.cpu()

    for i, s in enumerate(samples):
        if scale_images:
            s = (s + 1) * 127.5
        else:
            s = s 
        
        # convert to a proper numpy
        s = to_displayable_np(s)

        # save the image
        Image.fromarray(s).save(os.path.join(sampled_dir, f"sample_{i}.png"))



    # at this point calculate the scores
    metrics_dict = calculate_metrics(
        input1=sampled_dir,
        input2=real_dir,
        cuda=True,
        isc=False,
        fid=True,
        kid=True,
        verbose=True,
    )

    
    if remove_sampled_dir:
        shutil.rmtree(sampled_dir)

    # remove the real directory anyway
    shutil.rmtree(real_dir)

    return metrics_dict




