import torch, os, numpy as np

from tqdm import tqdm
from PIL import Image
from torch.utils.data import DataLoader
from typing import Tuple, Union


from mypt.shortcuts import P
from mypt.loggers import BaseLogger
from mypt.visualization.general import visualize
from mypt.code_utils import directories_and_files as dirf
from mypt.nets.conv_nets.diffusion_unet.wrapper.diffusion_unet1d import DiffusionUNetOneDim

from diffusers.models.unets.unet_2d import UNet2DModel
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.pipelines.ddpm.pipeline_ddpm import DDPMPipeline


def sample_with_diffusers(model: UNet2DModel, 
                          noise_scheduler: DDPMScheduler,
                          device: torch.device,
                          num_samples: int = 10,
                          num_inference_steps: int = 10) -> torch.Tensor:
    
    
    # set the number of inference time steps
    noise_scheduler.set_timesteps(num_inference_steps=num_inference_steps)
    
    pipeline = DDPMPipeline(unet=model, scheduler=noise_scheduler)
    pipeline.to(device)

    images = pipeline(
        batch_size=num_samples,
        generator=torch.Generator(device='cpu').manual_seed(42),
        output_type="tensor", # if output_type is "pil", it will return a list of PIL images, any other value it will return a numpy array
        num_inference_steps=num_inference_steps
    ).images

    # convert to tensor and return
    return torch.from_numpy(images.transpose(0, 3, 1, 2)) * 255.0


def sample_manual(model: DiffusionUNetOneDim, 
                    shape: Tuple[int, int, int, int],
                    noise_scheduler: DDPMScheduler,
                    device: torch.device,
                    num_inference_steps: int = 1000,
                    ) -> torch.Tensor:
    
    model = model.to(device)
    model.eval()

    noise_scheduler.set_timesteps(num_inference_steps=num_inference_steps)

    samples = torch.randn(*shape, device=device)

    for _, t in enumerate(tqdm(noise_scheduler.timesteps, desc="Sampling")):
    
        with torch.no_grad():
            # 1. predict noise residual
            residual = model.forward(samples, torch.ones(samples.shape[0], device=device) * t) # generate images with the same label and the same masks as the validation batch
            
            if isinstance(model, UNet2DModel):
                # for whatever reason, the outputs are a wrapper around a tensor.
                residual = residual.sample

            # 2. compute previous image and set x_t -> x_t-1
            samples = noise_scheduler.step(residual, t, samples).prev_sample

    return samples


def sample_from_diffusion_model(model: Union[DiffusionUNetOneDim, UNet2DModel], 
                          noise_scheduler: DDPMScheduler,
                          device: torch.device,
                          shape: Tuple[int, int, int, int],
                          num_samples: int = 10,
                          num_inference_steps: int = 10) -> Tuple[torch.Tensor, bool]:
    
    if isinstance(model, UNet2DModel):
        return sample_with_diffusers(model, noise_scheduler, device, num_samples, num_inference_steps), False
    else:
        return sample_manual(model, shape, noise_scheduler, device, num_inference_steps), True


def sample_diffusion_epoch(model: Union[DiffusionUNetOneDim, UNet2DModel],  
                        noise_scheduler: DDPMScheduler,
                        val_loader: DataLoader, 
                        device: torch.device,
                        log_dir: P,
                        epoch: int,
                        debug: bool = False):
    """
    Run one epoch of validation.
    
    Args:
        model: The model to validate
        val_loader: DataLoader for validation data
        criterion: Loss function
        device: Device to validate on
    
    Returns:
        Average validation loss for the epoch
    """

    images = next(iter(val_loader))

    with torch.no_grad():
        diffusion_samples, scale_images = sample_from_diffusion_model(model, 
                                                                    noise_scheduler, 
                                                                    device, 
                                                                    images.shape, 
                                                                    num_samples=20, 
                                                                    num_inference_steps=250)

        epoch_dir = os.path.join(log_dir, 'samples', f"epoch_{epoch+1}")
        dirf.process_path(epoch_dir, dir_ok=True, file_ok=False, must_exist=False)

        for i, s in enumerate(diffusion_samples):
            # rescale to [0, 255] for saving and visualization
            if scale_images:
                vis_s = (s + 1) * 127.5
            else:
                vis_s = s

            if debug:
                visualize(vis_s, window_name=f"sampled_image_{i}")

            # convert to numpy for saving
            if vis_s.shape[0] >= 3: # (C, H, W) with C > 3
                save_s = vis_s.cpu().permute(1, 2, 0)
            else: # (C, H, W) with C <= 3
                save_s = vis_s.cpu().squeeze(0)
                
            save_s = save_s.numpy().astype(np.uint8)
            Image.fromarray(save_s).save(os.path.join(epoch_dir, f"sample_{i}.png"))
