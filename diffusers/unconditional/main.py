import os
import torch

from diffusers.models.unets.unet_2d import UNet2DModel
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.pipelines.ddpm.pipeline_ddpm import DDPMPipeline
from diffusers.optimization import get_cosine_schedule_with_warmup

from mypt.shortcuts import P
from mypt.loggers import get_logger
from mypt.visualization.general import visualize_grid
from mypt.code_utils import directories_and_files as dirf

from training.model import set_model
from training.time_steps import set_timesteps_sampler
from training.data import set_data, prepare_log_directory
from training.config import ModelConfig, OptimizerConfig, TrainingConfig
from training.unconditional_train_diffusion import train_diffusion_model


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

    # TODO: I need to better understand the strategies of setting the noise scheduler parameters.
    # noise_scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule="linear", beta_end=0.0015) 

    noise_scheduler = DDPMScheduler(num_train_timesteps=1000)

    # timesteps_sampler = set_timesteps_sampler(
    #     train_config.timesteps_sampler_type, 
    #     noise_scheduler.config.num_train_timesteps,
    #     bins=train_config.timestep_bins,
    #     thresholds=train_config.loss_thresholds
    # )

    # why ? no good reason, just copied from 
    # https://colab.research.google.com/github/huggingface/notebooks/blob/main/diffusers/training_example.ipynb#scrollTo=my90vVcmxU5V
    training_steps = (len(train_loader) * train_config.num_epochs)

    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=int(0.05 * training_steps),
        num_training_steps=training_steps,
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
        timestep_bins=train_config.timestep_bins,
        validation_timesteps=train_config.validation_timesteps,
        timesteps_sampler_type=train_config.timesteps_sampler_type,
        device=device, 
        logger=logger,
        log_dir=exp_log_dir,
        debug=False,
        val_per_epoch=train_config.val_per_epoch,
        
        # time_steps_kwargs are passed to the set_timesteps_sampler function
        time_steps_kwargs={
            "bins": train_config.timestep_bins,
            "thresholds": train_config.loss_thresholds
        }
    )
    
    # save the model in a way that can be loaded by the diffusers library
    if isinstance (trained_model, UNet2DModel):
        pipeline = DDPMPipeline(unet=trained_model, scheduler=noise_scheduler)
        pipeline.save_pretrained(os.path.join(exp_log_dir, 'model')) 

    else:
        # TODO: save my custom model
        pass

    # Save the config
    model_config.save(os.path.join(exp_log_dir, 'model_config.json'))
    opt_config.save(os.path.join(exp_log_dir, 'opt_config.json'))
    train_config.save(os.path.join(exp_log_dir, 'train_config.json'))
    print("Training completed!")




def inference(folder_path: P, num_samples: int = 20, save_image_path: P = None, num_inference_steps: int = 250):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipeline = DDPMPipeline.from_pretrained(folder_path)
    pipeline.to(device)

    images = pipeline(batch_size=num_samples, num_inference_steps=num_inference_steps).images     # type: ignore (since the linter does not understand that the return type is ImagePipelineOutput with images attribute)

    if save_image_path is None:
        visualize_grid(images, title="sampled_images")
    else:
        for i, im in enumerate(images):
            im.save(os.path.join(save_image_path, f"inference_image_{i}.png"))

    return images

if __name__ == '__main__':
    main()
    # script_dir = os.path.dirname(os.path.abspath(__file__))
    # runs_dir = os.path.join(script_dir, 'training', 'runs')
    # run1, ckpnt1 = os.path.join(runs_dir, 'run_1', 'inference'), os.path.join(runs_dir, 'run_1', 'model')
    # run2, ckpnt2 = os.path.join(runs_dir, 'run_2', 'inference'), os.path.join(runs_dir, 'run_2', 'model')
    # run3, ckpnt3 = os.path.join(runs_dir, 'run_3', 'inference'), os.path.join(runs_dir, 'run_3', 'model')

    # dirf.process_path(run1, dir_ok=True, file_ok=False, must_exist=False)
    # dirf.process_path(run2, dir_ok=True, file_ok=False, must_exist=False)
    # dirf.process_path(run3, dir_ok=True, file_ok=False, must_exist=False)


    # inference(ckpnt1, num_samples=20, num_inference_steps=1000, save_image_path=run1)
    # inference(ckpnt2, num_samples=20, num_inference_steps=1000, save_image_path=run2)
    # inference(ckpnt3, num_samples=20, num_inference_steps=1000, save_image_path=run3)



