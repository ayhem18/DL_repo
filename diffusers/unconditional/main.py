import os
import torch

from diffusers.models.unets.unet_2d import UNet2DModel
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.pipelines.ddpm.pipeline_ddpm import DDPMPipeline
from diffusers.optimization import get_cosine_schedule_with_warmup

from mypt.shortcuts import P
from mypt.loggers import get_logger


from uncon_diffusion_train import train_diffusion_model
from config import ModelConfig, OptimizerConfig, TrainingConfig


from model import set_model
from data import set_data, prepare_log_directory


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
        device=device, 
        logger=logger,
        log_dir=exp_log_dir,
        debug=False,
        val_per_epoch=5 
    )
    
    # save the model in a way that can be loaded by the diffusers library
    if isinstance (trained_model, UNet2DModel):
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
    images = pipeline(batch_size=num_samples, num_inference_steps=num_inference_steps, return_dict=False).images
    
    for i, im in enumerate(images):
        visualize(im, window_name=f"sampled_image_{i}")
    
    return images


if __name__ == '__main__':
    # try to make it work with a checkpoint path
    # checkpoint_path = os.path.join(SCRIPT_DIR, 'runs', 'run_1')
    # # main(checkpoint_path)
    main()
    # checkpoint_path = os.path.join(SCRIPT_DIR, 'runs', 'run_2', 'model')
    # inference(checkpoint_path)
