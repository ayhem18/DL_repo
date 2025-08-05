import os, json, torch

from torch.utils.data import DataLoader
from diffusers.models.unets.unet_2d import UNet2DModel
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.pipelines.ddpm.pipeline_ddpm import DDPMPipeline
from diffusers.optimization import get_cosine_schedule_with_warmup

from mypt.shortcuts import P
from mypt.loggers import get_logger
from mypt.code_utils import pytorch_utils as pu
from mypt.visualization.general import visualize_grid

from training.model import set_model
from training.evaluation import evaluate_diffusion_model
from training.config import ModelConfig, OptimizerConfig, TrainingConfig
from training.unconditional_train_diffusion import train_diffusion_model
from training.data import set_data, prepare_log_directory, prepare_tensor_for_metrics
    

def main(checkpoint_tolerant: bool = False):
    
    model_config = ModelConfig()
    opt_config = OptimizerConfig()
    train_config = TrainingConfig() 

    pu.seed_everything(train_config.seed)

    model = set_model(model_config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")   

    train_loader, val_loader = set_data(model_config, train_config)

    # Define loss function and optimizer
    criterion = torch.nn.MSELoss()  
    optimizer = torch.optim.AdamW(model.parameters(), lr=opt_config.learning_rate)


    # Initialize TensorBoard writer
    exp_log_dir = prepare_log_directory(checkpoint_tolerant)
    logger = get_logger('tensorboard', log_dir=os.path.join(exp_log_dir, 'logs'))

    # TODO: I need to better understand the strategies of setting the noise scheduler parameters.
    # noise_scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule="linear", beta_end=0.0015) 

    noise_scheduler = DDPMScheduler(num_train_timesteps=1000)

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



def evaluate(model_path: P, loader: DataLoader, metrics_dir: P, num_inference_steps: int = 1000):
    device = pu.get_default_device()

    # load the model and the noise scheduler
    model = UNet2DModel.from_pretrained(os.path.join(model_path, "unet"))
    noise_scheduler = DDPMScheduler.from_pretrained(os.path.join(model_path, "scheduler"))

    sampled_dir = os.path.join(metrics_dir, "sampled")

    metrics_dict = evaluate_diffusion_model(
        model=model,
        noise_scheduler=noise_scheduler,
        device=device,
        val_loader=loader,
        num_inference_steps=num_inference_steps,
        sampled_dir=sampled_dir,
        remove_sampled_dir=False
    )

    # round the results
    # save the metrics dict to a json file
    with open(os.path.join(metrics_dir, "metrics.json"), "w") as f:
        json.dump(metrics_dict, f, indent=4)

    return metrics_dict



if __name__ == '__main__':
    main(checkpoint_tolerant=True)

    # script_dir = os.path.dirname(os.path.abspath(__file__))
    # runs_dir = os.path.join(script_dir, 'training', 'runs')

    # ckpnt7 = os.path.join(runs_dir, 'run_7', 'model')
    # metrics_dir7 = os.path.join(runs_dir, 'run_7', 'metrics')

    # model_config = ModelConfig()
    # train_config = TrainingConfig()

    # model_config.dataset = "butterflies" 
    # train_config.val_batch_size = 400

    # train_loader, _ = set_data(model_config, train_config, item_transforms=prepare_tensor_for_metrics)

    # # evaluate(ckpnt7, train_loader, metrics_dir7, num_inference_steps=1000)

    # ckpnt8 = os.path.join(runs_dir, 'run_8', 'model')
    # metrics_dir8 = os.path.join(runs_dir, 'run_8', 'metrics')
    # evaluate(ckpnt8, train_loader, metrics_dir8, num_inference_steps=1000)


    # d = os.path.join(runs_dir, 'run_7', 'metrics', 'sampled')

    # ds = GenericFolderDS(d, image_transforms=[], item_transforms=prepare_tensor_for_metrics)

    # imgs = [ds[i] for i in range(10)]

    # visualize_grid(imgs, title="sampled_images")



    # # run1, ckpnt1 = os.path.join(runs_dir, 'run_1', 'inference'), os.path.join(runs_dir, 'run_1', 'model')
    # # run2, ckpnt2 = os.path.join(runs_dir, 'run_2', 'inference'), os.path.join(runs_dir, 'run_2', 'model')
    # # run3, ckpnt3 = os.path.join(runs_dir, 'run_3', 'inference'), os.path.join(runs_dir, 'run_3', 'model')

    # run6, ckpnt6 = os.path.join(runs_dir, 'run_6', 'inference'), os.path.join(runs_dir, 'run_6', 'model')


    # # dirf.process_path(run1, dir_ok=True, file_ok=False, must_exist=False)
    # # dirf.process_path(run2, dir_ok=True, file_ok=False, must_exist=False)
    # dirf.process_path(run6, dir_ok=True, file_ok=False, must_exist=False)


    # inference(ckpnt6, num_samples=20, num_inference_steps=1000, save_image_path=run6)
    # # inference(ckpnt2, num_samples=20, num_inference_steps=1000, save_image_path=run2)
    # # inference(ckpnt3, num_samples=20, num_inference_steps=1000, save_image_path=run3)



