"""
This script contains the implementation of the tuning process
"""
import os, wandb, json

from pathlib import Path
from typing import Union, Dict, Optional, Tuple
from functools import partial

from mypt.code_utilities import directories_and_files as dirf
from mypt.models.simClr.simClrModel import ResnetSimClr

from .training import run_pipeline
from .evaluating import evaluate_model

WANDB_PROJECT_NAME="SimClr"

_BATCH_SIZE = 512 

def _sweep_function( 
        train_data_folder:Union[str, Path],
        val_data_folder:Optional[Union[str, Path]],
        output_shape:Tuple[int, int],
        epochs_per_sweep: int, 
        batch_size: int,
        
        temperature: float,
        log_dir: Union[str, Path],

        val_per_epoch: int = 3,
        seed:int = 69,
        num_train_samples_per_cls: Optional[int]=None,
        num_val_samples_per_cls: Optional[int]=None,
        evaluate: bool=False,
        num_neighbors:Optional[int]=None
        ):
    
    if evaluate and num_neighbors is None:
        raise ValueError(f"the argument 'evaluate' is set to True while the argument 'num_neighbor' is not explicitly passed !!")

    wandb.init(project=WANDB_PROJECT_NAME)
        #config=sweep_config)

    # extract the number of fully connected layers form the sweep configuration
    num_fc_layers = wandb.config.num_fc_layers
    lr = 10 ** wandb.config.lr    


    model = ResnetSimClr(input_shape=(3,) + output_shape, output_dim=128, num_fc_layers=num_fc_layers, freeze=False)

    # create a directory where to save the checkpoint
    ckpnt_dir = os.path.join(log_dir,  f'sweep_{len(os.listdir(log_dir)) + 1}')

    run_result = run_pipeline(
            model=model, 
            train_data_folder=train_data_folder, 
            val_data_folder=val_data_folder,   
            output_shape=output_shape,
            num_epochs=epochs_per_sweep,
            batch_size=batch_size, 
            learning_rates=lr,
            temperature=temperature,
            ckpnt_dir=ckpnt_dir,
            val_per_epoch=val_per_epoch,
            seed=seed,
            num_train_samples_per_cls=num_train_samples_per_cls,
            num_val_samples_per_cls=num_val_samples_per_cls)

    train_loss, val_loss, ckpnt = None, None, None

    if len(run_result) == 2:
        train_loss, ckpnt = run_result
    else:
        train_loss, val_loss, ckpnt = run_result

    model_config = {"num_fc_layers": num_fc_layers, 
                    "lr": lr,
                    "train_loss": train_loss}

    if val_loss is not None:
        model_config['val_loss'] = val_loss


    if evaluate:
        accuracy = evaluate_model(model=model, 
                                  model_ckpnt=ckpnt, 
                                  train_data_folder=train_data_folder,
                                  val_data_folder=val_data_folder,
                                  output_shape=output_shape,
                                  num_train_samples_per_cls=num_train_samples_per_cls,
                                  num_val_samples_per_cls=num_val_samples_per_cls,
                                  num_neighbors=num_neighbors)

        model_config['val_accuracy'] = accuracy


    # save the configuration along with the checkpoint
    with open(os.path.join(ckpnt_dir, 'config.json'), 'w') as f:
        json.dump(model_config, f, indent=4)
    


def tune(train_data_folder: Union[str, Path],
         val_data_folder: Optional[Union[str, Path]],

         log_dir: Union[str, Path],
         temperature: float,

         output_shape:Tuple[int, int],

         lr_options: Dict,
         num_layers_options: Dict,

         objective: str,
         tune_method:str,

         epochs_per_sweeps:int,
         sweep_count:int,
         **kwargs
        ):

    if objective not in ['train_loss', 'val_loss']:
        raise NotImplementedError(f"objective not currently supported. Found: {objective}, supporting only: {['train_loss', 'val_loss']}")
    

    #initialize the sweep
    sweep_name = f'{os.path.basename(Path(log_dir).parent)}_{os.path.basename(log_dir)}'

    # prepare the wandb configuration
    sweep_config = {
        "name": sweep_name, 
        "metric": {"name": objective, "goal": "minimize"}, 
        "method": tune_method, 
        "parameters": {"num_fc_layers": num_layers_options, "lr": lr_options},
    }

    # remove any empty directories
    dirf.clear_directory(directory=log_dir, condition=lambda p: os.path.isdir(p) and len(os.listdir(p)) == 0)

    log_dir = dirf.process_path(log_dir, 
                                dir_ok=True, 
                                file_ok=False, 
                                condition=lambda p: len(os.listdir(p)) == 0, 
                                error_message="The log directory must be empty")


    # let's define the object that the sweep agent will run
    sweep_function_object = partial(_sweep_function,
                                    train_data_folder=train_data_folder,
                                    val_data_folder=val_data_folder,
                                    # sweep_config=sweep_config,
                                    output_shape=output_shape, 
                                    epochs_per_sweep=epochs_per_sweeps, 
                                    batch_size=_BATCH_SIZE,
                                    temperature=temperature,
                                    log_dir=log_dir,
                                    val_per_epoch=3,
                                    seed=0,
                                    **kwargs
                                    )

    # define the sweep id
    sweep_id = wandb.sweep(sweep=sweep_config, 
                    project=WANDB_PROJECT_NAME)

    # run the sweep
    wandb.agent(sweep_id, 
            function=sweep_function_object,
            count=sweep_count)
    
