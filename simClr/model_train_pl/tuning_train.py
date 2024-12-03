"""
This script contains the code for tuning the Pytorch Lightning module
"""
import os, pytorch_lightning as L


from typing import Union, Optional, Tuple, List
from pathlib import Path

from clearml import Task, Logger
from clearml.automation.optimization import HyperParameterOptimizer

from clearml.automation import DiscreteParameterRange, LogUniformParameterRange
from clearml.automation.optuna import OptimizerOptuna

from mypt.code_utilities import pytorch_utilities as pu, directories_and_files as dirf

from .set_ds import _set_data_tune
from .train import  _set_logging, _set_ckpnt_callback
from .simClrWrapper import ResnetSimClrWrapper
from .constants import (_TEMPERATURE, _OUTPUT_DIM, _OUTPUT_SHAPE, 
                        ARCHITECTURE_IMAGENETTE, ARCHITECTURE_FOOD_101, TUNING_TASK_RUN_NAME) 


def tune_template_task_function(
        train_data_folder:Union[str, Path],          
        val_data_folder:Optional[Union[str, Path]],
        dataset: str,

        num_epochs: int, 
        batch_size: int,

        parent_log_dir: Union[str, Path], # parent_log_dir where all the logs from the different sub-folders will be saved
        tune_exp_number: int,
        output_shape:Tuple[int, int]=None, 
        num_warmup_epochs:int=10,
        val_per_epoch: int = 3,
        log_per_batch:int=5,
        seed:int = 69,
        set_up: bool = False,
        initial_metric_value:float=10
    ):
    """
    This function represents the 'template_task': the piece of code that will be cloned by the ClearML optimizer.
    
    few ideas to keep in mind: 

    * there are two types of parameters: 
        1. passed through the function parameters: train_data_folder, val_data_folder, num_warmup_epochs.. those are not hp optimized by the HpOptimizer
        2. hyperparameters modified by the HpOptimizer

    * calling the function first with 'set_up=True' will create a task and a dictionary called 'args', connect the 'args' to the task
    and hence save all non-hp to remotely so that different clones of the task will all contain this information (accessed through the 'args' dictionary !!!!)
    
    """

    if output_shape is None:
        output_shape = _OUTPUT_SHAPE

    # and since the 'sub_exp_logger' is actually connected to the 'task' object, ClearML should be able to capture the desired metrics
    task, sub_exp_logger = _set_logging(use_logging=True, 
                                        run_name=f'{TUNING_TASK_RUN_NAME}_{tune_exp_number}', 
                                        return_task=True,
                                        init_task=True,                                         
                                        auto_connect_arg_parser=False, 
                                        auto_connect_frameworks=False, 
                                        auto_connect_streams=False
                                        )
    
    
    # connect the task to the 'num_fc_layers' and 'dropout' parameters
    args = {'num_fc_layers': 2, 'dropout': None}
    
    if set_up:
        # add the non-hp parameters to the 'args' so they can be used across different clones
        args.update({
                "train_data_folder":train_data_folder,                 
                "val_data_folder":val_data_folder,
                "dataset": dataset,

                "num_epochs": num_epochs, 
                "batch_size": batch_size,

                "parent_log_dir": parent_log_dir,
                "tune_exp_number": tune_exp_number,
                "output_shape":output_shape, 
                "num_warmup_epochs":num_warmup_epochs,
                "val_per_epoch": val_per_epoch,
                "seed":seed,
                'log_per_batch': log_per_batch,
                'set_up': True
                })

    task.connect(args)
    
    if args['set_up']:
        # log the 'val_epoch_loss' metric so that the hyperparameter optimizer can find it 
        sub_exp_logger.report_scalar(title='val_epoch_loss', series='val_epoch_loss', value=initial_metric_value, iteration=0)
        task.close()
        return 


    # set the data
    train_dl, val_dl = _set_data_tune(train_data_folder=args['train_data_folder'],
                                val_data_folder=args['val_data_folder'], 
                                dataset=args['dataset'],
                                output_shape=args['output_shape'],
                                batch_size=args['batch_size'], 
                                seed=args['seed'])
    
    print(f"train_dl length {len(train_dl)}" )
    print(f"val_dl length {len(val_dl)}" )

    initial_lr = 0.3 * (args['batch_size'] / 256) # according to the paper's formula
    
    parent_log_dir = dirf.process_path(args['parent_log_dir'], 
                                       dir_ok=True, 
                                       file_ok=False)
    
    # clear any empty directories in the 'parent_log_dir'
    dirf.clear_directory(parent_log_dir, lambda x: os.path.isdir(x) and len(os.listdir(x)) == 0)

    # the exact loging directory depends on the number of experiments conducted so far
    sub_exp_log_dir = dirf.process_path(os.path.join(parent_log_dir, f'sub_exp_{len(os.listdir(parent_log_dir)) + 1}'), 
                                        dir_ok=True, 
                                        file_ok=False)

    checkpnt_callback = _set_ckpnt_callback(val_dl=val_dl, 
                                            num_epochs=args['num_epochs'], 
                                            val_per_epoch=args['val_per_epoch'], 
                                            log_dir=sub_exp_log_dir)

    wrapper = ResnetSimClrWrapper(
                                # model parameters
                                input_shape=(3,) + args['output_shape'], 
                                output_dim=_OUTPUT_DIM,
                                num_fc_layers=args['num_fc_layers'],

                                #logging parameters
                                logger=sub_exp_logger,
                                log_per_batch=args['log_per_batch'],
                                
                                # loss parameters 
                                temperature=_TEMPERATURE,
                                debug_loss=True,

                                # learning 
                                lrs=initial_lr,
                                num_epochs=args['num_epochs'],
                                num_warmup_epochs=args['num_warmup_epochs'],

                                # more model parameters
                                dropout=args['dropout'],
                                architecture=ARCHITECTURE_FOOD_101 if args['dataset']=='food101' else ARCHITECTURE_IMAGENETTE,
                                freeze=False
                                )

    # get the default device
    device = pu.get_default_device()

    # define the trainer
    trainer = L.Trainer(
                        accelerator='gpu' if 'cuda' in device else 'cpu',
                        devices=1, 
                        logger=False, 
                        default_root_dir=sub_exp_log_dir,
                        max_epochs=args['num_epochs'],
                        check_val_every_n_epoch=args['val_per_epoch'],
                        log_every_n_steps=1 if len(train_dl) < 10 else 10,
                        callbacks=[checkpnt_callback])


    trainer.fit(model=wrapper,
                train_dataloaders=train_dl,
                val_dataloaders=val_dl,
                )

    # close the task
    task.close()
