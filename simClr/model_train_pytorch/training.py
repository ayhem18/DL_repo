import torch, os, warnings

from clearml import Task, Logger
from time import sleep
from typing import Union, Optional, Tuple, List
from pathlib import Path

from tqdm import tqdm
from torch.optim.sgd import SGD
from torch.optim.lr_scheduler import LinearLR, SequentialLR, CosineAnnealingLR
from torch.utils.data import DataLoader


from mypt.losses.simclr_loss import SimClrLoss
from mypt.code_utilities import pytorch_utilities as pu, directories_and_files as dirf
from mypt.schedulers.annealing_lr import AnnealingLR
from mypt.models.simClr.simClrModel import SimClrModel

from .train_per_epoch import train_per_epoch, validation_per_epoch
from ._set_ds import _set_data

_TRACK_PROJECT_NAME = "SimClr"
PREFERABLE_BATCH_SIZE = 512
OUTPUT_SHAPE = (200, 200)

def _set_optimizer(model: SimClrModel, 
                  lrs: Union[Tuple[float], float], 
                  num_epochs: int,
                  num_warmup_epochs:int,
                  ) -> Tuple[SGD, AnnealingLR]:

    # importing the Lars optimizer inside this function, simply because the "flash" package is noticeably slow to load ....
    # hence slowing down the entire code base even when I am not training
    from flash.core.optimizers import LARS

    if num_warmup_epochs >= num_epochs:
        raise ValueError(f"The number of warmup epochs must be strictly less than the total number of epochs !!. found warmup : {num_warmup_epochs}, and epochs: {num_epochs}")

    if num_warmup_epochs == 0:
        warnings.warn(message="The number of warmup epochs is set to 0 !!")

    if isinstance(lrs, float):
        lr1, lr2 = lrs, 10 * lrs
    elif isinstance(lrs, (Tuple,List)) and len(lrs) == 2:
        lr1, lr2 = lrs

    elif isinstance(lrs, (Tuple,List)) and len(lrs) == 1:
        return _set_optimizer(model, lrs[0])
    else:
        raise ValueError(f"The current implementation supports at most 2 learning rates. Found: {len(lrs)} learning rates")


    # set the optimizer
    optimizer = LARS(params=[{"params": model.fe.parameters(), "lr": lr1}, # using different learning rates with different components of the model 
                            {"params": model.flatten_layer.parameters(), "lr": lr1},
                            {"params": model.ph.parameters(), "lr": lr2}
                            ], 
                            lr=lr1, 
                            weight_decay=10 ** -6)
                

    if num_warmup_epochs > 0:
        linear_scheduler = LinearLR(optimizer=optimizer, start_factor=0.01, end_factor=1, total_iters=num_warmup_epochs)

        cosine_scheduler = CosineAnnealingLR(optimizer=optimizer, T_max=num_epochs - num_warmup_epochs)


        lr_scheduler = SequentialLR(optimizer=optimizer, 
                                    schedulers=[linear_scheduler, cosine_scheduler], 
                                    milestones=[num_warmup_epochs])

        return optimizer, lr_scheduler

    # consider the case where warm up is not needed
    cosine_scheduler = CosineAnnealingLR(optimizer=optimizer, T_max=num_epochs)

    return optimizer, cosine_scheduler 


def _run(
        model:SimClrModel,
        
        train_dl: DataLoader,
        val_dl: Optional[DataLoader],

        loss_obj: SimClrLoss,
        optimizer: torch.optim.Optimizer, 
        lr_scheduler: Optional[torch.optim.lr_scheduler.LRScheduler], 
                
        accumulate_grad: int,

        ckpnt_dir: Optional[Union[str, Path]],

        num_epochs:int,
        val_per_epoch: int,
        device: str,

        logger:Logger=None,
        batch_stats:bool=False
        ):

    # process the checkpoint directory
    ckpnt_dir = dirf.process_path(ckpnt_dir, 
                                  dir_ok=True, 
                                  file_ok=False)

    # keep two variables: min_train_loss, min_val_loss
    min_train_loss, min_val_loss = float('inf'), float('inf')

    for epoch_index in tqdm(range(num_epochs), desc=f'training the model'):
        
        # train the model for one epoch
        epoch_train_metrics = train_per_epoch(model=model, 
                        dataloader=train_dl,
                        loss_function=loss_obj,
                        epoch_index=epoch_index,
                        device=device, 
                        log_per_batch=0.1, 
                        optimizer=optimizer,
                        scheduler=lr_scheduler,
                        logger=logger,
                        batch_stats=batch_stats,
                        accumulate_grads=accumulate_grad)

        ################### Logging ###################
        epoch_train_log = epoch_train_metrics.copy()
        

        # the dict object cls not allow altering the keys while iterating through the object
        items = list(epoch_train_log.items())

        for k, val in items:
            epoch_train_log.pop(k)
            epoch_train_log[f'train_{k}'] = val

        print("#" * 25)
        print(epoch_train_log)

        # extract the epoch_train_loss
        epoch_train_loss = epoch_train_metrics["loss"]

        ######################## Train Loss Checkpointing ########################        

        if val_dl is None: # meaning we are saving checkpoints according to the trainloss
            if min_train_loss < epoch_train_loss:
                continue 

            # at this point the epoch training loss is the minimum so far
            min_train_loss = epoch_train_loss
            ckpnt_file_name = f'ckpnt_train_loss-{round(min_train_loss, 4)}_epoch-{epoch_index}.pt'

            if len(os.listdir(ckpnt_dir)) != 0:
                # keep only one checkpoint             
                dirf.clear_directory(directory=ckpnt_dir, condition=lambda _: True)

            pu.save_checkpoint(model=model, 
                                optimizer=optimizer, 
                                lr_scheduler=lr_scheduler,
                                path=os.path.join(ckpnt_dir, ckpnt_file_name),
                                train_loss=epoch_train_loss, epoch=epoch_index)

            continue

        
        # make sure to update the minimum training loss
        min_train_loss = min(min_train_loss, epoch_train_loss)

        # at this point we know that val_dl is not None and hence the validation loss is the criteria to save checkpoints
        # run the model on the validation set every "val_per_epoch" and the last epoch
        if not ((epoch_index + 1) % val_per_epoch == 0 or epoch_index == num_epochs - 1):
            continue

        epoch_val_metrics = validation_per_epoch(model=model, 
                                        dataloader=val_dl,
                                        loss_function=loss_obj,
                                        epoch_index=epoch_index + 1, 
                                        device=device,
                                        log_per_batch=0.2,
                                        logger=logger,
                                        batch_stats=batch_stats
                                        )

        # add the 'val' to each key in the 'epoch_val_metrics' dictionary
        epoch_val_log = epoch_val_metrics.copy()
        items = list(epoch_val_log.items())
        
        for k, val in items:
            epoch_val_log.pop(k)
            epoch_val_log[f'val_{k}'] = val
        
        print("#" * 25)
        print(epoch_val_log)

        # extract the validation loss
        epoch_val_loss = epoch_val_metrics["loss"]

        if epoch_val_loss > min_val_loss:
            continue


        # at this point epoch_val_loss is the minimal validation loss achieved so far
        min_val_loss = epoch_val_loss
        ckpnt_file_name = f'ckpnt_val_loss_{round(min_val_loss, 4)}_epoch_{epoch_index}.pt'

        if len(os.listdir(ckpnt_dir)) != 0:
            # keep only one checkpoint             
            dirf.clear_directory(directory=ckpnt_dir, condition=lambda _: True)

        pu.save_checkpoint(model=model, 
                            optimizer=optimizer, 
                            lr_scheduler=lr_scheduler,
                            path=os.path.join(ckpnt_dir, ckpnt_file_name),
                            val_loss=epoch_val_loss, 
                            epoch=epoch_index)

    # make sure to return the checkpoint 
    final_ckpnt = os.path.join(ckpnt_dir, ckpnt_file_name) 

    if val_dl is not None:
        return min_train_loss, min_val_loss, final_ckpnt 

    return min_train_loss, final_ckpnt


def run_pipeline(model: SimClrModel, 
        train_data_folder:Union[str, Path],          
        val_data_folder:Optional[Union[str, Path]],
        output_shape:Tuple[int, int], 
        dataset: str,
        num_epochs: int, 
        batch_size: int,

        # learning_rates: Union[Tuple[float, float], float],
        temperature: float,

        ckpnt_dir: Union[str, Path],
        num_warmup_epochs:int=10,
        val_per_epoch: int = 3,
        seed:int = 69,

        run_name: str = 'sim_clr_run',
        use_logging: bool = True,
        batch_stats:bool=False,
        debug_loss:bool=True,

        num_train_samples_per_cls: Union[int, float]=None, # the number of training samples per cls
        num_val_samples_per_cls: Union[int, float]=None, # the number of validation samples per cls
        ):    
    
    if use_logging:
        # create a clearml task object
        task = Task.init(project_name=_TRACK_PROJECT_NAME,
                         task_name=run_name,
                         )
        logger = task.get_logger()
    else:
        logger = None

    # get the default device
    device = pu.get_default_device()

    
    # set the loss object    
    loss_obj = SimClrLoss(temperature=temperature, debug=debug_loss)

    # DATA: 
    train_dl, val_dl = _set_data(train_data_folder=train_data_folder,
                                val_data_folder=val_data_folder, 
                                dataset=dataset,
                                output_shape=output_shape,
                                batch_size=batch_size, 
                                num_train_samples_per_cls=num_train_samples_per_cls,
                                num_val_samples_per_cls=num_val_samples_per_cls,
                                seed=seed)

    lr = 0.3 * (batch_size / 256) # according to the paper's formula

    optimizer, lr_scheduler = _set_optimizer(model=model, 
                                             lrs=lr, 
                                             num_warmup_epochs=num_warmup_epochs,
                                             num_epochs=num_epochs)

    res = None

    try:
        res = _run(model=model, 
                train_dl=train_dl, 
                val_dl=val_dl, 

                loss_obj=loss_obj, 
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                accumulate_grad=1, # keep it simple, no gradient accumulation for now

                ckpnt_dir=ckpnt_dir,
                num_epochs=num_epochs,
                val_per_epoch=val_per_epoch,
                device=device,
                logger=logger, 
                batch_stats=batch_stats
                )

    # this piece of code is taken from the fairSeq repo (by the Facebook AI research team) as recommmended on the Pytorch forum:
    except RuntimeError as e:
        print(e)
        # at this point, the error is known to be an out of memory error
        pu.cleanup()

    return res
