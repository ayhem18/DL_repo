import wandb, torch, os

import torchvision.transforms as tr, albumentations as A

from time import sleep
from typing import Union, Optional, Tuple, List
from pathlib import Path

from tqdm import tqdm
from torch.optim.sgd import SGD
from torch.utils.data import DataLoader
from functools import partial

from mypt.losses.object_detection.object_localization import ObjectLocalizationLoss
from mypt.data.datasets.obj_detect_loc.object_localization import ObjectLocalizationDs
from mypt.data.dataloaders.standard_dataloaders import initialize_train_dataloader, initialize_val_dataloader
from mypt.code_utilities import pytorch_utilities as pu, directories_and_files as dirf
from mypt.schedulers.annealing_lr import AnnealingLR
from mypt.models.object_detection.obj_localization import ObjectLocalizationModel
from mypt.code_utilities import annotation_utilites as au

from .train_per_epoch import train_per_epoch, validation_per_epoch
from .data_preparation import extract_annotation_from_xml, get_annotation_file_path


_WANDB_PROJECT_NAME = "ObjLoc"


def img_2_annotation(file_path: Union[str, Path], ann_folder: Union[str, Path]):
    ann_file = os.path.join(ann_folder, get_annotation_file_path(file_path))

    if not os.path.exists(ann_file):
        # this means that the sample is a background sample
        return [
            ['background'], 
            [au.DEFAULT_BBOX_BY_FORMAT[au.PASCAL_VOC]]
                ]    

    cls, bbox, img_shape = extract_annotation_from_xml(ann_file)
    if len(bbox) != 1:
        bbox = [bbox]

    if not isinstance(cls, (List, Tuple)):
        cls = [cls]

    return cls, bbox, img_shape


def _set_data(train_data_folder: Union[str, Path],
             val_data_folder: Optional[Union[str, Path]],
             annotation_folder: Union[str, Path],
             batch_size:int,
             output_shape: Tuple[int, int],
             seed:int=69) -> Tuple[DataLoader, Optional[DataLoader]]:
    
    train_data_folder = dirf.process_path(train_data_folder, dir_ok=True, file_ok=False, 
                                    condition=lambda d: all([os.path.splitext(f)[-1].lower() in dirf.IMAGE_EXTENSIONS for f in os.listdir(d)]), # basically make sure that all files are either images of .xml files
                                    error_message=f'Make sure the directory contains only image data'
                                    )


    annotation_folder = dirf.process_path(annotation_folder, dir_ok=True, file_ok=False, 
                                    condition=lambda d: all([os.path.splitext(f)[-1].lower() == '.xml' for f in os.listdir(d)]), # basically make sure that all files are either images of .xml files
                                    error_message=f'Make sure the directory contains only xml files'
                                    )

    train_ds = ObjectLocalizationDs(root_dir=train_data_folder, 
                                    img_augs=[A.RandomSizedBBoxSafeCrop(*output_shape, p=0.5), A.ColorJitter(p=0.5)],

                                    output_shape=output_shape,
                                    compact=True,

                                    image2annotation=partial(img_2_annotation, ann_folder=annotation_folder),

                                    target_format=au.ALBUMENTATIONS,
                                    current_format=au.PASCAL_VOC,
                                    background_label='background', 
                                    )


    if val_data_folder is not None:
        val_data_folder = dirf.process_path(val_data_folder, dir_ok=True, file_ok=False, 
                                        condition=lambda d: all([os.path.splitext(f)[-1].lower() in dirf.IMAGE_EXTENSIONS for f in os.listdir(d)]), # basically make sure that all files are either images of .xml files
                                        error_message=f'Make sure the directory contains only image data'
                                        )

        val_ds = ObjectLocalizationDs(root_dir=val_data_folder, 
                                        img_augs=[],

                                        output_shape=output_shape,
                                        compact=True,

                                        image2annotation=partial(img_2_annotation, ann_folder=annotation_folder),

                                        target_format=au.ALBUMENTATIONS,
                                        current_format=au.PASCAL_VOC,
                                        background_label='background', 
                                        )

    else:
        val_ds = None

    ## data loaders
    train_dl = initialize_train_dataloader(dataset_object=train_ds, 
                                         seed=seed,
                                         batch_size=batch_size,
                                         num_workers=0,
                                         warning=False # the num_workers=0 is deliberately set to 0  
                                         )

    if val_ds is not None:
        val_dl = initialize_val_dataloader(dataset_object=val_ds, 
                                            seed=seed,
                                            batch_size=batch_size,
                                            num_workers=0,
                                            )
    else:
        val_dl = None

    return train_dl, val_dl


def _set_optimizer(model: ObjectLocalizationModel, 
                  lrs: Union[Tuple[float], float], 
                  num_epochs: int) -> Tuple[SGD, AnnealingLR]:

    if isinstance(lrs, float):
        lr1, lr2 = lrs, 10 * lrs
    elif isinstance(lrs, List) and len(lrs) == 2:
        lr1, lr2 = lrs

    elif isinstance(lrs, List) and len(lrs) == 1:
        return _set_optimizer(model, lrs[0])
    else:
        raise ValueError(f"The current implementation supports at most 2 learning rates. Found: {len(lrs)} learning rates")


    # set the optimizer
    optimizer = SGD(params=[{"params": model.fe.parameters(), "lr": lr1}, # using different learning rates with different components of the model 
                            {"params": model.flatten_layer.parameters(), "lr": lr1},
                            {"params": model.head.parameters(), "lr": lr2}
                            ])

    lr_scheduler = AnnealingLR(optimizer=optimizer, 
                               num_epochs=num_epochs,
                               alpha=10,
                               beta= 0.75)

    return optimizer, lr_scheduler



def _run(
        model:ObjectLocalizationModel,
        
        train_dl: DataLoader,
        val_dl: Optional[DataLoader],

        loss_obj: ObjectLocalizationLoss,
        optimizer: torch.optim.Optimizer, 
        lr_scheduler: Optional[torch.optim.lr_scheduler.LRScheduler], 
        
        accumulate_grad: int,

        ckpnt_dir: Optional[Union[str, Path]],

        num_epochs:int,
        val_per_epoch: int,
        device: str,

        use_wandb:bool=True,
        ):

    # process the checkpoint directory
    ckpnt_dir = dirf.process_path(ckpnt_dir, 
                                  dir_ok=True, 
                                  file_ok=False)

    # keep two variables: min_train_loss, min_val_loss
    min_train_loss, min_val_loss = float('inf'), float('inf')

    for epoch_index in tqdm(range(num_epochs), desc=f'training the model'):
        epoch_train_metrics = train_per_epoch(model=model, 
                        dataloader=train_dl,
                        loss_function=loss_obj,
                        epoch_index=epoch_index,
                        device=device, 
                        log_per_batch=0.1, 
                        optimizer=optimizer,
                        scheduler=lr_scheduler,
                        use_wandb=use_wandb,
                        accumulate_grads=accumulate_grad)

        epoch_train_log = epoch_train_metrics.copy()
        
        # the dict object cls not allow altering the keys while iterating through the object
        items = list(epoch_train_log.items())
        for k, val in items:
            epoch_train_log.pop(k)
            epoch_train_log[f'train_{k}'] = val

        print("#" * 25)
        print(epoch_train_log)

        # extract the epoch_train_loss
        epoch_train_loss = epoch_train_metrics[loss_obj._loss_name]
        
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
            
        # at this point we know that val_dl is None and hence the validation loss is the criteria to save checkpoints
        # run the model on the validation set every "val_per_epoch" and the last epoch
        if not ((epoch_index + 1) % val_per_epoch == 0 or epoch_index == num_epochs - 1):
            continue
        
        # run the model on the validation dataset
        epoch_val_metrics = validation_per_epoch(model=model, 
                                        dataloader=val_dl,
                                        loss_function=loss_obj,
                                        epoch_index=epoch_index + 1, 
                                        device=device,
                                        log_per_batch=0.2,
                                        use_wandb=use_wandb,
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
        epoch_val_loss = epoch_val_metrics[loss_obj._loss_name]

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
    return os.path.join(ckpnt_dir, ckpnt_file_name)



def run_pipeline(model: ObjectLocalizationModel, 
          train_data_folder:Union[str, Path],          
          val_data_folder:Optional[Union[str, Path]],

          annotation_folder: Union[str, Path],

          output_shape:Tuple[int, int], 

          num_epochs: int, 
          batch_size: int,
        
          learning_rates: Union[Tuple[float, float], float],

          ckpnt_dir: Union[str, Path],
          val_per_epoch: int = 3,
          seed:int = 69,
          run_name: str = 'sim_clr_run',
          use_wandb: bool = True,
          ):    

    if use_wandb:
        # this argument was added to avoid initializing wandb twice during the tuning process
        wandb.init(project=_WANDB_PROJECT_NAME, 
                name=run_name)

    # get the default device
    device = pu.get_default_device()


    loss_obj = ObjectLocalizationLoss()

    # DATA: 
    train_dl, val_dl = _set_data(train_data_folder=train_data_folder,
                                val_data_folder=val_data_folder,
                                annotation_folder=annotation_folder, 
                                output_shape=output_shape,
                                batch_size=batch_size, 
                                seed=seed)
    
    # optimizer = _set_optimizer(model=model, learning_rate=initial_lr)
    optimizer, lr_scheduler = _set_optimizer(model=model, lrs=learning_rates, num_epochs=num_epochs)

    res = None
    try:
        res_ckpnt = _run(model=model, 
                train_dl=train_dl, 
                val_dl=val_dl, 

                loss_obj=loss_obj, 
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                
                accumulate_grad=1, # make sure to add the gradient accumulation factor

                ckpnt_dir=ckpnt_dir,
                num_epochs=num_epochs,
                val_per_epoch=val_per_epoch,
                device=device,
                use_wandb=use_wandb, 
                )

    # this piece of code is taken from the fairSeq repo (by the Facebook AI research team) as recommmended on the Pytorch forum:
    except RuntimeError as e:
        if 'out of memory' not in str(e):
            raise e
        
        # at this point, the error is known to be an out of memory error
        pu.cleanup()
        # make sure to close the wandb log
        wandb.finish()
        # stop the program for 30 seconds for the cleaning to take effect
        sleep(30)

        batch_size = int(batch_size / 1.2)
        res_ckpnt = run_pipeline(model=model, 
                                     
            train_data_folder=train_data_folder,

            val_data_folder=val_data_folder,

            output_shape=output_shape, 

            num_epochs=num_epochs, 
            batch_size=batch_size,

            learning_rates=learning_rates,

            ckpnt_dir=ckpnt_dir,
            val_per_epoch=val_per_epoch,
            seed=seed,
            run_name=run_name, 
            use_wandb=use_wandb, 
            )

    
    wandb.finish()
    return res_ckpnt
