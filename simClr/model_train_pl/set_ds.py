"""
This script contains few functions to handle working with multiple datasets
"""
import os
from pathlib import Path
from typing import Optional, Tuple, List, Iterable
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms as tr

from mypt.shortcuts import P
from mypt.code_utilities import directories_and_files as dirf
from mypt.data.dataloaders.standard_dataloaders import initialize_train_dataloader, initialize_val_dataloader
from mypt.data.datasets.parallel_augmentation.parallel_aug_ds_wrapper import Food101Wrapper, ImagenetterWrapper
from mypt.data.datasets.parallel_augmentation.parallel_aug_dir import ParallelAugDirDs 
from mypt.data.datasets.genericFolderDs import GenericFolderDS
from mypt.data.datasets.genericFolderDs import ImagenetteGenericWrapper, Food101GenericWrapper

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
current = SCRIPT_DIR

while 'data' not in os.listdir(current):
    current = Path(current).parent

DATA_FOLDER = os.path.join(current, 'data')    


_DEFAULT_DATA_AUGS = [
    tr.RandomHorizontalFlip(p=1),
    tr.RandomVerticalFlip(p=1),

    tr.RandomResizedCrop((200, 200), scale=(0.7, 1)),    
    tr.RandomErasing(p=1, scale=(0.05, 0.15)),

    tr.RandomGrayscale(p=1),
    tr.GaussianBlur(kernel_size=(5, 5)),
    tr.ColorJitter(brightness=0.5, 
                   contrast=0.5)
]

_UNIFORM_DATA_AUGS = []


_SUPPORTED_DATASETS = ['imagenette', 'food101']


def _process_paths(train_path: P, val_path: Optional[P]) -> Tuple[P, Optional[P]]:
    train_data_folder = dirf.process_path(train_path, 
                                          dir_ok=True, 
                                          file_ok=False, 
                                          )
    
    val_data_folder = dirf.process_path(val_path, 
                                        dir_ok=True, 
                                        file_ok=False, 
                                        )

    return train_data_folder, val_data_folder


def _set_dataloaders(train_ds: Dataset, 
                     val_ds: Optional[Dataset],
                     seed:int, 
                     
                     train_batch_size:int, 
                     val_batch_size:int,
                     samples_weights: Optional[Iterable[float]],
                     ) -> Tuple[DataLoader, Optional[DataLoader]]:

    ## data loaders
    # the weights are only passed to the train dataloader
    train_dl = initialize_train_dataloader(dataset_object=train_ds, 
                                         seed=seed,
                                         batch_size=train_batch_size,
                                         num_workers=2,
                                         warning=False,
                                         weights=samples_weights,
                                         )

    if val_ds is not None:
        val_dl = initialize_val_dataloader(dataset_object=train_ds, 
                                            seed=seed,
                                            batch_size=val_batch_size,
                                            num_workers=2,
                                            )
    else:
        val_dl = None

    return train_dl, val_dl


def _set_food101_ds_debug_ds(
            train_data_folder: P,
            output_shape: Tuple[int, int],
            batch_size:int,
            num_train_samples_per_cls:Optional[int],
            ):
    
    train_path = dirf.process_path(train_data_folder, 
                                    dir_ok=True, 
                                    file_ok=False,
                                    must_exist=True 
                                    )
    
    train_ds = Food101GenericWrapper(root_dir=train_path, 
                                    augmentations=[tr.ToTensor(), tr.Resize(size=output_shape)],
                                    train=True,
                                    samples_per_cls=num_train_samples_per_cls)
    
    return initialize_train_dataloader(dataset_object=train_ds, 
                                       seed=0, 
                                       batch_size=batch_size, 
                                       num_workers=2, 
                                       drop_last=False, 
                                       warning=False, 
                                       pin_memory=False)

def _set_food101_ds(
            train_data_folder: P,
            val_data_folder: Optional[P],
            output_shape: Tuple[int, int],
            num_train_samples_per_cls:Optional[int],
            num_val_samples_per_cls:Optional[int],
            num_sampled_augs: int,
            sampled_augs: List):
    
    # train_path, val_path = _process_paths(train_data_folder, val_data_folder)
    train_ds = Food101Wrapper(root_dir=train_data_folder, 
                                output_shape=output_shape,
                                augs_per_sample=num_sampled_augs, 
                                sampled_data_augs=sampled_augs,
                                uniform_augs_before=[],
                                uniform_augs_after=_UNIFORM_DATA_AUGS,
                                train=True,
                                samples_per_cls=num_train_samples_per_cls
                                )

    if val_data_folder is not None:
        val_data_folder = dirf.process_path(val_data_folder, 
                                          dir_ok=True, 
                                          file_ok=False, 
                                          )

        val_ds = Food101Wrapper(root_dir=val_data_folder, 
                                output_shape=output_shape,
                                augs_per_sample=2, 
                                sampled_data_augs=sampled_augs,
                                uniform_augs_before=[],
                                uniform_augs_after=_UNIFORM_DATA_AUGS,
                                train=False,
                                samples_per_cls=num_val_samples_per_cls,    
                                )

    return train_ds, val_ds


def _set_imagenette_ds_debug(
            train_data_folder: P,
            output_shape: Tuple[int, int],
            batch_size:int,
            num_train_samples_per_cls:Optional[int],
            ):
    
    train_path = dirf.process_path(train_data_folder, 
                                    dir_ok=True, 
                                    file_ok=False,
                                    must_exist=True 
                                    )
    
    train_ds = ImagenetteGenericWrapper(root_dir=train_path, 
                                        augmentations=[tr.ToTensor(), tr.Resize(size=output_shape)],
                                        train=True,
                                        samples_per_cls=num_train_samples_per_cls)
    
    return initialize_train_dataloader(dataset_object=train_ds, 
                                       seed=0, 
                                       batch_size=batch_size, 
                                       num_workers=2, 
                                       drop_last=False, 
                                       warning=False, 
                                       pin_memory=False)

def _set_imagenette_ds(
            train_data_folder: P,
            val_data_folder: Optional[P],
            output_shape: Tuple[int, int],
            num_train_samples_per_cls:Optional[int],
            num_val_samples_per_cls:Optional[int],
            num_sampled_augs:int,    
            sampled_augs: List,
            ):
    
    # train_path, val_path = _process_paths(train_data_folder, val_data_folder)
    train_ds = ImagenetterWrapper(root_dir=train_data_folder, 
                                output_shape=output_shape,
                                augs_per_sample=num_sampled_augs, 
                                sampled_data_augs=sampled_augs,
                                uniform_augs_before=[],
                                uniform_augs_after=_UNIFORM_DATA_AUGS,
                                train=True,
                                samples_per_cls=num_train_samples_per_cls
                                )

    if val_data_folder is not None:
        val_data_folder = dirf.process_path(val_data_folder, 
                                          dir_ok=True, 
                                          file_ok=False, 
                                          )

        val_ds = ImagenetterWrapper(root_dir=val_data_folder, 
                                output_shape=output_shape,
                                augs_per_sample=2, 
                                sampled_data_augs=sampled_augs,
                                uniform_augs_before=[],
                                uniform_augs_after=_UNIFORM_DATA_AUGS,
                                train=False,
                                samples_per_cls=num_val_samples_per_cls,    
                                )

    return train_ds, val_ds


_ds_name_ds_function = {"imagenette": _set_imagenette_ds, 
                        "food101": _set_food101_ds}

_ds_name_ds_debug_function = {"imagenette": _set_imagenette_ds_debug, 
                        "food101": _set_food101_ds_debug_ds}


def _verify_paths(dataset: str, 
                  train_data_folder: P, 
                  val_data_folder: Optional[P],
                  output_shape: Tuple[int, int]
                  ):
    if dataset not in _SUPPORTED_DATASETS:
        raise NotImplementedError(f"The current implementation only works for the following datasets: {_SUPPORTED_DATASETS}")

    train_path, val_path = _process_paths(train_data_folder, val_data_folder)

    # make sure the dataset name aligns with the path to the dataset 
    # (passing a 'dataset' argument as imagenette but passing a path where the Food101 dataset is downloaded might raise unexpected errors (or even worse: a silent bug...))

    for d in _SUPPORTED_DATASETS:
        if d == dataset:
            if not (d in str(train_path) and d in str(val_path)):
                raise ValueError(f"The dataset name {dataset} must be present in the train and validation paths !!")
        else:
            if d in str(train_path) or d in str(val_path):
                raise ValueError(f"Only the selected dataset name can be present in either the train and validation paths")

    # copy the default data augmentations
    sampled_aus = _DEFAULT_DATA_AUGS.copy()

    for a in sampled_aus:
        if 'resize' or 'size' in str(type(a)) and hasattr(a, 'size'):
            # set any intermediate augmentation that invovles resizing to the correct size
            a.size = output_shape

    return train_path, val_path, sampled_aus


def _set_data(
            # dataset arguments
            train_data_folder: P,
            val_data_folder: Optional[P],
            dataset:str,

            num_sampled_augs:int,
            num_train_samples_per_cls:Optional[int],
            num_val_samples_per_cls:Optional[int],
            output_shape: Tuple[int, int],

            # dataloader arguments
            train_batch_size: int,
            val_batch_size: int,
            samples_weights: Optional[Iterable[float]],        
            seed:int=69) -> Tuple[DataLoader, Optional[DataLoader]]:

    train_path, val_path, sampled_augs = _verify_paths(dataset=dataset, 
                                                       train_data_folder=train_data_folder,
                                                       val_data_folder=val_data_folder,
                                                       output_shape=output_shape)

    train_ds, val_ds = _ds_name_ds_function[dataset](
            train_data_folder=train_path,
            val_data_folder=val_path,
            num_sampled_augs=num_sampled_augs,

            num_train_samples_per_cls=num_train_samples_per_cls,
            num_val_samples_per_cls=num_val_samples_per_cls,
            output_shape=output_shape,
            sampled_augs=sampled_augs,
            )

    train_dl_debug = _ds_name_ds_debug_function[dataset] (train_data_folder=train_data_folder, 
                                              output_shape=output_shape, 
                                              batch_size=val_batch_size,
                                              num_train_samples_per_cls=num_train_samples_per_cls
                                              )

    train_dl, val_dl = _set_dataloaders(train_ds, 
                                        val_ds, 
                                        train_batch_size=train_batch_size, 
                                        val_batch_size=val_batch_size,
                                        samples_weights=samples_weights,
                                        seed=seed)

    return train_dl, val_dl, train_dl_debug

def _set_dataset_tune(
            train_data_folder: P,
            val_data_folder: Optional[P],
            output_shape: Tuple[int, int],
            sampled_augs: List):
        
    train_ds = ParallelAugDirDs(root_dir=train_data_folder, 
                                output_shape=output_shape,
                                augs_per_sample=2, 
                                sampled_data_augs=sampled_augs,
                                uniform_augs_before=[],
                                uniform_augs_after=_UNIFORM_DATA_AUGS,
                                )

    if val_data_folder is not None:
        val_data_folder = dirf.process_path(val_data_folder, 
                                          dir_ok=True, 
                                          file_ok=False, 
                                          )

        val_ds = ParallelAugDirDs(root_dir=val_data_folder, 
                                output_shape=output_shape,
                                augs_per_sample=2, 
                                sampled_data_augs=sampled_augs,
                                uniform_augs_before=[],
                                uniform_augs_after=_UNIFORM_DATA_AUGS,
                                )

    return train_ds, val_ds


def _set_data_tune(
            train_data_folder: P,
            val_data_folder: Optional[P],
            dataset:str,
            batch_size:int,
            output_shape: Tuple[int, int],
            seed:int=69) -> Tuple[DataLoader, Optional[DataLoader]]:

    train_path, val_path, sampled_augs = _verify_paths(dataset=dataset, 
                                                       train_data_folder=train_data_folder,
                                                       val_data_folder=val_data_folder,
                                                       output_shape=output_shape)

    train_ds, val_ds = _set_dataset_tune(
            train_data_folder=train_path,
            val_data_folder=val_path,
            output_shape=output_shape,
            sampled_augs=sampled_augs
            )

    return _set_dataloaders(train_ds, val_ds, batch_size=batch_size, seed=seed)
