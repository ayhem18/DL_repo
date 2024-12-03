"""
This script contains few functions to handle working with multiple datasets
"""

from typing import Optional, Tuple
from torch.utils.data.dataset import Dataset
from torchvision import transforms as tr

from mypt.shortcuts import P
from mypt.code_utilities import directories_and_files as dirf
from mypt.data.dataloaders.standard_dataloaders import initialize_train_dataloader, initialize_val_dataloader
from mypt.data.datasets.parallel_augmentation.parallel_aug_ds_wrapper import Food101Wrapper, ImagenetterWrapper


_DEFAULT_DATA_AUGS = [
    tr.RandomHorizontalFlip(p=0.5), 
    tr.RandomResizedCrop((200, 200), scale=(0.4, 1)),
    
    tr.RandomErasing(p=0.8, scale=(0.05, 0.15)),
    tr.RandomGrayscale(p=0.8),
    tr.GaussianBlur(kernel_size=(5, 5)),
    tr.ColorJitter(brightness=0.5, contrast=0.5)
]

_UNIFORM_DATA_AUGS = [] # [tr.Normalize(mean=[0.485, 0.456, 0.406],  std=[0.229, 0.224, 0.225])] : one of the augmentations used with the original Resnet models 


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
                     batch_size:int):

    ## data loaders
    train_dl = initialize_train_dataloader(dataset_object=train_ds, 
                                         seed=seed,
                                         batch_size=batch_size,
                                         num_workers=2,
                                         warning=False  
                                         )

    if val_ds is not None:
        val_dl = initialize_val_dataloader(dataset_object=train_ds, 
                                            seed=seed,
                                            batch_size=batch_size,
                                            num_workers=2,
                                            )
    else:
        val_dl = None

    return train_dl, val_dl


def _set_food101_ds(
            train_data_folder: P,
            val_data_folder: Optional[P],
            output_shape: Tuple[int, int],
            num_train_samples_per_cls:Optional[int],
            num_val_samples_per_cls:Optional[int]):
    
    # train_path, val_path = _process_paths(train_data_folder, val_data_folder)
    train_ds = Food101Wrapper(root_dir=train_data_folder, 
                                output_shape=output_shape,
                                augs_per_sample=2, 
                                sampled_data_augs=_DEFAULT_DATA_AUGS,
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
                                sampled_data_augs=_DEFAULT_DATA_AUGS,
                                uniform_augs_before=[],
                                uniform_augs_after=_UNIFORM_DATA_AUGS,
                                train=False,
                                samples_per_cls=num_val_samples_per_cls,    
                                )

    return train_ds, val_ds


def _set_imagenette_ds(
            train_data_folder: P,
            val_data_folder: Optional[P],
            output_shape: Tuple[int, int],
            num_train_samples_per_cls:Optional[int],
            num_val_samples_per_cls:Optional[int],    
            ):
    
    # train_path, val_path = _process_paths(train_data_folder, val_data_folder)
    train_ds = ImagenetterWrapper(root_dir=train_data_folder, 
                                output_shape=output_shape,
                                augs_per_sample=2, 
                                sampled_data_augs=_DEFAULT_DATA_AUGS,
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
                                sampled_data_augs=_DEFAULT_DATA_AUGS,
                                uniform_augs_before=[],
                                uniform_augs_after=_UNIFORM_DATA_AUGS,
                                train=False,
                                samples_per_cls=num_val_samples_per_cls,    
                                )

    return train_ds, val_ds


_ds_name_ds_function = {"imagenette": _set_imagenette_ds, "food101": _set_food101_ds}


def _set_data(train_data_folder: P,
            val_data_folder: Optional[P],
            dataset:str,
            batch_size:int,
            output_shape: Tuple[int, int],
            num_train_samples_per_cls:Optional[int],
            num_val_samples_per_cls:Optional[int],    
            seed:int=69):

    if dataset not in _SUPPORTED_DATASETS:
        raise NotImplementedError(f"The current implementation only works for the following datasets: {_SUPPORTED_DATASETS}")

    train_path, val_path = _process_paths(train_data_folder, val_data_folder)

    # make sure the dataset name aligns with the path to the dataset 
    # (passing a 'dataset' argument as imagenette but passing a path to the path where the Food101 dataset is downloaded might raise unexpected errors (or even worse act as a silent bug...))

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

    train_ds, val_ds = _ds_name_ds_function[dataset](
            train_data_folder=train_path,
            val_data_folder=val_path,
            output_shape=output_shape,
            num_train_samples_per_cls=num_train_samples_per_cls,
            num_val_samples_per_cls=num_val_samples_per_cls)


    return _set_dataloaders(train_ds, val_ds, batch_size=batch_size, seed=seed)
