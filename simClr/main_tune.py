"""
This script contains the code to tune the SimClr models on both the 
"""

import os

from pathlib import Path
from typing import Tuple

from mypt.shortcuts import P
from mypt.data.datasets.image_classification import utils as icu
from mypt.code_utilities import directories_and_files as dirf

from model_train_pl.tuning_optimizer import tune_main_function
from model_train_pl.constants import VAL_SPLIT, _LARGEST_BATCH_SIZE

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
current = SCRIPT_DIR
while 'data' not in os.listdir(current):
    current = Path(current).parent

DATA_FOLDER = os.path.join(current, 'data')


def tuning_data_preparation(imagenette_ds_folder: P,
                            des_data_folder: P) -> Tuple[P, P, P]:
    """
    This function assumes the dataset was downloaded using the builtin Pytorch dataset. 
    The data will be moved to a different direction
    """
    train_split, test_split = icu.builtin_imagenette2img_cls_ds(ds_folder=imagenette_ds_folder, 
                                            des_folder=des_data_folder,
                                            split_names=None, # use the default names
                                            )
    
    val_split = os.path.join(Path(train_split).parent, 'val') 

    # split the train data further into training and validation (the validation split will be used for tuning purposes) 
    dirf.classification_ds_partition(directory_with_classes=train_split, 
                                     destination_directory=val_split, 
                                     portion=VAL_SPLIT,
                                     copy=False)

    return train_split, val_split, test_split
    # the next step is to simply wrap the folders with instances of the DataFolder class
    
    
def tune_main():
    imagenette_tune = os.path.join(DATA_FOLDER, 'imagenette_tune')

    if sorted(os.listdir(imagenette_tune)) != ['test', 'train', 'val']:
        raise ValueError((f"the data folder is not in the expected structure. Expected 3 subfolders: {['train', 'test', 'val']}. " 
                         f"Found: {os.listdir(imagenette_tune)}. Make sure to call the 'tuning_data_preparation' fuction"))


    parent_log_dir = dirf.process_path(os.path.join(SCRIPT_DIR, 'logs', 'tune_logs'), 
                                       dir_ok=True, file_ok=False)
    
    # make sure to exclude all the empty directories
    dirf.clear_directory(directory=parent_log_dir, condition=lambda x: os.path.isdir(x) and len(os.listdir(x)) == 0)

    n = len(os.listdir(parent_log_dir))

    tune_parent_dir = os.path.join(parent_log_dir, f'tune_exp_{n + 1}')

    tune_main_function(train_data_folder=os.path.join(imagenette_tune, 'train'),
                       val_data_folder=os.path.join(imagenette_tune, 'val'),
                       dataset='imagenette',
                       tune_exp_number=n + 1,
                       num_epochs_per_job=2,
                       val_per_epoch=1,
                       batch_size=_LARGEST_BATCH_SIZE,
                       parent_log_dir=tune_parent_dir,
                       num_warmup_epochs=0,
                       )


if __name__ == '__main__':  
    imagenette_ds_folder = os.path.join(DATA_FOLDER, 'imagenette', 'train')
    imagenette_tune = os.path.join(DATA_FOLDER, 'imagenette_tune')

    # make sure the function commented below is called at least once !!! (to prepare the data for hyper-parameter tuning...)
    train_folder, val_folder, test_folder = tuning_data_preparation(imagenette_ds_folder=imagenette_ds_folder, 
                                                                    des_data_folder=imagenette_tune)


    # tune_main()
