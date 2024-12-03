"""
This script contains code to prepare the data for the simClr Training
"""
import os

from typing import Union
from pathlib import Path
from mypt.code_utilities import directories_and_files as dirf


SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))

def prepare_data_folder(folder_path: Union[str, Path], 
                        des_folder_name: str,
                        rename_org:bool=False):
    folder_path = dirf.process_path(folder_path,
                                file_ok=False,
                                dir_ok=True,
                                # make sure that all the sub files are indeed directories
                                condition=lambda x: all([os.path.isdir(os.path.join(x, p))
                                                        for p in os.listdir(x)]),
                                error_message=f'The root directory is expected to have only directories as'
                                                f' inner files.')

    if rename_org: 
        for cls_dir in sorted(os.listdir(folder_path)):
            cls_path = os.path.join(folder_path, cls_dir)
            for index, f in enumerate(os.listdir(cls_path)):
                ext = os.path.splitext(f)[-1]
                os.rename(os.path.join(cls_path, f), os.path.join(cls_path, f'{cls_dir}_{index}{ext}'))

    folder_parent_dir = Path(folder_path).parent
    new_folder_path = os.path.join(folder_parent_dir, des_folder_name)

    # iterate through the classes in the original data and copy the images to the new unlabeled data folders
    for cls_dir in sorted(os.listdir(folder_path)):
        dirf.copy_directories(src_dir=os.path.join(folder_path, cls_dir), des_dir=new_folder_path, copy=True)


    
if __name__ == '__main__':
    plant_seedlings = os.path.join(SCRIPT_DIR, 'plant_seedlings', 'original_data')
    prepare_data_folder(plant_seedlings, des_folder_name='unlabeled_data', rename_org=True)
