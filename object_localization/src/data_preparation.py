"""
This script contains the code to prepare the data for training on the object localization task
"""

import os, random, numpy as np
import xml.etree.ElementTree as ET

from typing import Union, Tuple, List, Optional
from pathlib import Path

from mypt.code_utilities import directories_and_files as dirf
from PIL import Image

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
PARENT_DIR = Path(SCRIPT_DIR).parent

def extract_annotation_from_xml(xml_file_path: Union[str, Path]) -> Tuple[Tuple[int, int], List[int]]:
    tree = ET.parse(xml_file_path)
    root = tree.getroot()

    size = {
        'width': int(root.find('size').find('width').text),
        'height': int(root.find('size').find('height').text),
    }

    # find the 
    x_min = int(root.find('object').find('bndbox').find('xmin').text)
    x_max = int(root.find('object').find('bndbox').find('xmax').text)
    y_min = int(root.find('object').find('bndbox').find('ymin').text)
    y_max = int(root.find('object').find('bndbox').find('ymax').text)

    # extract the classs, bounding box and image shape
    return [root.find('object').find('name').text.lower()], [x_min, y_min, x_max, y_max], (size['width'], size['height']) 


def get_annotation_file_path(sample_file_path: Union[str, Path]) -> Union[str, Path]:
    file_name = os.path.basename(sample_file_path)
    file_name, _ = os.path.splitext(file_name)
    return f'{file_name}.xml'    


def convert_img_2_background(img: np.ndarray, bbox:List[int]) -> np.ndarray:
    img = img.copy()
    x_min, y_min, x_max, y_max = bbox
    b = img[y_min : y_max + 1, x_min: x_max + 1, :]
    
    # working with np.pad is such a hassle
    # let's do this the old way
    padded_b = np.zeros(shape=img.shape, dtype=img.dtype)
    padded_b[y_min : y_max + 1, x_min: x_max + 1, :] = b

    img_b = img - padded_b

    sums = np.sum(img_b, axis=(0, 1))
    area = (img.shape[0] * img.shape[1]) - ((x_max - x_min + 1) * (y_max - y_min + 1))
    values = (sums / area).astype(img.dtype)
    
    img[y_min : y_max + 1, x_min: x_max + 1, :] = values
    return img


def seperate_data(data_folder: Union[str, Path], ann_folder: Union[str, Path]=None):
    data_folder = dirf.process_path(data_folder, dir_ok=True, file_ok=False, 
                                    condition=lambda d: all([os.path.splitext(f)[-1].lower() in ['.xml'] + dirf.IMAGE_EXTENSIONS for f in os.listdir(d)]), # basically make sure that all files are either images of .xml files
                                    error_message=f'Make sure the directory contains only .xml or image data'
                                    )
    if ann_folder is None:
        ann_folder = os.path.join(Path(data_folder).parent, f'{os.path.basename(data_folder)}_annotations')
        ann_folder = dirf.process_path(ann_folder, dir_ok=True, file_ok=False)
        
    else:
        ann_folder = dirf.process_path(ann_folder, 
                                       dir_ok=True, 
                                       file_ok=False, 
                                       condition=lambda d: (len(os.listdir(d))) == 0, 
                                       error_message=f"The annotation dir must be empty ")

    # move the .xml files to annotation folder
    dirf.copy_directories(src_dir=data_folder, 
                          des_dir=ann_folder, 
                          copy=False, # setting copy to False will move the files 
                          filter_directories=lambda p: os.path.splitext(p)[-1].lower() == ".xml" #move onle .xml files
                          )
    return ann_folder


def add_background_cls(data_folder: Union[str, Path],
                       ann_folder: Union[str, Path], 
                       num_samples:int, 
                       seed:int=10):
    data_folder = dirf.process_path(data_folder, dir_ok=True, file_ok=False, 
                                    condition=lambda d: all([os.path.splitext(f)[-1].lower() in dirf.IMAGE_EXTENSIONS for f in os.listdir(d)]), # basically make sure that all files are either images of .xml files
                                    error_message=f'Make sure the directory contains only image data'
                                    )

    ann_folder = dirf.process_path(ann_folder, dir_ok=True, file_ok=False, 
                                    condition=lambda d: all([os.path.splitext(f)[-1].lower() == '.xml' for f in os.listdir(d)]), # basically make sure that all files are either images of .xml files
                                    error_message=f'Make sure the directory contains only xml files'
                                    )
    
    # iterate through each img in the datafolder
    # 1. extract the bounding boxes
    # 2. calculate the average pixel value outside of the bounding box for each channel
    # 3. set the pixel values within the bounding box to the estimated values in step 2

    random.seed(seed)
    files = [os.path.join(data_folder, f) for f in os.listdir(data_folder)]
    
    num_samples = int(round(len(files) * num_samples)) if isinstance(num_samples, float) else num_samples

    files = random.sample(files, num_samples)
    
    for index, f in enumerate(files, start=1):
        _, ext = os.path.splitext(f)
        img = np.asarray(Image.open(f))
        ann_path = os.path.join(ann_folder, get_annotation_file_path(f))
        _, bbox, _ = extract_annotation_from_xml(ann_path)
        bg_img = convert_img_2_background(img, bbox)                
        # save the image 
        Image.fromarray(bg_img).save(os.path.join(data_folder, f'background_{index}{ext}'))    


def group_data_by_cls(data_folder: Union[str, Path], 
                      des_dir: Optional[Union[str, Path]]=None, 
                      copy=False):
    if des_dir is None:
        des_dir = data_folder

    # first check the classes
    classes = set()
    for f in os.listdir(data_folder):
        file_name, _ = os.path.splitext(f)
        cls =file_name.split('_')[0].lower()
        classes.add(cls)

    assert sorted(list(classes)) == ['background', 'cucumber', 'eggplant', 'mushroom'], f"a different list of classes than expected: {classes}"
    
    for c in classes:
        cf = os.path.join(des_dir, c) 
        os.makedirs(cf, exist_ok=True)
        dirf.copy_directories(src_dir=data_folder, 
                              des_dir=cf, 
                              copy=copy, 
                              filter_directories=
                              lambda file: (os.path.isfile(os.path.join(data_folder, file)) and os.path.basename(file).startswith(c)) # copy only files (no dirs) that start with the cls name
                              )


def split_data_train_val(data_folder: Union[str, Path], val_split:float=0.15):
    train_dir = os.path.join(Path(data_folder).parent, f'{os.path.basename(data_folder)}_train')
    val_dir = os.path.join(Path(data_folder).parent, f'{os.path.basename(data_folder)}_val')

    # move the val split
    dirf.classification_ds_partition(directory_with_classes=data_folder,destination_directory=val_dir, copy=False, portion=val_split)
    dirf.classification_ds_partition(directory_with_classes=data_folder, destination_directory=train_dir, copy=True, portion=1.0)

    return train_dir, val_dir


def prepare_data(data_folder: Union[str, Path], ann_folder: Optional[Union[str, Path]]=None) -> Tuple[str, str, str]:
    # 1. seperate the data from the annotation files
    ann_folder = seperate_data(data_folder=data_folder, ann_folder=ann_folder)
    # 2. create the background class
    add_background_cls(data_folder=data_folder, ann_folder=ann_folder, num_samples=0.4,)
    # 3. group the files by cls: to guarantee a similar split for all classes
    group_data_by_cls(data_folder=data_folder)
    # 4. split the data into train and validation splits
    train_dir, val_dir = split_data_train_val(data_folder=data_folder, val_split=0.15)  

    # flatten both the training and validation folders: required by the object localization dataset class
    for c in os.listdir(train_dir):
        c = os.path.join(train_dir, c)
        dirf.copy_directories(c, train_dir, copy=False)

    for c in os.listdir(val_dir):
        c = os.path.join(val_dir, c)
        dirf.copy_directories(c, val_dir, copy=False)

    return ann_folder, train_dir, val_dir


if __name__ == '__main__':
    folder = os.path.join(PARENT_DIR, 'data', 'plants_ds')    
    ann_folder = os.path.join(PARENT_DIR, 'data', 'plants_ds_annotations')    
    prepare_data(data_folder=folder)

    # split_data_train_val(folder)

    # data preparation is taken across 3 steps:

    # move the annotation to a different folder
    # add some background images
    # split the data into train and validation splits

    # add_background_cls(data_folder=folder, ann_folder=ann_folder, num_samples=0.4)
    # group_data_by_cls(data_folder=folder)

    # import cv2 as cv
    # import matplotlib.pyplot as plt

    # files = random.sample(os.listdir(folder), 10)
    # for f in files:
    #     f = os.path.join(folder, f)
    #     ann_path = os.path.join(ann_folder, get_annotation_file_path(f))
    #     _, bbox, _ = extract_annotation_from_xml(ann_path)

    #     img = np.asarray(Image.open(f)).copy()

    #     xmin, ymin, xmax, ymax = bbox
    #     # cv.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255,0), 1)    
    #     # plt.imshow(img)
    #     # plt.show()

    #     b_img = convert_img_2_background(img, bbox)
    #     plt.imshow(b_img)
    #     plt.show()
