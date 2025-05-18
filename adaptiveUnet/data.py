import os

import numpy as np

from PIL import Image
from pathlib import Path

from mypt.code_utils import directories_and_files as dirf


current_dir = Path(__file__).parent

while 'data' not in os.listdir(current_dir):
    current_dir = current_dir.parent

DATA_DIR = os.path.join(current_dir, 'data') 


def build_class_mapping(annotation_file_path):
    """
    Build a dictionary mapping filenames to their class labels by reading an annotation file.
    
    Args:
        annotation_file_path (str): Path to the annotation file (e.g., trainval.txt or test.txt)
        
    Returns:
        dict: Dictionary mapping filenames to their class labels (the second number in each line)
    """
    class_mapping = {}
    
    try:
        with open(annotation_file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:  # Skip empty lines
                    continue
                    
                parts = line.split()
                if len(parts) >= 3:  # Ensure we have at least filename and class label
                    filename = parts[0]
                    class_label = int(parts[2])  # The second number is the class label
                    class_mapping[filename] = class_label

    except Exception as e:
        print(f"Error reading annotation file {annotation_file_path}: {e}")
    
    return class_mapping


def get_unique_image_shapes(folder_path):
    """
    Read all images in a folder and return their unique shapes.
    
    Args:
        folder_path (str): Path to the folder containing images
        
    Returns:
        set: Set of tuples containing unique image shapes (height, width)
    """
    unique_shapes = set()
    
    try:
        for filename in os.listdir(folder_path):
            # Skip hidden files
            if filename.startswith('.'):
                continue
                
            image_path = os.path.join(folder_path, filename)
            
            try:
                unique_shapes.add(Image.open(image_path).size)
            except Exception as e:
                print(f"Error reading {filename}: {e}")
                continue
                
        return unique_shapes
        
    except Exception as e:
        print(f"Error accessing folder {folder_path}: {e}")
        return set()


def read_gif(gif_path):
    """
    Read a GIF file and return its frames.
    
    Args:
        gif_path (str): Path to the GIF file
        
    Returns:
        list: List of PIL Image objects (frames)
    """
    frames = []
    try:
        # Open the GIF
        with Image.open(gif_path) as gif:
            # Loop through each frame
            for frame_idx in range(gif.n_frames):
                gif.seek(frame_idx)
                # Convert to RGB to ensure consistency
                frames.append(gif.convert('RGB'))
                
        return frames
    except Exception as e:
        print(f"Error reading GIF {gif_path}: {e}")
        return []


from mypt.code_utils import directories_and_files as dirf

import shutil

def prepare_data():
    original_dir = os.path.join(DATA_DIR, 'original')
    org_train_dir = os.path.join(original_dir, 'train')     
    org_train_masks_dir = os.path.join(original_dir, 'train_masks')

    working_dir = dirf.process_path(os.path.join(DATA_DIR, 'working_version'), file_ok=False, dir_ok=True, must_exist=False)
    new_train_dir = dirf.process_path(os.path.join(working_dir, 'train'), file_ok=False, dir_ok=True, must_exist=False) 
    new_train_masks_dir = dirf.process_path(os.path.join(working_dir, 'train_masks'), file_ok=False, dir_ok=True, must_exist=False)

    # copy directories
    shutil.copytree(org_train_dir, new_train_dir, dirs_exist_ok=True)
    shutil.copytree(org_train_masks_dir, new_train_masks_dir, dirs_exist_ok=True)
    
    # convert each .gif file to a .png file 
    files = [file for file in os.listdir(new_train_masks_dir) if file.endswith('.gif')]

    for file in files:
        gif_path = os.path.join(new_train_masks_dir, file)
        frames = read_gif(gif_path)

        if len(frames) != 1:
            raise ValueError(f"Expected 1 frame in {file}, got {len(frames)}")

        frame = frames[0]

        file_name = os.path.splitext(file)[0]

        if file_name.endswith('_mask'):
            new_file_name = file_name[:-5] + '.png'
        else:
            new_file_name = file_name + '.png'

        # remove the .gif file
        os.remove(os.path.join(new_train_masks_dir, file))

        # save the frame as a .png file
        frame.save(os.path.join(new_train_masks_dir, new_file_name))

    

    # create a validation set
    val_dir = dirf.process_path(os.path.join(working_dir, 'val'), file_ok=False, dir_ok=True, must_exist=False)
    val_masks_dir = dirf.process_path(os.path.join(working_dir, 'val_masks'), file_ok=False, dir_ok=True, must_exist=False)
    
    # move 10% of the files to the validation set
    dirf.directory_partition(new_train_dir, val_dir, portion=0.1, copy=False)

    # make sure to move the corresponding masks
    val_masks_names = set([os.path.splitext(file)[0] for file in os.listdir(val_dir)])

    # copy the masks to the validation set  
    dirf.copy_directories(new_train_masks_dir, val_masks_dir, copy=False, filter_directories=lambda f: os.path.splitext(f)[0] in val_masks_names)

    # remove the validation masks from the train directory
    dirf.clear_directory(new_train_masks_dir, lambda f: os.path.splitext(f)[0] in val_masks_names)

    train_files = [os.path.splitext(file)[0] for file in os.listdir(new_train_dir)]
    train_masks_files = [os.path.splitext(file)[0] for file in os.listdir(new_train_masks_dir)]

    assert sorted(train_files) == sorted(train_masks_files), "The train files and the train masks files are not the same"

    val_files = [os.path.splitext(file)[0] for file in os.listdir(val_dir)]
    val_masks_files = [os.path.splitext(file)[0] for file in os.listdir(val_masks_dir)]

    assert sorted(val_files) == sorted(val_masks_files), "The validation files and the validation masks files are not the same"

    assert set(train_files) & set(val_files) == set(), "There are some files that are in both the train and the val set"
    

if __name__ == '__main__':
    prepare_data()



