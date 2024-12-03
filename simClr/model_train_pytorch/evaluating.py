"""
This script contains the implementation of the downstream evaluation process of the code
"""

import numpy as np

from typing import Union, Optional, Tuple
from pathlib import Path


# from mypt.data.dataloaders.standard_dataloaders import initialize_train_dataloader, initialize_val_dataloader
from mypt.code_utilities import directories_and_files as dirf
from mypt.subroutines.neighbors.knn import KnnClassifier
from mypt.models.simClr.simClrModel import SimClrModel
from .ds_wrapper import Food101Wrapper
from .training import _UNIFORM_DATA_AUGS


def _set_data_classification_data(train_data_folder: Union[str, Path],
            val_data_folder: Optional[Union[str, Path]],
            # batch_size:int,
            output_shape: Tuple[int, int],
            num_train_samples_per_cls:Optional[int],
            num_val_samples_per_cls:Optional[int],    
            # seed:int=69
            ) -> Tuple[Food101Wrapper, Food101Wrapper]:
    
    train_data_folder = dirf.process_path(train_data_folder, 
                                          dir_ok=True, 
                                          file_ok=False, 
                                          )

    train_ds = Food101Wrapper(root_dir=train_data_folder, 
                                output_shape=output_shape,
                                augs_per_sample=2, 
                                sampled_data_augs=[],
                                uniform_augs_before=[],
                                uniform_augs_after=_UNIFORM_DATA_AUGS,
                                train=True,
                                samples_per_cls=num_train_samples_per_cls,
                                classification_mode=True,
                                )

    if val_data_folder is not None:
        val_data_folder = dirf.process_path(val_data_folder, 
                                          dir_ok=True, 
                                          file_ok=False, 
                                          )

        val_ds = Food101Wrapper(root_dir=val_data_folder, 
                                output_shape=output_shape,
                                augs_per_sample=2, 
                                sampled_data_augs=[],
                                uniform_augs_before=[],
                                uniform_augs_after=_UNIFORM_DATA_AUGS,
                                train=False,
                                samples_per_cls=num_val_samples_per_cls,
                                classification_mode=True)

    else:
        val_ds = None

    return train_ds, val_ds


EVAL_BATCH_SIZE = 1024

def evaluate_model(
            model: SimClrModel,
            model_ckpnt: Optional[Union[str, Path]],
            train_data_folder: Union[str, Path],
            val_data_folder: Union[str, Path],
            output_shape: Tuple[int, int],
            num_train_samples_per_cls:Optional[int],
            num_val_samples_per_cls:Optional[int],
            num_neighbors: int,
            inference_batch_size:int=EVAL_BATCH_SIZE
            ) -> float:

    if val_data_folder is None:
        raise TypeError(f"Make sure to pass the validation dataset")

    # get the data
    train_ds, val_ds = _set_data_classification_data(train_data_folder, val_data_folder, 
                                            output_shape=output_shape, 
                                            num_train_samples_per_cls=num_train_samples_per_cls, 
                                            num_val_samples_per_cls=num_val_samples_per_cls)

    classifier = KnnClassifier(train_ds=train_ds, 
                               train_ds_inference_batch_size=inference_batch_size,
                                process_model_output=lambda m, x: m(x)[0], # the forward call returns a tuple 
                                model=model,
                                model_ckpnt=model_ckpnt,
                                process_item_ds=lambda i:i[0], # ignore the class label from the dataset during training
                               )        

    # use the classifier to predict
    predictions = classifier.predict(val_ds=val_ds, 
                                    inference_batch_size=inference_batch_size, 
                                    num_neighbors=num_neighbors, 
                                    measure='cosine_sim', 
                                    measure_as_similarity=True, 
                                    num_workers=2,
                                    process_item_ds=lambda i:i[0], # ignore the class label from the dataset during training
                                    )

    # extract the correct labels
    labels = np.asarray([val_ds[i][1] for i in range(len(val_ds))])

    # calculate the accuracy
    acc = float(round(np.mean((predictions == labels).astype(int)), 4))

    return acc
