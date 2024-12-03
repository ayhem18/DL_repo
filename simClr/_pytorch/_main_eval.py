import os, random
import numpy as np

import torchvision.transforms as tr
import matplotlib.pyplot as plt
from typing import Dict
from pathlib import Path

from mypt.code_utilities import directories_and_files as dirf
from mypt.models.simClr.simClrModel import ResnetSimClr

from model_train_pytorch.training import OUTPUT_SHAPE
from model_train_pytorch.evaluating import _set_data_classification_data
from mypt.subroutines.neighbors.knn import KNN
from mypt.shortcuts import P

from simClr._main_train import _OUTPUT_DIM

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))


def visualize_neighbors(evaluation_result: P | Dict, 
                        train_per_cls:int|None=None, 
                        val_per_cls:int|None=None,
                        num_images:int=10):

    train_folder = os.path.join(SCRIPT_DIR, 'data', 'food101', 'train')
    val_folder = os.path.join(SCRIPT_DIR, 'data', 'food101', 'val')


    if isinstance(evaluation_result, (str, Path)):
        import pickle
        with open(evaluation_result, 'rb') as f:
            evaluation_result = pickle.load(f)        


    train_ds, val_ds = _set_data_classification_data(train_folder, 
                                                     val_folder, 
                                                     OUTPUT_SHAPE[1:], 
                                                     train_per_cls, 
                                                     val_per_cls)

    results_sample = random.sample(list(evaluation_result.items()), num_images)

    for sample_index, info in results_sample:
        n_indices, n_metrics = info['neighbor_indices'], info['neighbor_metrics']
        x, val_label = val_ds[sample_index]
        x = np.moveaxis(x.numpy(), 0,-1)

        fig = plt.figure() 
        fig.add_subplot(1, 1 + len(n_indices), 1) 
        plt.imshow(x)
        plt.title(f"original image, label: {val_label}")

        for rank, (index, measure) in enumerate(zip(n_indices, n_metrics)):
            n, n_label = train_ds[index] 
            n = np.moveaxis(n.numpy(), 0, -1)
            fig.add_subplot(1, 1 + len(n_indices), rank + 2)
            plt.imshow(n)
            plt.title(f"nearest neighbor: {rank + 1}\nlabel: {n_label}\nsimilarity: {round(measure, 4)}")

        plt.show()


def evaluate(model, 
             model_ckpnt, 
             train_per_cls:int, 
             val_per_cls: int,
             ):
    train_folder = os.path.join(SCRIPT_DIR, 'data', 'food101', 'train')
    val_folder = os.path.join(SCRIPT_DIR, 'data', 'food101', 'val')

    train_ds, val_ds = _set_data_classification_data(train_folder, val_folder, OUTPUT_SHAPE[1:], train_per_cls, val_per_cls)
    
    knn = KNN(train_ds=train_ds, 
                train_ds_inference_batch_size=5000, # change this value depending on the computational resources
                process_model_output=lambda m, x: m(x)[0], # the forward call returns a tuple 
                model=model,
                model_ckpnt=model_ckpnt,
                process_item_ds=lambda i:i[0], # ignore the class label from the dataset during training
                ) 

    metrics, indices = knn.predict(val_ds, 
                                inference_batch_size=5000, # change this value depending on the computational resources  
                                num_neighbors=5, 
                                measure='cosine_sim',
                                measure_as_similarity=True
                                )

    labels = np.asarray([val_ds[i][1] for i in range(len(val_ds))])
    

    res = {}
    for i in range(len(labels)):
        res[i] = {"neighbor_indices": indices[i, :].tolist(), 
                  "neighbor_metrics": metrics[i, :].tolist()}

    results_directory = os.path.join(SCRIPT_DIR, 'eval_res')

    res_file_name = os.path.splitext(os.path.basename(model_ckpnt))[0] + "_results.obj"

    res_path = dirf.process_path(results_directory, file_ok=False)
    res_path = os.path.join(results_directory, res_file_name)

    import pickle
    with open(res_path, 'wb') as f:
        pickle.dump(res, f)

    return res


if __name__ == '__main__':
    model = ResnetSimClr(input_shape=OUTPUT_SHAPE,  
                         output_dim=_OUTPUT_DIM, 
                         num_fc_layers=4, 
                         freeze=False, 
                         architecture=101)

    ckpnt = os.path.join(SCRIPT_DIR, 'logs', 'train_logs', 'iteration_8', 'ckpnt_val_loss_4.6184_epoch_49.pt')

    # evaluate(model=model, 
    #          model_ckpnt=ckpnt, 
    #          train_per_cls=100, 
    #          val_per_cls=100)

    res = os.path.join(SCRIPT_DIR, 'eval_res', 'ckpnt_val_loss_4.6184_epoch_49_results.obj')

    visualize_neighbors(res, train_per_cls=100, val_per_cls=100, num_images=25)
