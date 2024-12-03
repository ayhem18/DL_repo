"""
This script contains the code to prepare the data for training on the object localization task
"""

import os
import albumentations as A
import numpy as np

from typing import Union, List, Tuple
from pathlib import Path
from functools import partial

from mypt.code_utilities import annotation_utilites as au
from mypt.code_utilities import pytorch_utilities as pu
from mypt.data.datasets.obj_detect_loc.object_localization import ObjectLocalizationDs
from mypt.models.object_detection.obj_localization import AlexnetObjectLocalization

from src.training import run_pipeline, img_2_annotation

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))


# def debug():
#     samples = [os.path.join(train_data, f) for f in os.listdir(train_data)] 

#     ds = ObjectLocalizationDs(root_dir=train_data, 
#                               img_augs=[],#[A.RandomSizedBBoxSafeCrop(*output_shape, p=1), A.ColorJitter(p=1)],
                              
#                               output_shape=output_shape,
#                               compact=True,

#                               image2annotation=partial(img_2_annotation, ann_folder=ann_folder),

#                               target_format='albumentations',
#                               current_format='pascal_voc',
#                               background_label='background', 
#                               )
    
#     import matplotlib.pyplot as plt
#     import cv2 as cv    
#     import random

#     n = len(ds)

#     indices = random.sample(range(n), 10)
    
#     for i in indices:
#         x, y = ds[i]
        
#         obj_indicator, bbox, c = y[0].item(), y[1:5].tolist(), y[5:].tolist()

#         x = (np.moveaxis(x.numpy(), 0,-1) * 255).astype(np.uint8)
#         x = np.ascontiguousarray(x)

#         pascal_voc_bbox =au.convert_bbox_annotation(annotation=bbox, current_format=au.ALBUMENTATIONS, target_format=au.PASCAL_VOC, img_shape=x.shape[:2])

#         if obj_indicator:
#             xmin, ymin, xmax, ymax = pascal_voc_bbox
#             cv.rectangle(x, (xmin, ymin), (xmax, ymax), (0, 255,0), 1)    

#         plt.title(f'class : {c}')
#         plt.imshow(x)
#         plt.show()

#         # b_img = convert_img_2_background(img, bbox)
#         # plt.imshow(b_img)
#         # plt.show()



if __name__ == '__main__':
    train_data = os.path.join(SCRIPT_DIR, 'data', 'plants_ds_train')
    val_data = os.path.join(SCRIPT_DIR, 'data', 'plants_ds_val')

    ann_folder = os.path.join(SCRIPT_DIR, 'data', 'plants_ds_annotations')
    output_shape = (227, 227)

    model = AlexnetObjectLocalization(input_shape=(3, ) + output_shape, 
                                      num_classes=3, 
                                      num_fc_layers=3, 
                                      freeze=False)
    
    # print(model.head)

    model_ckpnt = run_pipeline(model=model, 
                train_data_folder=train_data,
                val_data_folder=val_data,
                annotation_folder=ann_folder,
                output_shape=output_shape,
                num_epochs=20,
                batch_size=64,
                learning_rates=0.01,
                ckpnt_dir=os.path.join(SCRIPT_DIR, 'logs'),
                use_wandb=False
                )
    
    # model_ckpnt = os.path.join(SCRIPT_DIR, 'logs', 'ckpnt_val_loss_0.5625_epoch_8.pt')

    val_ds = ObjectLocalizationDs(root_dir=val_data, 
                                    img_augs=[],

                                    output_shape=output_shape,
                                    compact=True,

                                    image2annotation=partial(img_2_annotation, ann_folder=ann_folder),

                                    target_format=au.ALBUMENTATIONS,
                                    current_format=au.PASCAL_VOC,
                                    background_label='background', 
                                    )
    


    from mypt.data.dataloaders.standard_dataloaders import initialize_val_dataloader
    from tqdm import tqdm
    import torch

    model.load_state_dict(torch.load(model_ckpnt)['model_state_dict'])

    device=pu.get_default_device()
    model = model.to(device)
    model.eval()

    val_dl =initialize_val_dataloader(val_ds, seed=0, batch_size=32, num_workers=0)

    model_predictions = None
    for batch_index, (x, y) in tqdm(enumerate(val_dl)):
        x = x.to(device)

        with torch.no_grad():
            pred = model.forward(x).cpu().numpy()
            # convert to 
            if model_predictions is None:
                model_predictions = pred
            else:
                model_predictions = np.concatenate([model_predictions, pred], axis=0)                    
            
    import matplotlib.pyplot as plt
    import cv2 as cv

    # iterate again through the dataloader
    for batch_index, (batch, _) in enumerate(val_dl):
        for item_index, x in enumerate(batch):
            x = (np.moveaxis(x.numpy(), 0,-1) * 255).astype(np.uint8)
            x = np.ascontiguousarray(x)
            
            p = model_predictions[batch_index * len(batch) + item_index, :]
            obj_indicator, bbox, c = int(p[0].item() > 0), p[1:5].tolist(), np.argmax(p[5:]).item()
            pascal_voc_bbox =au.convert_bbox_annotation(annotation=bbox, current_format=au.ALBUMENTATIONS, target_format=au.PASCAL_VOC, img_shape=x.shape[:2])

            if obj_indicator:
                xmin, ymin, xmax, ymax = pascal_voc_bbox
                cv.rectangle(x, (xmin, ymin), (xmax, ymax), (0, 255,0), 1)    

            plt.title(f'There is an object: {obj_indicator}, class : {val_ds.cls_index_2_cls[c]}')
            plt.imshow(x)
            plt.show()


            