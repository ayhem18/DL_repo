"""
This script
"""
import os, numpy as np, torch
import matplotlib.pyplot as plt

from pathlib import Path

from mypt.data.dataloaders.standard_dataloaders import initialize_train_dataloader
from mypt.data.datasets.parallel_augmentation.parallel_aug_ds_wrapper import Food101Wrapper, ImagenetterWrapper

from model_train_pytorch._set_ds import _DEFAULT_DATA_AUGS, _UNIFORM_DATA_AUGS


SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))

current = SCRIPT_DIR
while 'data' not in os.listdir(current):    
    current = Path(current).parent

DATA_FOLDER = os.path.join(current, 'data')


def check_data_aug(data_folder, dataset: str):
    
    if dataset not in ['food101', 'imagenette']:
        raise NotImplementedError()
    
    if dataset == 'food101':
        train_ds = Food101Wrapper(root_dir=data_folder,
                                output_shape=(200, 200), 
                                augs_per_sample=2, 
                                sampled_data_augs=_DEFAULT_DATA_AUGS, 
                                uniform_augs_after=_UNIFORM_DATA_AUGS,
                                uniform_augs_before=[],
                                samples_per_cls=None)

    else:
        train_ds = ImagenetterWrapper(root_dir=data_folder,
                                  output_shape=(200, 200), 
                                  augs_per_sample=2, 
                                  sampled_data_augs=_DEFAULT_DATA_AUGS, 
                                  uniform_augs_after=_UNIFORM_DATA_AUGS, 
                                  uniform_augs_before=[],
                                  samples_per_cls=None)
    

    # create a dataloader
    dl = initialize_train_dataloader(train_ds, seed=0, batch_size=10, num_workers=1)

    # for i, (x1, x2) in enumerate(dl):
    #     x = torch.cat([x1, x2])

    #     n = len(x1)

    #     for j in range(n):
    #         fig = plt.figure() 
    #         p1, p2 = x[j].squeeze(), x[(j + n) % (2 * n)].squeeze()
    #         p1, p2 = np.moveaxis(p1.numpy(), 0,-1), np.moveaxis(p2.numpy(), 0, -1)

    #         fig.add_subplot(1, 2, 1) 
    #         plt.imshow(p1) 
    #         plt.title("augmented image 1") 

    #         fig.add_subplot(1, 2, 2) 
    #         plt.imshow(p2) 
    #         plt.title("augmented image 2") 

    #         plt.show()

    #     break


    for i in range(25):
        im = train_ds._ds[i][0]
        im = np.array(im)

        x1, x2 = train_ds[i]

        x1, x2 = x1.numpy(), x2.numpy()

        assert x1.shape == (3, 200, 200) and x2.shape == (3, 200, 200), "Shapes are not as expected"

        # transpose the image 
        x1, x2 = np.moveaxis(x1, 0,-1), np.moveaxis(x2, 0, -1)
        fig = plt.figure() 

        fig.add_subplot(1, 3, 1) 
        plt.imshow(im) 
        plt.title("original image") 

        fig.add_subplot(1, 3, 2) 
        plt.imshow(x1) 
        plt.title("augmented image 1") 

        fig.add_subplot(1, 3, 3) 
        plt.imshow(x2) 
        plt.title("augmented image 2") 

        plt.show()





if __name__ == '__main__':
    dataset = 'food101'
    # dataset = 'imagenette'
    train_data_folder = os.path.join(DATA_FOLDER, dataset, 'train')
    check_data_aug(train_data_folder, dataset)
