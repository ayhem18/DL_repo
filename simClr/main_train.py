"""
This script contains the implementation of the main training process 
"""

import os

from mypt.code_utilities import directories_and_files as dirf, pytorch_utilities as pu
from model_train_pl.train import ResnetSimClrWrapper, train_simClr, train_simClr_single_round
from model_train_pl.constants import TRACK_PROJECT_NAME
from model_train_pl.set_ds import _DEFAULT_DATA_AUGS

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))


def train_main():
    dataset_name = 'imagenette'
    train_data_folder = os.path.join(SCRIPT_DIR, 'data', dataset_name, 'train')
    val_data_folder = os.path.join(SCRIPT_DIR, 'data', dataset_name, 'val')
    
    ckpnt_dir_parent = dirf.process_path(os.path.join(SCRIPT_DIR, 'logs', 'train_logs'), 
                                         dir_ok=True, 
                                         file_ok=False)

    # delete folders that either empty of have only empty sub-folders...
    dirf.clear_directory(ckpnt_dir_parent, 
                         condition=lambda x: os.path.isdir(x) and 
                         len([y for y in os.listdir(x) 
                              if not (os.path.isdir(os.path.join(x, y)) and len(os.listdir(os.path.join(x, y))) == 0)
                              ]
                              ) 
                         == 0)
    
    n = len([x for x in os.listdir(ckpnt_dir_parent) if dataset_name in x])

    run_name = f'{TRACK_PROJECT_NAME}_{dataset_name}_iteration_{n + 1}'
    
    # train_simClr_single_round(
    #             train_data_folder=train_data_folder,
    #             val_data_folder=val_data_folder,
    #             dataset=dataset_name,
    #             num_sampled_augs=2,
    #             log_dir=os.path.join(ckpnt_dir_parent, f'{dataset_name}_iteration_{n + 1}'),
    #             num_epochs=20,
    #             samples_weights=None,
    #             continue_last_task=False, 
    #             train_batch_size=400,
    #             val_batch_size=1024, # the larger the validation size the better...  
    #             seed=0, 
    #             use_logging=False,
    #             num_warmup_epochs=5,
    #             val_per_epoch=3,
    #             num_train_samples_per_cls=None,
    #             num_val_samples_per_cls=None,
    #             run_name=run_name,     
    #             debug_augmentations=_DEFAULT_DATA_AUGS                   
    #             )

    train_simClr(
                train_data_folder=train_data_folder,
                val_data_folder=val_data_folder,
                dataset=dataset_name,
                log_dir=os.path.join(ckpnt_dir_parent, f'{dataset_name}_iteration_{n + 1}'),
                num_epochs=20, 
                train_batch_size=400,
                val_batch_size=1024, # the larger the validation size the better...  
                seed=0, 
                use_logging=True,
                num_warmup_epochs=5,
                val_per_epoch=3,
                num_train_samples_per_cls=None,
                num_val_samples_per_cls=None,
                run_name=run_name,     
                debug_augmentations=_DEFAULT_DATA_AUGS                   
                )

if __name__ == '__main__':
    train_main()   


    # wrapper = ResnetSimClrWrapper(
    #                               # model parameters
    #                               input_shape=(3,) + (200, 200), 
    #                               output_dim=256,
    #                               num_fc_layers=4,

    #                               #logging parameters
    #                               logger=None,
    #                               log_per_batch=3,
    #                               val_per_epoch=3,
    #                               # loss parameters 
    #                               temperature=0.5,
    #                               debug_loss=True,

    #                               # learning 
    #                               lrs=0.6,
    #                               num_epochs=3,
    #                               num_warmup_epochs=0,

    #                               debug_embds_vs_trans=True, 
    #                               debug_augmentations=[],

    #                               # more model parameters
    #                               dropout=0.2,
    #                               architecture=50,
    #                               freeze=False, 
    #                               save_hps=False, # no need to save the hyperparameters, this way the model will only load the weights
    #                               )    

    # initial_checkpoint = os.path.join(SCRIPT_DIR, 'logs', 'train_logs', 'food101_iteration_4', 'round_1', 'val-epoch=05-val_epoch_loss=4.942599.ckpt')

    # w = ResnetSimClrWrapper.load_from_checkpoint(initial_checkpoint)

    # print(w.train_epoch_index)
    # print(w.val_epoch_index)

    # import torch

    # wrapper.model.load_state_dict(torch.load(initial_checkpoint)['state_dict'])
    # # let's see if we can load the state of optimizers and schedulers as well
    # wrapper.optimizers().optimizer.load_state_dict(torch.load(initial_checkpoint)['optimizer'])
    # wrapper.lr_schedulers().scheduler.load_state_dict(torch.load(initial_checkpoint)['lr_scheduler'])

