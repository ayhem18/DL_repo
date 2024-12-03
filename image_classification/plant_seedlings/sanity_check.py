import wandb, os, shutil

# we are assuming the file is in the same directory as the 'train.py' script
import train
import src.utilities.directories_and_files as dirf

from pathlib import Path
from typing import Union
from torch import nn

from torch.optim.sgd import SGD
from torchvision.datasets import ImageFolder

from src.utilities.pytorch_utilities import seed_everything, get_default_device
from src.data.dataloaders.standard_dataloaders import initialize_train_dataloader
from src.schedulers.annealing_lr import AnnealingLR


SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
DATA_FOLDER = os.path.join(SCRIPT_DIR, 'data')

def get_sanity_check_dir():
    train_dir = os.path.join(DATA_FOLDER, 'train')
    dir_path = os.path.join(DATA_FOLDER, 'sanity_check_dir')

    if not os.path.exists(dir_path):
        # create the val directory
        os.makedirs(dir_path)
        dirf.dataset_portion(directory_with_classes=train_dir, 
                             destination_directory=dir_path, 
                             portion=0.1, # use only a 10 % of the training dataset
                             copy=True)
    return dir_path

def perform_sanity_check(
                seed: int, 
                log_dir: Union[str, Path], 
                epoch_per_sweep: int, 
                sweep_count: int):
    
    # create the file if needed
    log_dir = dirf.process_path(log_dir,
                                file_ok=False,
                                condition=lambda p: len(os.listdir(p)) == 0,
                                error_message=f'The directory must be empty to make sure not to override previous sweeps')
    
    sweep_config = {
        "method": "grid",
        "metric": {"goal": "minimize", "name": "train_loss"},
        "parameters": {
            "num_fe_blocks": {"values": [1, 2, 3, 4]}, 
            "num_classification_layers": {"values": [1, 2, 3]}
        },
    }

    # get a small portion of the training dataset
    sanity_check_dir = get_sanity_check_dir()
    device = get_default_device()

    # create the sweep
    sweep_id = wandb.sweep(sweep=sweep_config, 
                            project=train.WANDB_PROJECT_NAME)

    # define the function that will run at each sweep
    def sweep_function():
        # set the seed 
        seed_everything(seed)
        # first define the sub log dir
        log_subdir = dirf.process_path(os.path.join(log_dir, f'log_{len(os.listdir(log_dir)) + 1}'), 
                                       file_ok=False)

        # initialize wanbd 
        wandb.init(project=train.WANDB_PROJECT_NAME,
                    config=sweep_config,
                    name=os.path.basename(log_dir))

        # extract the values from the wandb configuration
        num_fe_blocks = wandb.config.num_fe_blocks
        num_classification_layers = wandb.config.num_classification_layers

        # initialize the model
        model, resnet_image_transformation = train.set_model(num_classification_layers=num_classification_layers, 
                          num_fe_blocks=num_fe_blocks, 
                          num_classes=len(os.listdir(sanity_check_dir)))

        ds_train = ImageFolder(root=sanity_check_dir, transform=resnet_image_transformation)

        dl_train = initialize_train_dataloader(dataset_object=ds_train, 
                                    seed=seed, 
                                    batch_size=128,
                                    num_workers=3)

        # set the optimizer
        optimizer = SGD(model.parameters(), 
                        lr=10 ** -3, 
                        momentum=0.99)

        lr_scheduler = AnnealingLR(optimizer=optimizer, 
                                num_epochs=epoch_per_sweep, 
                                alpha=10, 
                                beta=0.75, 
                                verbose=False)

        for epoch_index in range(1, epoch_per_sweep + 1):
            train_epoch_loss, train_epoch_acc = train.train_per_epoch(model=model, 
                                                                dataloader=dl_train,
                                                                loss_function=nn.CrossEntropyLoss(),
                                                                optimizer=optimizer, 
                                                                scheduler=lr_scheduler,
                                                                epoch_index=epoch_index,
                                                                log_per_batch=15,
                                                                device=device)
        

        # create the file name
        log_file_name = f'sanity_check_final_train_loss_{round(train_epoch_loss, 5)}_.pt'
        
        log_metrics = {"final_train_loss": train_epoch_loss, 
                    "final_epoch_acc": train_epoch_acc}
        
        train.log_model(model_path=os.path.join(log_subdir, log_file_name), 
                model=model,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                metrics=log_metrics)
        
        wandb.finish()

    # try:
        # run the sweep
    wandb.agent(sweep_id,  
            function=sweep_function,
            count=sweep_count)

if __name__ == '__main__':
    perform_sanity_check(seed=69, 
                         log_dir=os.path.join(SCRIPT_DIR, 'sanity_check_sweep'), 
                         epoch_per_sweep=10,
                         sweep_count=10)
