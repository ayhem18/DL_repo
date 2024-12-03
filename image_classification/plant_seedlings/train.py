"""
This script mainly contains the code to train an initial model for the classification task 
"""

import os, torch, random, wandb
import torchvision.transforms as tr
import src.utilities.directories_and_files as dirf

from torch import nn
from torch.optim.sgd import SGD
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Tuple, Union, Dict, List
from pathlib import Path
    
from src.backbones.resnetFeatureExtractor import ResNetFeatureExtractor
from src.classification.classification_head import ExponentialClassifier
from src.utilities.pytorch_utilities import seed_everything, get_default_device, get_module_device
from src.data.dataloaders.standard_dataloaders import initialize_train_dataloader, initialize_val_dataloader
from src.schedulers.annealing_lr import AnnealingLR
from src.dimensions_analysis.dimension_analyser import DimensionsAnalyser

# set the project name
WANDB_PROJECT_NAME = 'PLANTS_CLASSIFICAITON'

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
DATA_FOLDER = os.path.join(SCRIPT_DIR, 'data')

def set_data():
    train_dir = os.path.join(DATA_FOLDER, 'train')
    val_dir = os.path.join(DATA_FOLDER, 'val')
    if not os.path.exists(val_dir):
        # create the val directory
        os.makedirs(val_dir)
        dirf.dataset_portion(directory_with_classes=train_dir, 
                             destination_directory=val_dir, 
                             portion=0.15,
                             copy=False)
    return train_dir, val_dir


def train_per_batch(model: nn.Sequential, 
                    batch_x: torch.Tensor,
                    batch_y: torch.Tensor,
                    loss_function: nn.Module,
                    optimizer: torch.optim.Optimizer,
                    ) -> Tuple[float, float]:
    
    device = get_module_device(model)
    # make sure to move the data to the 'device'
    x, y = batch_x.to(device=device), batch_y.to(device=device)    
    # set the optimizer's gradients to zero
    optimizer.zero_grad()
    
    # forward pass through the model
    model_output = model.forward(x)
    
    # metrics: loss + accuracy
    batch_loss_obj = loss_function.forward(model_output, y)

    batch_loss = batch_loss_obj.item()
    batch_accuracy = torch.mean((torch.argmax(model_output, dim=-1) == y).to(torch.float32))

    # perform the backpropagation
    batch_loss_obj.backward()

    # perform update the weights
    optimizer.step()

    return batch_loss, batch_accuracy

def train_per_epoch(model: nn.Sequential, 
                dataloader: DataLoader,
                loss_function: nn.Module,
                optimizer: torch.optim.Optimizer, 
                scheduler: torch.optim.lr_scheduler.LRScheduler, 
                epoch_index: int,
                device: str,
                log_per_batch: int) -> Tuple[float, float]:
    
    # set the model to the train mode
    model = model.to(device=device)
    model.train()

    # define the loss function 
    loss_function = nn.CrossEntropyLoss()

    # define a function to save the average loss per epoch
    epoch_train_loss, epoch_train_accuracy = 0, 0

    for batch_index, (x, y) in tqdm(enumerate(dataloader), desc=f'training batch at epoch {epoch_index }'): 
        batch_train_loss, batch_train_accuracy = train_per_batch(model=model, 
                                                batch_x=x, 
                                                batch_y=y, 
                                                loss_function=loss_function,
                                                optimizer=optimizer)
        
        # log the batch loss depending on the batch index
        if batch_index % log_per_batch:
            wandb.log({"epoch": epoch_index, "train_loss": batch_train_loss, "train_accuracy": batch_train_accuracy})

        # include the batch loss into the overall epoch loss (same for accuracy)
        epoch_train_loss += epoch_train_loss
        epoch_train_accuracy += epoch_train_accuracy

    param_index = random.randint(0, len(optimizer.param_groups))

    lr_before = optimizer.param_groups[param_index]['lr']
    # make sure to call the scheduler to update the learning rate
    scheduler.step()
    lr_after = optimizer.param_groups[param_index]['lr']

    if lr_before <= lr_after: 
        raise ValueError(f"The learning rate is not descreasing. lr before: {lr_before}, lr after: {lr_after}")

    # make sure to average the metric
    epoch_train_loss = epoch_train_loss / len(dataloader)
    epoch_train_accuracy = epoch_train_accuracy / len(dataloader)

    # log the metrics
    wandb.log({"epoch": epoch_index, 
               "train_loss": epoch_train_loss, 
               "train_accuracy": epoch_train_accuracy})

    return epoch_train_loss, epoch_train_accuracy

def validation_per_batch(model: nn.Module, 
                     batch_x: torch.Tensor,
                     batch_y: torch.Tensor,
                     loss_function: nn.Module,
                     ):
    
    device = get_module_device(device)
    # the first idea is to set the model 
    x, y = batch_x.to(device), batch_y.to(device)
    with torch.no_grad():
        # forward pass
        y_logits = model.forward(x)
        loss_object = loss_function.forward(input=y_logits, target=y)

        # metrics: 
        batch_val_loss = loss_object.item()
        batch_val_accuracy = torch.mean((torch.argmax(y_logits, dim=-1) == y).to(torch.float32))

    return batch_val_loss, batch_val_accuracy

def validation_per_epoch(model: nn.Module,
                         dataloader: DataLoader,
                         epoch_index: int,
                         device: str, 
                        log_per_batch: int) -> Tuple[float, float]:

    epoch_val_loss = 0
    epoch_val_accuracy = 0
    loss_function = nn.CrossEntropyLoss()        

    model = model.to(device=device)
    model.eval()
    for batch_index, (x, y) in tqdm(enumerate(dataloader), desc=f'validation batch for epoch {epoch_index}'):
        batch_loss, batch_accuracy = validation_per_batch(model=model, 
                                                          batch_x=x, 
                                                          batch_y=y, 
                                                          loss_function=loss_function)

        if batch_index % log_per_batch == 0:
            wandb.log({"epoch": epoch_index, "val_loss": batch_loss, "val_accuracy": batch_accuracy})

        # add the batch loss and accuracy to the epoch loss / accuracy        
        epoch_val_loss += batch_loss
        epoch_val_accuracy += batch_accuracy

    wandb.log({"epoch": epoch_index, "val_loss": epoch_val_loss / len(dataloader), "val_accuracy": epoch_val_accuracy / len(dataloader)})


def log_model(
              model_path: Union[str, Path],
              model: nn.Module, 
              optimizer: torch.optim.Optimizer,
              lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
              metrics: Dict[str, Union[List[float], float]]  
              ):
    # process the path 
    model_path = dirf.process_path(model_path, 
                                   dir_ok=False, 
                                   file_ok=True, 
                                   condition=lambda p: os.path.basename(p).endswith('.pt'), 
                                   error_message=f'the checkpoint file must end with a .pt extension. Found: {os.path.basename(model_path)}')
    
    if not isinstance(metrics, Dict):
        raise ValueError(f"The metrics objects is expected to be a dictionary. Found: {metrics}") 
    
    checkpoint_dict = metrics.copy()
    checkpoint_dict['model_state_dict'] = model.state_dict()
    checkpoint_dict['optimizer_state_dict'] = optimizer.state_dict()    
    checkpoint_dict['lr_scheduler_state_dict'] = lr_scheduler.state_dict()
    
    torch.save(checkpoint_dict, model_path)


def model_train(
                   model: nn.Sequential, 
                   image_transform: tr, 
                   num_epochs: int, 
                   dir_train: Union[str, Path],
                   dir_val: Union[str, Path],
                   val_every_epoch: int, 
                   log_every_epoch: int, 
                   log_dir: Union[str, Path],
                   seed: int
                   ):
    seed_everything(seed=seed)
    device = get_default_device() 
    
    dir_train = dirf.process_path(dir_train, 
                        dir_ok=True, 
                        file_ok=False, 
                        condition=dirf.image_dataset_directory, 
                        error_message=dirf.image_dataset_directory_error)

    dir_val = dirf.process_path(dir_val, 
                        dir_ok=True, 
                        file_ok=False, 
                        condition=dirf.image_dataset_directory, 
                        error_message=dirf.image_dataset_directory_error)

    log_dir = dirf.process_path(log_dir, file_ok=False, 
                                condition=lambda p : len(os.listdir(p)) == 0, 
                                error_message=f'Make sure the ')


    train_ds = ImageFolder(root=dir_train, transform=image_transform) 
    val_ds = ImageFolder(root=dir_val, transform=image_transform) 

    dl_train = initialize_train_dataloader(dataset_object=train_ds, 
                                   seed=seed, 
                                   batch_size=128,
                                   num_workers=3)
    
    dl_val = initialize_val_dataloader(dataset_object=val_ds, 
                                       seed=seed, 
                                       batch_size=120, 
                                       num_workers=0)

    # set the optimizer
    optimizer = SGD(model.parameters(), 
                    lr=10 ** -3, 
                    momentum=0.99)

    lr_scheduler = AnnealingLR(optimizer=optimizer, 
                               num_epochs=num_epochs, 
                               alpha=10, 
                               beta=0.75, 
                               verbose=False)

    train_losses = []
    train_accuracies = []

    val_losses = []
    val_accuracies = []

    for epoch_index in range(1, num_epochs + 1):
        train_epoch_loss, train_epoch_acc = train_per_epoch(model=model, 
                                                            dataloader=dl_train,
                                                            optimizer=optimizer, 
                                                            lr_scheduler=lr_scheduler,
                                                            epoch_index=epoch_index,
                                                            log_per_batch=15,
                                                            device=device)

        train_losses.append(train_epoch_loss)
        train_accuracies.append(train_epoch_acc)

        # validation pass
        if epoch_index % val_every_epoch == 0:
            epoch_train_loss, epoch_train_acc = validation_per_epoch(model=model,
                                dataloader=dl_val,
                                epoch_index=epoch_index,
                                device=device, 
                                log_per_batch=15) 

            val_losses.append(train_epoch_loss)
            val_accuracies.append(train_epoch_acc)

        # logging the model
        if epoch_index % log_every_epoch == 0:
            epoch_val_loss, epoch_val_acc = validation_per_epoch(model=model,
                                dataloader=dl_val,
                                epoch_index=epoch_index,
                                device=device, 
                                log_per_batch=15) 

            # create the file name
            log_file_name = f'epoch_{epoch_index}_val_loss_{round(epoch_val_loss, 5)}_.pt'
            
            log_metrics = {"train_loss": epoch_train_loss, 
                           "train_acc": epoch_train_acc, 
                           "val_loss": epoch_val_loss, 
                           "val_acc": epoch_val_acc}
            
            log_model(model_path=os.path.join(log_dir, log_file_name), 
                      model=model,
                      optimizer=optimizer,
                      lr_scheduler=lr_scheduler,
                      metrics=log_metrics)


def set_model(num_classification_layers: int,
              num_fe_blocks: int,
              num_classes: int) -> Tuple[nn.Module, tr.Compose]:
    
    fe = ResNetFeatureExtractor(num_layers=num_fe_blocks, 
                                add_fc=False, 
                                freeze=num_fe_blocks - 1,
                                freeze_layers=True,
                                architecture=50)

    # analyze the model
    _, in_features =  DimensionsAnalyser(nn.Sequential(fe, nn.Flatten())).analyse_dimensions((5, ) + (3, 224, 224))

    ch = ExponentialClassifier(num_classes=num_classes, 
                               in_features=in_features,
                               num_layers=num_classification_layers
                               )

    return nn.Sequential(fe, nn.Flatten(), ch), fe.transform

# def train_fast(seed: int = 69):
#     # let's set 
#     set_everything(seed=seed)

#     device = get_default_device() 
#     dir_train, dir_val = set_data()

#     # let's set some model
#     fe = ResNetFeatureExtractor(num_layers=-1, 
#                                 add_fc=False, 
#                                 freeze_layers=True, 
#                                 freeze=3) # do not freeze the last layer
    
#     # let's say we have 2 layer for classification
#     ch = ExponentialClassifier(in_features=2048, num_classes=12, num_layers=2)

#     model = nn.Sequential(fe, nn.Flatten(), ch)

#     # create the dataset and dataloaders 
#     train_ds = ImageFolder(root=dir_train, transform=fe.transform) 
#     val_ds = ImageFolder(root=dir_val, transform=fe.transform) 

#     dl_train = initialize_train_dataloader(dataset_object=train_ds, 
#                                    seed=seed, 
#                                    batch_size=128,
#                                    num_workers=3)
    
#     dl_val = initialize_val_dataloader(dataset_object=val_ds, 
#                                        seed=seed, 
#                                        batch_size=120, 
#                                        num_workers=0)

#     loss_function = nn.CrossEntropyLoss()
#     optimizer = Adam(model.parameters(), lr=0.001)  
#     # set the model to 'device'
#     model = model.to(device=device)

#     # before training let's set everything correctly
#     wandb.init(project=WANDB_PROJECT_NAME)

#     # time to track the gradients of the model
#     wandb.watch(models=model, 
#                 log_freq=20)

#     for epoch_index in tqdm(range(5), desc='training epochs'):
#         epoch_train_loss = 0 
#         epoch_train_accuracy = 0
        
#         # set the model to 'train' every time
#         model.train()
#         for batch_index, (x, y) in tqdm(enumerate(dl_train), desc=f'training batch at epoch {epoch_index + 1}'): 
#             # set the optimizer
#             optimizer.zero_grad()
            
#             # the first idea is to set the model 
#             x, y = x.to(device), y.to(device)
#             # forward pass
#             y_logits = model.forward(x)
#             loss_object = loss_function.forward(input=y_logits, target=y)

#             # metrics: 
#             epoch_train_loss += loss_object.item()
#             batch_accuracy = torch.mean((torch.argmax(y_logits, dim=-1) == y).to(torch.float32))
#             epoch_train_accuracy += batch_accuracy

#             # backward
#             loss_object.backward()
#             optimizer.step()

#             # log the loss of the batch 
#             if batch_index % 20 == 0:
#                 wandb.log({"epoch": epoch_index, "train_loss": loss_object.item()}) 
#                 wandb.log({"epoch": epoch_index, "train_accuracy": batch_accuracy}) 

#         # log the results for the epoch overall
#         wandb.log({"epoch": epoch_index, "train_loss": epoch_train_loss / len(dl_train)})
#         wandb.log({"epoch": epoch_index, "train_accuracy": epoch_train_accuracy / len(dl_train)})
        
#         epoch_val_loss = 0
#         epoch_val_accuracy = 0

#         # eval: model
#         model.eval()
#         with torch.no_grad():
#             for x, y in tqdm(dl_val, desc=f'validation batch for epoch {epoch_index + 1}'):
#                 # the first idea is to set the model 
#                 x, y = x.to(device), y.to(device)
#                 # forward pass
#                 y_logits = model.forward(x)
#                 loss_object = loss_function.forward(input=y_logits, target=y)

#                 # metrics: 
#                 epoch_val_loss += loss_object.item()
#                 epoch_val_accuracy += torch.mean((torch.argmax(y_logits, dim=-1) == y).to(torch.float32))

#         wandb.log({"epoch": epoch_index, "val_loss": epoch_val_loss / len(dl_val)})
#         wandb.log({"epoch": epoch_index, "val_accuracy": epoch_val_accuracy / len(dl_val)})

#     wandb.finish()

if __name__ == '__main__':
    pass
