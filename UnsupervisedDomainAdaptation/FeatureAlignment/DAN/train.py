import torch, wandb, os
import torchvision.transforms as tr

import src.utilities.pytorch_utilities as pu

from pathlib import Path
from torch import nn
from typing import Tuple, List, Union
from torch.utils.data import DataLoader
from copy import deepcopy
from pathlib import Path
from torch.optim.sgd import SGD
from torch.optim.adam import Adam
from tqdm import tqdm

from transferable_alexnet import TransferAlexNet
from src.distances.MMD import GaussianMMD

from src.schedulers.annealing_lr import AnnealingLR
from src.distances.MMD import GaussianMMD
from data import get_dataloader
from transferable_alexnet import TransferAlexNet


home = os.path.dirname(os.path.realpath(__file__))
current = home
while 'data' not in os.listdir(current):
    current = Path(current).parent
DATA_FOLDER = os.path.join(current, 'data')

def calculate_loss(source_logits: torch.Tensor, 
                   source_labels: torch.Tensor,
                   source_features: List[torch.Tensor], 
                   target_features: List[torch.Tensor], 
                   loss_coefficient: float,
                   reduction: str = 'mean', 
                   sigma: float = 0.5) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
    
    if len(source_features) != len(target_features):
        raise ValueError(f"Please make sure the number of features is the same acoss target and source domains")

    # first part is the cross entropy loss as usual
    cls_loss = nn.CrossEntropyLoss(reduction=reduction).forward(source_logits, source_labels)
    
    # this is the main object used for backpropagation
    domain_confusion_loss = None    

    # a list to store and track the different similarities along with training
    distribution_losses = []

    for fs, ft in zip(source_features, target_features):
        loss_obj = GaussianMMD(sigma=sigma).forward(x=fs, y=ft)

        # make sure to detach and pass to 'cpu'
        distribution_losses.append(loss_obj.detach().cpu())

        if domain_confusion_loss is None:
            domain_confusion_loss = loss_obj
        else: 
            domain_confusion_loss += loss_obj

    return cls_loss + loss_coefficient * domain_confusion_loss, cls_loss, distribution_losses

def train_epoch(
                model: TransferAlexNet,
                source_train_dir: DataLoader[Tuple[torch.Tensor, torch.Tensor]], 
                target_dir: DataLoader[torch.Tensor], 
                image_transform: tr,
                optimizer: torch.optim.Optimizer,
                lr_scheduler: torch.optim.lr_scheduler.LRScheduler, 
                loss_coefficient: float,
                device: str, 
                seed: int, 
                epoch_index: int,
                batch_size: int, 
                sigma: float) -> List:

    # make sure to set the model to the 'train' mode
    model = model.to(device)
    model.train()

    dl_source_train = get_dataloader(root=source_train_dir, 
                                     image_transform=image_transform,
                                     batch_size=batch_size, 
                                     seed=seed, 
                                     num_workers=0)

    dl_target = get_dataloader(root=target_dir, 
                                image_transform=image_transform,
                                batch_size=batch_size, 
                                seed=seed, 
                                num_workers=0)
    # convert them to iterator
    dl_source_train = iter(dl_source_train)
    dl_target = iter(dl_target)


    source_batch = next(dl_source_train, None)
    target_batch = next(dl_target, None)
    source_over, target_over = source_batch is None , target_batch is None

    epoch_loss = 0
    epoch_cls_loss = 0
    epoch_acc = 0
    epoch_feats_losses = []
    batch_count = 0

    # create a progress bar for the given batch
    pbar = tqdm(desc=f'epoch: {epoch_index}')
    
    while not source_over:
        # make sure to update the batch_count
        batch_count += 1 

        # make sure to reset the gradients to zero
        optimizer.zero_grad()
        
        xs, ys = source_batch
        xs, ys = xs.to(device), ys.to(device)
        # forward pass: source
        model_output_source = model.forward(xs)
        source_features, logits = model_output_source[:-1], model_output_source[-1]

        # forward pass: target
        xt, _ = target_batch
        xt = xt.to(device)
        model_output_target = model.forward(xt)
        target_features = model_output_target[:-1]
        
        # the loss consists of
        final_loss, cls_loss, feature_losses = calculate_loss(source_logits=logits,
                                    source_labels=ys, 
                                    source_features=source_features,
                                    target_features=target_features,
                                    loss_coefficient=loss_coefficient, 
                                    sigma=sigma,
                                    )

        epoch_loss += final_loss.item()
        epoch_cls_loss += cls_loss.item()
        # compute accuracy
        epoch_acc += torch.mean((torch.argmax(logits, dim=1) == ys).to(torch.float32)).item()
        
        # update each current loss with its corresponding loss in the new batch
        if len(epoch_feats_losses) == 0:
            epoch_feats_losses = [fl.item() for fl in feature_losses]
        else:
            [c + n.item() for c, n in zip(epoch_feats_losses, feature_losses)]
        
        # backpropagation
        final_loss.backward()
        optimizer.step()

        # make sure to check the dataloaders
        source_batch = next(dl_source_train, None)
        target_batch = next(dl_target, None)
        source_over, target_over = source_batch is None, target_batch is None

        # if the target domain is already exhausted, then re-initialize it
        if target_over and not source_over: # (there is no point for re-initializing the target domain if the source domain is exhausted as well) 
            dl_target = get_dataloader(root=target_dir, 
                                       image_transform=image_transform, 
                                       batch_size=batch_size, 
                                       seed=seed+1, 
                                       num_workers=0) # choose a different seed to use different combinations from the target domain
            dl_target = iter(dl_target)
            target_batch = next(dl_target)

        # update the progress bar 
        pbar.update(1)

    pbar.close()

    # update the learning rate with the learning rate scheduler
    lr_scheduler.step()
    
    # divide by the number of batches
    epoch_loss /= batch_count
    epoch_cls_loss /= batch_count
    epoch_acc /= batch_count
    epoch_feats_losses = [l / batch_count for l in epoch_feats_losses]

    return epoch_loss, epoch_cls_loss, epoch_acc, epoch_feats_losses

def standard_validation_epoch(
                            model: TransferAlexNet,
                            device: str, 
                            seed: int,                              
                            val_dir: Union[str, Path] = None,
                            val_dataloader: DataLoader = None, 
                            image_transform: tr = None) -> float:
    if (val_dir is None) == (val_dataloader is None):
        raise ValueError(f"Please make sure to pass either 'val_dir' or 'val_dataloader' arguments. Found dir: {val_dir} and dataloader: {val_dataloader}")


    if val_dataloader is None:
        if image_transform is None:
            raise ValueError(f"if the validation directory is passed, make sure 'image_transform' is not set to None !!")
        val_dataloader = get_dataloader(root=val_dir, 
                                 image_transform=image_transform, 
                                 batch_size=256, 
                                 num_workers=2, 
                                 seed=seed)
    
    # set the model to the 'eval' model
    model.eval()

    loss_function = nn.CrossEntropyLoss()

    val_cls_loss = 0
    val_acc = 0

    with torch.no_grad():
        for x, y in val_dataloader:
            # map the input and the labels to the 'device'
            x, y = x.to(device), y.to(device)
            # get the model output
            model_output = model.forward(x)
            model_logits = model_output[-1]

            # let's see how it goes
            loss_obj = loss_function.forward(input=model_logits, target=y)    

            # calculate the loss
            val_cls_loss += loss_obj.item()
            val_acc += torch.mean((torch.argmax(model_logits, dim=1) == y).to(torch.float32)).item()

    return val_cls_loss / len(val_dataloader), val_acc / len(val_dataloader)
        
def train_model(source_train_dir: Union[str, Path],
                source_val_dir: Union[str, Path],
                target_dir: Union[str, Path],
                num_epochs: int, 
                run_name: str,
                batch_size: int = 32, 
                seed: int = 69, 
                wandb_project_name: str = 'replicate_DAN'
                ):
    # set the seed
    pu.seed_everything(seed)
    
    # # before training let's set everything correctly
    # wandb.init(project=wandb_project_name, name=run_name)

    # initialize the model 
    model = TransferAlexNet(input_shape=(3, 224, 224), 
                            num_classes=31, 
                            alexnet_blocks='conv_block_adapool', 
                            alexnet_frozen_blocks=['conv1', 'conv2', 'conv3'], 
                            num_classification_layers=3)

    # set the device
    device = pu.get_default_device()
    # set the image transformation
    image_transformation = model.image_transformation
    
    # set the optimizer: the initial learning rate for the feature extractor will be 0.001 and the fully connected layers
    # will be assigned a 0.01 learning rate
    # optimizer = SGD(params=[{"params": model.fe.parameters(), "lr": 10 * -3}, 
    #                         {"params": model.ch.parameters(), "lr": 10 * -2}], 
    #                 momentum=0.9)

    optimizer = Adam(params=[{"params": model.fe.parameters(), "lr": 10  ** -3}, 
                             {"params": model.ch.parameters(), "lr": 10 ** -2}])

    lr_scheduler = AnnealingLR(optimizer=optimizer, num_epochs=num_epochs, alpha=10, beta=0.75)

    train_losses, train_cls_losses, train_accs, train_feats_losses = [], [], [], []
    val_losses,  val_accs = [], []

    for epoch_index in tqdm(range(1, num_epochs + 1),desc='model training'):
        # training epoch
        train, train_cls, train_acc, train_feats = train_epoch(model=model, 
                                                               source_train_dir=source_train_dir, 
                                                               target_dir=target_dir, 
                                                               image_transform=image_transformation,
                                                               optimizer=optimizer,
                                                               lr_scheduler=lr_scheduler, 
                                                               loss_coefficient=0.5, 
                                                               device=device, 
                                                               seed=seed,
                                                               epoch_index=epoch_index,
                                                               batch_size=batch_size,
                                                               sigma=0.25
                                                               )

        train_logging_dict = {"train_loss": train, 
                        "train_cls_loss": train_cls, 
                        "train_accuracy": train_acc}
        
        for i, f in enumerate(train_feats, start=1):
            train_logging_dict[f'domain_confusion_loss_{i}']= f
        
        print(train_logging_dict)
        # logging: 
        # wandb.log(logging_dict)


        # validation epoch
        val, val_acc = standard_validation_epoch(model=model, 
                                                 device=device, 
                                                 seed=seed, 
                                                 image_transform=image_transformation,
                                                 val_dir=source_val_dir)

        # logging: 
        val_logging_dict = {"val_cls_loss": val, 
                        "val_acc": val_acc}
        
        print("#" * 10)
        print(val_logging_dict)
        # wandb.log(logging_dict)

    # make sure the wandb agent exits after the training
    # wandb.finish()

if __name__ == '__main__':
    import random
    # get the data
    amazon_dir = os.path.join(DATA_FOLDER, 'office31', 'amazon', 'amazon')
    amazon_train = os.path.join(DATA_FOLDER, 'office31', 'amazon', 'amazon_splitted', 'train')
    amazon_val = os.path.join(DATA_FOLDER, 'office31', 'amazon', 'amazon_splitted', 'val')

    dslr_dir = os.path.join(DATA_FOLDER, 'office31', 'dslr', 'dslr')

    train_model(
                source_train_dir=amazon_train, 
                source_val_dir=amazon_val, 
                target_dir=dslr_dir, 
                num_epochs=3, 
                run_name=f'DAN_attempt_{random.randint(0, 10 ** 4)}', 
                seed=69)
