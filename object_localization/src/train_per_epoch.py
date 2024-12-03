import torch, wandb, math

from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Tuple, Union, Optional


from mypt.code_utilities import pytorch_utilities as pu
from mypt.models.object_detection.obj_localization import ObjectLocalizationModel
from mypt.losses.object_detection.object_localization import ObjectLocalizationLoss

def train_per_batch(model: ObjectLocalizationModel,
                    x: torch.Tensor,
                    y:torch.Tensor,
                    loss_function: ObjectLocalizationLoss,
                    optimizer: torch.optim.Optimizer,
                    optimizer_zero_grad:bool,
                    optimizer_step:bool,
                    device: str,
                    ) -> dict[str, float]:

    model = model.to(device=device)
    # make sure to stack the two batches into a single batch 
    x = x.to(device=device)
    y = y.to(device=device)

    if optimizer_zero_grad:
        # set the optimizer's gradients to zero
        optimizer.zero_grad()
    
    # forward pass through the model
    model_output = model.forward(x)

    batch_res = loss_function.forward(x=model_output, y=y, all=True)

    # extract the final loss object for backpropagation
    batch_loss_obj = batch_res[loss_function._loss_name]

    # perform the backpropagation
    batch_loss_obj.backward()

    if optimizer_step:
        # update the weights
        optimizer.step()

    return {loss_name: l.item() for loss_name, l in batch_res.items()}


def train_per_epoch(model: ObjectLocalizationModel, 
                dataloader: DataLoader,
                loss_function: ObjectLocalizationLoss,
                optimizer: torch.optim.Optimizer, 
                scheduler: Optional[torch.optim.lr_scheduler.LRScheduler], 
                epoch_index: int,
                device: str,
                log_per_batch: Union[int, float],
                accumulate_grads: int = 1,
                use_wandb:bool=True
                ) -> dict[str, float]:

    if isinstance(log_per_batch, float):
        num_batches = len(dataloader)
        log_per_batch = int(math.ceil(num_batches * log_per_batch))

    # set the model to the train mode
    model = model.to(device=device)
    model.train()

    # # define a function to save the average loss per epoch
    # epoch_train_loss = 0
    
    num_batches = len(dataloader)

    epoch_metrics = None

    for batch_index, (x, y) in tqdm(enumerate(dataloader, start=1), desc=f'training batch at epoch {epoch_index }'): 
        # to make the gradient accumulation work

        # 1. the optimizer.zero_grad() method should be called with batch indices such that batch_index % acc_grad == 1 (or no gradient accumulation at all (acc_grad == 1))

        # 2. the optimizer.step() should be called with batch_indices divisbly by acc_grad or the very last batch
 
        # the batch index should be 1-indexed not 0-indexed for the implementation (as described above) to work
        
        batch_res = train_per_batch(model=model,
                                    x=x,
                                    y=y, 
                                    loss_function=loss_function,
                                    optimizer=optimizer, 
                                    optimizer_zero_grad=((accumulate_grads == 1) or (batch_index % accumulate_grads) == 1),
                                    optimizer_step=((batch_index % accumulate_grads == 0) or (batch_index == num_batches)),
                                    device=device)
            
        
        # log the batch loss depending on the batch index
        if batch_index % log_per_batch == 0 and use_wandb:
            log_dict = {"epoch": epoch_index, "batch": batch_index}
            log_dict.update(batch_res)    
            wandb.log(log_dict)                

        if epoch_metrics is None:
            epoch_metrics = batch_res.copy()
        else:
            for loss_key, _ in epoch_metrics.items():
                epoch_metrics[loss_key] += batch_res[loss_key]
        
        # epoch_train_loss += batch_res[loss_function._loss_name] 

    # make sure to call the scheduler to update the learning rate
    if scheduler is not None:
        scheduler.step()

    # average the metrics across batches
    for loss_key, _ in epoch_metrics.items():
        epoch_metrics[loss_key] /= len(dataloader)

    # # make sure to average the metric
    # epoch_train_loss = epoch_train_loss / len(dataloader)

    # add the epoch to the metrics
    epoch_metrics['epoch'] = epoch_index

    # log the metrics
    if use_wandb:
        wandb.log(epoch_metrics)

    return epoch_metrics

def validation_per_batch(model: ObjectLocalizationModel, 
                        x: torch.Tensor,
                        y: torch.Tensor,
                        loss_function: ObjectLocalizationLoss,
                     ) -> dict[str, float]:

    device = pu.get_module_device(model)
    # make sure to stack the two batches into a single batch 
    x = x.to(device)
    y = y.to(device)
    
    with torch.no_grad():
        # forward pass through the model
        model_output = model.forward(x)

        batch_loss_res = loss_function.forward(x=model_output,y=y, all=True)

        # extract the values from the loss tensors
        batch_loss_res = {l_key: l.item() for l_key, l in batch_loss_res.items()}

    return batch_loss_res

def validation_per_epoch(model: ObjectLocalizationModel,
                        dataloader: DataLoader,
                        loss_function: nn.Module,
                        epoch_index: int,
                        device: str, 
                        log_per_batch: Union[float, int],
                        use_wandb:bool=True) -> dict[str, float]:

    if isinstance(log_per_batch, float):
        num_batches = len(dataloader)
        log_per_batch = int(math.ceil(num_batches * log_per_batch))


    model = model.to(device=device)
    model.eval()

    epoch_val_metrics = None

    for batch_index, (x, y) in tqdm(enumerate(dataloader, start=1), desc=f'validation batch for epoch {epoch_index}'):
        batch_val_res = validation_per_batch(model=model, 
                                              x=x,
                                              y=y,
                                        loss_function=loss_function)

        # log the batch loss depending on the batch index
        if batch_index % log_per_batch == 0 and use_wandb:
            log_dict = {"epoch": epoch_index, "batch": batch_index} 
            log_dict.update(batch_val_res)
            wandb.log(log_dict)                

        if epoch_val_metrics is None:
            epoch_val_metrics = batch_val_res.copy()
        else:
            for loss_key, _ in epoch_val_metrics.items():
                epoch_val_metrics[loss_key] += batch_val_res[loss_key]        

        # epoch_val_loss += batch_val_loss


    # average by epoch
    for loss_key, _ in epoch_val_metrics.items():
        epoch_val_metrics[loss_key] /= len(dataloader)        

    epoch_val_metrics['epoch'] = epoch_index

    if use_wandb:
        wandb.log(epoch_val_metrics)
        # wandb.log({"epoch": epoch_index, "val_loss": epoch_val_loss / len(dataloader)})

    return epoch_val_metrics
