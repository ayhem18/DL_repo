"""
This file contains a pytorch lightning wrapper for the SimClr Model
"""
import torch, warnings, torch_optimizer as topt, numpy as np

from typing import Union, Tuple, List, Dict, Optional, Iterator, Callable
from collections import defaultdict
from torch.optim.lr_scheduler import CosineAnnealingLR
from pytorch_lightning import LightningModule
from clearml import Logger

from mypt.losses.simclr_loss import SimClrLoss
from mypt.code_utilities import pytorch_utilities as pu
from mypt.models.simClr.simClrModel import SimClrModel, ResnetSimClr
from mypt.similarities.cosineSim import CosineSim
from mypt.schedulers.warmup import set_warmup_epochs


AUGMENTATIONS_PER_CURVE = 5

def cosine_sims_model_embeddings_with_augmentations(model: torch.nn.Module,
                                                      process_model_output: Callable,
                                                      batch: torch.nn.Module,
                                                      augmentations: List[Callable]
                                                      ):
    n_samples = len(batch)
    # set the model to eval mode
    model.eval()
    # apply the augmentations to the batches
    sims_augmentations = []

    with torch.no_grad():
        batch_embds = process_model_output(model.forward(batch))

        for t in augmentations:
            batch_t = t.forward(batch)

            batch_t_embds = process_model_output(model.forward(batch_t)) 
            # compute the similarities between samples and their augmented versions
            sample_aug_similarities = torch.diagonal(CosineSim().forward(batch_embds, batch_t_embds)).unsqueeze(-1)

            assert sample_aug_similarities.shape == (n_samples, 1)

            sims_augmentations.append(sample_aug_similarities)


    final_sims = torch.concat(sims_augmentations, dim=1) # concatenate horizontally
    assert final_sims.shape == (n_samples, len(augmentations))
    return final_sims  


class SimClrModelWrapper(LightningModule):
    ######################### Parameters Verification #########################
    @classmethod
    def verify_optimizer_parameters(cls, num_epochs: int, num_warmup_epochs:int):
        if num_warmup_epochs >= num_epochs:
            raise ValueError(f"The number of warmup epochs must be strictly less than the total number of epochs !!. found warmup : {num_warmup_epochs}, and epochs: {num_epochs}")

        if num_warmup_epochs == 0:
            warnings.warn(message="The number of warmup epochs is set to 0 !!")

    @classmethod
    def _set_learning_rates(cls, lrs: Union[Tuple[float, float], float]) -> Tuple[float, float]:
        if isinstance(lrs, float):
            lr1, lr2 = lrs, 10 * lrs
        elif isinstance(lrs, (Tuple,List)) and len(lrs) == 2:
            lr1, lr2 = lrs

        elif isinstance(lrs, (Tuple,List)) and len(lrs) == 1:
            lr1, lr2 = lrs[0], lrs[0] * 10
            
        else:
            raise ValueError(f"The current implementation supports at most 2 learning rates. Found: {len(lrs)} learning rates")

        return lr1, lr2
    

    def __init__(self,
                input_shape: Tuple[int, int, int],

                #logging arguments
                log_per_batch: int,
                val_per_epoch: int,
                logger: Optional[Logger], 
                # loss arguments
                temperature: float,
                debug_loss:bool,
                # optimizer arguments
                lrs: Union[Tuple[float], float], 
                num_epochs: int,
                num_warmup_epochs:int,                 

                # arguments for the extra dataloaders
                debug_embds_vs_trans: bool,
                debug_augmentations: Optional[List],
                save_hps: bool,
                get_augmentation_name: Optional[Union[Callable, Dict]] = None,
                ):
        # super class call
        super().__init__()

        # save the hyperparameters
        if save_hps:
            self.save_hyperparameters(ignore='logger') # the logger parameter cannot be pickled (when it is not None, that is)

        # counters to save the training and validation epochs
        self.train_epoch_index = 0
        self.val_epoch_index = 0

        # save the number of batches per train and validation epochs
        self.batches_per_train_epoch = None
        self.batches_per_val_epoch = None

        # the model
        self.model: SimClrModel = None
        self.input_shape = input_shape

        # make sure that the loss override the nn.Module and calls the super constructor (otherwise some unexpected error would be raised)
        self._loss = SimClrLoss(temperature=temperature, debug=debug_loss)
        
        # the optimizer parameters
        self.lr1, self.lr2 = self._set_learning_rates(lrs)
        self._num_epochs = num_epochs
        self._num_warmup_epochs = num_warmup_epochs

        # logging parameters: 
        # this parameter has to be passed since, I will be doing the logging to clearml myself 
        self.log_per_batch = log_per_batch  
        self.val_per_epoch = val_per_epoch
        self.myLogger = logger

        # logging variables
        # a field to save the metrics logged during the training batches
        self.train_batch_losses: List[Dict[str, float]] = []
        # a field to save the metrics logged during the validation batches 
        self.val_batch_losses: List[Dict[str, float]] = [] 

        self.debug_embds_vs_trans = debug_embds_vs_trans

        if self.debug_embds_vs_trans and debug_augmentations is None:
            raise ValueError(f"setting debug_embds_vs_trans to True without passing debugging augmentations")

        self.debug_augmentations = debug_augmentations
        self.augmentations_scores = defaultdict(lambda :[])

        if get_augmentation_name is None:
            warnings.warn(("The 'get_augmentation_name' argument was passed as a None object.\n" 
                          "The default function might not be suitable for more complex augmentations: Mainly those involving the `Compose` class"))
            get_augmentation_name = pu.get_augmentation_name
        
        self.get_augmentation_name = get_augmentation_name        



    ######################### Logging methods #########################
    def train_batch_log(self, 
                   log_dict: Dict, 
                   batch_idx: int) -> None:
        """
        Args:
            log_dict (Dict): the batch output saved as a dictionary
            batch_idx (int): The index of the batch (might be used to decide which batch output to log)
        """

        # make sure to refer to myLogger and not self.logger
        if self.myLogger is None or batch_idx % self.log_per_batch != 0: 
            return
        
        for key, value in log_dict.items():
            self.myLogger.report_scalar(title=key, 
                                    series=key, 
                                    value=value, 
                                    iteration=self.global_step
                                    )

    def val_batch_log(self, 
                      log_dict: Dict,
                      batch_idx:int) -> None:
        # make sure to refer to myLogger and not self.logger
        if self.myLogger is None: 
            return
        
        for key, value in log_dict.items():
            self.myLogger.report_scalar(title=key, 
                                    series=key, 
                                    value=value, 
                                    iteration=self.val_epoch_index * self.batches_per_val_epoch + batch_idx
                                    )
        
    def _aug_scores_logs(self, aug_scores: List[float], batch_idx:int):
        if self.myLogger is None:
            return 

        if len(aug_scores) != len(self.debug_augmentations):
            raise ValueError(f"The number of augmentation scores does not match the number of augmentations...")

        # divide the logging for augmentation scores into multiple curves. Otherwise, it might get too cluttered
        for i in range(0, len(self.debug_augmentations), AUGMENTATIONS_PER_CURVE):
            for aug_cls, aug_s in zip(self.debug_augmentations[i: i + AUGMENTATIONS_PER_CURVE], aug_scores[i: i + AUGMENTATIONS_PER_CURVE]):
                self.myLogger.report_scalar(title=f"augmentation scores_GROUP_{i // AUGMENTATIONS_PER_CURVE + 1}",
                                            series = self.get_augmentation_name(aug_cls),
                                            value=aug_s,
                                            iteration=self.val_epoch_index * self.batches_per_val_epoch + batch_idx
                                            )


    ######################### Model forward pass #########################
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.model.forward(x)
        
    def _step(self, batch: Tuple[torch.Tensor, torch.Tensor], split: str) -> Tuple[torch.Tensor, Dict]:
        x1_batch, x2_batch = batch
        
        # concatenate the input 
        x = torch.cat([x1_batch, x2_batch])
        
        # model forward
        _, g_x = self.forward(x)
        
        # calculate the loss
        batch_loss_obj = self._loss.forward(g_x)

        log_dict = {}

        if isinstance(batch_loss_obj, Tuple):
            batch_loss_obj, positive_pairs_sims, negative_pairs_sims = batch_loss_obj            

            pos_neg_sim_stats = {
                        f"{split}_avg_positive_pair_sim": torch.mean(positive_pairs_sims).item(),
                        f"{split}_avg_negative_pair_sim": torch.mean(negative_pairs_sims).item()
                    }

            log_dict.update(pos_neg_sim_stats)

        # at this point of the code we know that batch_loss_obj actually represents the loss tensor and the tuple with the logging stats
        log_dict.update({f"batch_{split}_loss": batch_loss_obj.item()})

        return batch_loss_obj, log_dict


    ######################### training methods #########################            
    def training_step(self,  
                      batch: Tuple[torch.Tensor, torch.Tensor], 
                      batch_idx: int) -> SimClrLoss:
        batch_loss_obj, log_dict = self._step(batch, split='train')
        self.train_batch_log(log_dict=log_dict, batch_idx=batch_idx)
        
        # save only the training loss
        self.train_batch_losses.append(log_dict['batch_train_loss'])
        return batch_loss_obj

    def on_train_epoch_end(self):
        # calculate the average of batch losses
        train_epoch_loss = np.mean(self.train_batch_losses)
        self.train_batch_losses.clear()

        # log to ClearMl
        if self.myLogger is not None:
            self.myLogger.report_scalar(
                                        title="train_epoch_loss", 
                                        series="train_epoch_loss", 
                                        value=train_epoch_loss, 
                                        iteration=self.train_epoch_index
                                        )
            
        # log the epoch validation loss to use it for checkpointing
        self.log(name='train_epoch_loss', 
                 value=train_epoch_loss,
                 sync_dist=False)
        self.train_epoch_index += 1

    ######################### validation methods #########################
    def validation_step_debug(self, 
                            batch: Tuple[torch.Tensor, torch.Tensor],
                            batch_idx:int,
                            ):

        with torch.no_grad():
            # step 1
            _, fx = self.forward(batch)
            # compute the cosine similarities between the embeddings of the batch while setting the diagonal values to "0"
            batch_cos_sims = CosineSim().forward(fx, fx).fill_diagonal_(0)
            
            # find the closest sample for each sample
            # broadcast it to have the same shape as the cos_sims_augs object 
            closest_neighbors = torch.broadcast_to(torch.max(batch_cos_sims, dim=1, keepdim=True)[0], # the maximum similarity per each sample (dim = 1), keepdim to keep my sanity
                                                   size=(len(batch), len(self.debug_augmentations))
                                                   )

            # step 2 
            cos_sims_augs = cosine_sims_model_embeddings_with_augmentations(model=self.model, 
                                                                            process_model_output=lambda x : x[1],
                                                                            batch = batch,
                                                                            augmentations=self.debug_augmentations)

            # calculate the number of samples for which is the augmentation version is less similar then the closest neighbor
            aug_scores = torch.mean((cos_sims_augs >= closest_neighbors).to(torch.float32), dim=0).cpu().tolist()
            self._aug_scores_logs(aug_scores=aug_scores, batch_idx=batch_idx)

            # save the augmentation scores as well
            for a, a_sc in zip(self.debug_augmentations, aug_scores):
                # get the augmentation name and append the scores
                self.augmentations_scores[self.get_augmentation_name(a)].append(a_sc)
      

    def validation_step_forward(self, val_batch: Tuple[torch.Tensor, torch.Tensor], 
                                batch_index:int) -> SimClrLoss:
        batch_loss_obj, log_dict = self._step(val_batch, split='val')
        self.val_batch_log(log_dict=log_dict, batch_idx=batch_index) # log all validation batches
            
        # track the validation batch loss
        self.val_batch_losses.append(log_dict['batch_val_loss'])
        return batch_loss_obj

    def validation_step(self, 
                        val_batch, 
                        val_batch_idx, 
                        *args,):
        if self.batches_per_val_epoch is None:
            self.batches_per_val_epoch = val_batch_idx + 1
        self.batches_per_val_epoch = max(self.batches_per_val_epoch, val_batch_idx + 1)

        # first check if we are setting the augmentation debug mode 
        if self.debug_embds_vs_trans:
            # make sure the dataloader index is passed
            if len(args) != 1:
                raise ValueError(f"The model is set to the debug embedding mode. The dataloader index must be passed")

        if self.debug_embds_vs_trans and args[0] == 1:
            self.validation_step_debug(val_batch, val_batch_idx)
            return
        
        if self.val_epoch_index % self.val_per_epoch == 0:
            return self.validation_step_forward(val_batch, val_batch_idx)
        
    def on_validation_epoch_end(self):
        # certain oeprations take place only when the self.validation_step_forward method was called
        if self.val_epoch_index % self.val_per_epoch == 0 and len(self.val_batch_losses) > 0:        
            # calculate the average of batch losses
            val_epoch_loss = float(np.mean(self.val_batch_losses))
            # clear the training and validation logs after the
            self.val_batch_losses.clear()

            # log to ClearMl
            if self.myLogger is not None:
                self.myLogger.report_scalar(
                                            title="val_epoch_loss", 
                                            series="val_epoch_loss", 
                                            value=val_epoch_loss,
                                            iteration=self.val_epoch_index)
                                            
            # log the epoch validation loss to use it for checkpointing
            self.log(name='val_epoch_loss', value=val_epoch_loss)

        # increment the validation epoch counter (discard the sanity check epochs)
        self.val_epoch_index += int(not (self._trainer.sanity_checking))
        return 

    ######################### optimizers #########################
    def configure_optimizers(self):
        # I found only one good implementation of LARS through the lightning.flash package. The latter requires pytorch lightning version < 2.0 as a dependency. The current version is 2.4 ...
        # switching to a supported implementation of Lamb (another large batch learning scheduler)

        optimizer = topt.Lamb(params=[{"params": self.model.fe.parameters(), "lr": self.lr1}, # using different learning rates with different components of the model 
                                {"params": self.model.flatten_layer.parameters(), "lr": self.lr1},
                                {"params": self.model.ph.parameters(), "lr": self.lr2}
                                ],
                                lr=self.lr1,
                                weight_decay=10 ** -6
                                )
                    
        cosine_scheduler = CosineAnnealingLR(optimizer=optimizer, T_max=self._num_epochs - self._num_warmup_epochs)

        if self._num_epochs > 0:
            lr_scheduler_with_warmup = set_warmup_epochs(optimizer=optimizer, 
                                                        main_lr_scheduler=cosine_scheduler, 
                                                        num_warmup_epochs=self._num_warmup_epochs, 
                                                        warmup_lr_scheduler='linear')

            return {"optimizer": optimizer, "lr_scheduler": lr_scheduler_with_warmup}

        return {"optimizer": optimizer, "lr_scheduler": cosine_scheduler}


    ######################### Pytorch.nn.Module methods #########################
    def to(self, *args, **kwargs):
        self.model = self.model.to(*args, **kwargs)        
        self._loss = self._loss.to(*args, **kwargs)
        return self

    def children(self) -> Iterator[torch.nn.Module]:
        # overloading this method to return the correct children of the wrapper: those of the self.model field
        return self.model.children()

    def modules(self) -> Iterator[torch.nn.Module]:
        return self.model.modules()

    def named_children(self) -> Iterator[Tuple[str, torch.nn.Module]]:
        return self.model.named_children()
        
    ######################### Checkpointing methods #########################
    def on_save_checkpoint(self, checkpoint: Dict):
        # add the augmentation scores
        checkpoint['augmentation_scores'] = dict(self.augmentations_scores)
        checkpoint['debug_augmentations'] = self.debug_augmentations

        
        # those fields are used in the logging of the validation metrics
        # they need to saved so that logging will be synchronized across consecutive runs
        checkpoint['val_epoch_index'] = self.val_epoch_index
        checkpoint['batches_per_val_epoch'] = self.batches_per_val_epoch
        checkpoint['train_epoch_index'] = self.train_epoch_index

        return super().on_save_checkpoint(checkpoint)

    def on_load_checkpoint(self, checkpoint: Dict):
        # the custom fields were not part of the checkpoint saving mechanism, can't expect them to be part of the 
        # custom loading mechanism

        self.val_epoch_index = checkpoint['val_epoch_index']
        self.batches_per_val_epoch = checkpoint['batches_per_val_epoch'] 
        self.train_epoch_index = checkpoint['train_epoch_index'] 

        # no need to load callbacks (the initialized ones should do the trick)
        del(checkpoint['callbacks'])
        return super().on_load_checkpoint(checkpoint)


class ResnetSimClrWrapper(SimClrModelWrapper):
    def __init__(self,
                input_shape: Tuple[int, int, int],
                output_dim: int,
                num_fc_layers: int,

                #logging parameters
                logger: Optional[Logger],
                log_per_batch: int,
                val_per_epoch:int,
                # loss parameters
                temperature: float,
                debug_loss:bool,
                # optimizer parameters
                lrs: Union[Tuple[float], float], 
                num_epochs: int,
                num_warmup_epochs:int,

                debug_embds_vs_trans: bool,
                debug_augmentations: Optional[List],
                save_hps: bool = False,

                # more model arguments
                dropout: Optional[float] = None,
                architecture: int = 50, 
                freeze: Union[int, bool]=False,  
                ):

        super().__init__(
                input_shape=input_shape,
                #logging parameters
                log_per_batch=log_per_batch,
                logger=logger,
                val_per_epoch=val_per_epoch,
                # loss parameters
                temperature=temperature,
                debug_loss=debug_loss,
                # optimizer parameters
                lrs=lrs, 
                num_epochs=num_epochs,
                num_warmup_epochs=num_warmup_epochs,

                debug_embds_vs_trans=debug_embds_vs_trans,
                debug_augmentations=debug_augmentations,
                save_hps=save_hps
                )

        self.model = ResnetSimClr(
                                input_shape=self.input_shape, 
                                output_dim=output_dim, 
                                num_fc_layers=num_fc_layers, 
                                
                                dropout=dropout, 
                                freeze=freeze, 
                                architecture=architecture
                                )
