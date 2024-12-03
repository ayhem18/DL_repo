import torch, os
import torchvision.transforms as tr 

from typing import Optional, List, Dict
from torch.optim.adam import Adam
from pytorch_lightning import LightningModule, Trainer
from torchvision.datasets import MNIST


from mypt.data.dataloaders.standard_dataloaders import initialize_train_dataloader
from mypt.linearBlocks.fully_connected_blocks import ExponentialFCBlock 
from mypt.code_utilities import pytorch_utilities as pu
from mypt.backbones.resnetFeatureExtractor import ResNetFeatureExtractor
from mypt.data.datasets.genericFolderDs import GenericDsWrapper
from mypt.shortcuts import P
from mypt.dimensions_analysis.dimension_analyser import DimensionsAnalyser
from mypt.models.simClr.simClrModel import ResnetSimClr

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))

class FcWrapper(LightningModule):
    def __init__(self, 
                 lr: float,
                 num_layers: int, 
                 dropout: Optional [float],
                 ):
        super().__init__()

        # self.backbone = ResNetFeatureExtractor(num_layers=1, freeze=False)
        self.backbone = ResnetSimClr(input_shape=(3, 28, 28), output_dim=10, num_fc_layers=num_layers, fe_num_blocks=1)

        self.flatten_layer = torch.nn.Flatten()
        self.lr = lr
        self.loss = torch.nn.CrossEntropyLoss() 

        # dim_anal = DimensionsAnalyser(net=torch.nn.Sequential(self.backbone, self.flatten_layer), method='static')
        
        # _, in_features = dim_anal.analyse_dimensions(input_shape=(10, 3, 28, 28))

        # self.fc = ExponentialFCBlock(output=10, 
        #                                 in_features=in_features, 
        #                                 num_layers=num_layers,
        #                                 dropout=dropout
        #                                 )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # return self.fc.forward(self.flatten_layer.forward(self.backbone.forward(x)))
        return self.backbone.forward(x)

    def configure_optimizers(self):
        # return Adam(self.backbone.parameters(), lr=0.01)
        return Adam(params=
                    [
                        {"params": self.backbone.parameters(), "lr": 0.01}, 
                        # {"params": self.fc.parameters(), "lr": 0.01}
                    ]
                    )


    def training_step(self, batch, *args, **kwargs):
        x, y = batch
        model_output = self.forward(x)[1]
        return self.loss.forward(model_output, y)
    
    # # adding this to test some hypothesis    
    def to(self, *args, **kwargs):
        self.backbone = self.backbone.to(*args, **kwargs)
        # self.fc = self.fc.to(*args, **kwargs)
        return self


# create a custom dataset around the MNIST dataset to see if multi-threading still works properly
class MnsitGenericWrapper(GenericDsWrapper):
    def __init__(self, 
                root_dir: P, 
                augmentations: List,
                train,
                samples_per_cls: Optional[int] = None) -> None:

        super().__init__(
                root_dir=root_dir,
                augmentations=augmentations,
                train=train)

        self._ds = MNIST(root=root_dir,     
                         train=True,
                         transform=tr.Compose(self.augmentations),
                         download=True)
    
        # call the self._set_samples_per_cls method after setting the self._ds field
        if samples_per_cls is not None:
            self.samples_per_cls_map = self._set_samples_per_cls(samples_per_cls)
            self._len = 10 * samples_per_cls
        else:
            self._len = len(self._ds)

    def __getitem__(self, index):
        return self._ds.__getitem__(index)

def run(data_folder: P):
    ds = MNIST(root=data_folder, 
          train=True, 
          transform=tr.Compose([tr.ToTensor(), tr.Lambda(lambda x : torch.concat([x, x, x], dim=0))]), 
          download=True
          )
    
    # # let's try to make it work with a custom dataset
    
    # ds = MnsitGenericWrapper(root_dir=data_folder, 
    #                          train=True, 
    #                          samples_per_cls=None,
    #                          augmentations=[tr.ToTensor(), tr.Lambda(lambda x : x.reshape(-1,))]
    #                          )

    dl = initialize_train_dataloader(dataset_object=ds, 
                                     seed=0, 
                                     batch_size=512, 
                                     num_workers=0, 
                                     warning=False)  

    log_dir = os.path.join(SCRIPT_DIR, 'temp_logs')

    device = pu.get_default_device()

    wrapper = FcWrapper(lr=0.1, 
                        num_layers=3, 
                        dropout=0.1)

    trainer = Trainer(
                    accelerator='gpu' if 'cuda' in device else 'cpu',  
                    devices=2, 
                    # strategy='ddp',
                    logger=False, # no logging for now
                    default_root_dir=log_dir,
                    max_epochs=5,
                    check_val_every_n_epoch=2,
                    log_every_n_steps=3,
                    )

    trainer.fit(wrapper, train_dataloaders=dl)

if __name__ == '__main__':
    mnist_data = os.path.join(SCRIPT_DIR, 'data', 'mnist')
    run(mnist_data)
