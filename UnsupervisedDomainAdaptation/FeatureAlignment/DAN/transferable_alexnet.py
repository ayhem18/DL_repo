"""
This script contains the implementation of convolutional classification model based on Alexnet. The implementation is not meant to be general
as it is tailored for Deep Transfer Learning later on.
"""
import torch
import re

from typing import Tuple, List, Iterator
from torch import nn
from collections import OrderedDict
from copy import deepcopy

from src.backbones.alexnetFeatureExtractor import AlexNetFeatureExtractor
from src.dimensions_analysis.dimension_analyser import DimensionsAnalyser
from src.linear_blocks.classification_head import ExponentialClassifier

class TransferAlexNet(nn.Module):
    def __init__(self,
                 input_shape: Tuple[int, int, int],
                 num_classes: int,
                 alexnet_blocks, 
                 alexnet_frozen_blocks,
                 num_classification_layers, 
                 *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.input_shape = input_shape
        self.num_classification_layers = num_classification_layers

        self.fe = AlexNetFeatureExtractor(model_blocks=alexnet_blocks, 
                                          frozen_model_blocks=alexnet_frozen_blocks)
        self.image_transformation = self.fe.transform
        self.flatten_layer = nn.Flatten()

        # to calculate the size of features passed to the classification head, we will make use of the dimension_analysis package
        _, in_features = DimensionsAnalyser(net=nn.Sequential(self.fe, self.flatten_layer)).analyse_dimensions(input_shape=(10, ) + self.input_shape)

        if not isinstance(in_features, int):
            raise ValueError(f"Make sure the output of the feature extractor is 2 dimensional so it can be passed to the fully connected block")

        self.ch = ExponentialClassifier(num_classes=num_classes, 
                                        in_features=in_features, 
                                        num_layers=num_classification_layers)

        self._model = nn.Sequential(OrderedDict([
                                                 ("feature extractor", self.fe), 
                                                 ("flatten", self.flatten_layer), 
                                                 ("classification_head", self.ch)
                                                 ]
                                                ))
        
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        x_temp = x[0] if x.ndim == 4 else deepcopy(x)
        if x_temp.shape != self.input_shape:
            raise ValueError(f"Please make sure the input to the model is expected. Expected: {self.input_shape}, Found: {x_temp.shape}")

        # first pass the model through the feature extractor
        x_embedding = self.flatten_layer.forward(self.fe.forward(x))
        output = [x_embedding]

        x = x_embedding        
        for fc_name, fc_block in self.ch.named_children():
            if re.match(r'fc_\d{1,2}', fc_name) is None:
                raise ValueError(f"Please make sure we are saving the outputs of each fully connected block and not inner layers. Found the ouput of a layer named: {fc_name}")
            x = fc_block.forward(x)
            output.append(x)

        # make sure the size of the 'output' list correspond to 1 + the number of layers
        assert len(output) == 1 + self.num_classification_layers, f"The size of output is expected to be {1 + self.num_classification_layers}. Found: {len(output)}"
        return output

    def children(self) -> Iterator[nn.Module]:
        return self._model.children()

    def named_children(self) -> Iterator[tuple[str, nn.Module]]:
        return self._model.named_children()

    def modules(self) -> Iterator[nn.Module]:
        return self._model.modules()

    def __str__(self) -> str:
        # the print statement will be much clearer after setting the __str__ method
        return self._model.__str__()

    def __repr__(self) -> str:
        # the print statement will be much clearer after setting the __str__ method
        return self._model.__repr__()
