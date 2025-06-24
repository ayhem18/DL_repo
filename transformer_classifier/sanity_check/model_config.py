import torch

from dataclasses import dataclass

from mypt.nets.transformers.transformer_classifier import TransformerClassifier


from config import GeneralConfig
from DL_repo.transformer_classifier.sanity_check.torch_transformer_classifier import PytorchTransformerClassifier


class GeneralModelConfig(GeneralConfig):
    """Configuration for the general model."""
    model_seed: int = 0



@dataclass
class MyTransformerModelConfig(GeneralModelConfig):
    """Configuration for my implementation of the Transformer Model."""
    model_type: str = "my_transformer"


    # Model parameters
    d_model: int = 64
    num_heads: int = 4
    key_dim: int = 32
    value_dim: int = 32
    num_transformer_blocks: int = 3
    num_classification_layers: int = 2
    dropout: float = 0.1
    pooling: str = "mean"  # Options: "cls", "mean", "max", "last"


@dataclass
class PytorchTransformerModelConfig(GeneralModelConfig):
    """Configuration for the Pytorch Transformer Model."""

    model_type = "pytorch_transformer"

    # Model parameters
    d_model: int = 64
    num_heads: int = 4
    num_transformer_blocks: int = 3
    num_classification_layers: int = 2
    dropout: float = 0.1
    pooling: str = "mean"  # Options: "cls", "mean", "max", "last"


def get_model_config(model_type: str) -> GeneralModelConfig:
    """Get the model config based on the model type."""
    for model_config in [MyTransformerModelConfig, PytorchTransformerModelConfig]:
        if model_config.model_type == model_type:
            return model_config
        
    raise ValueError(f"Invalid model type: {model_type}")


def get_model(config: GeneralModelConfig, num_classes: int) -> TransformerClassifier | PytorchTransformerClassifier:
    if isinstance(config, MyTransformerModelConfig):
        return TransformerClassifier(
            d_model=config.d_model,
            num_transformer_blocks=config.num_transformer_blocks,
            num_classification_layers=config.num_classification_layers,
            num_heads=config.num_heads,
            value_dim=config.value_dim,
            key_dim=config.key_dim,
            num_classes=num_classes,
            pooling=config.pooling,
            dropout=config.dropout
        )
    elif isinstance(config, PytorchTransformerModelConfig):
        
        return PytorchTransformerClassifier(
            d_model=config.d_model,
            num_transformer_blocks=config.num_transformer_blocks,
            num_classification_layers=config.num_classification_layers,
            num_heads=config.num_heads,
            num_classes=num_classes,
            pooling=config.pooling,
            dropout=config.dropout
        )


    raise ValueError(f"Invalid model type: {config.model_type}")