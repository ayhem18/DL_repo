from typing import Dict, Any
from dataclasses import dataclass


class GeneralConfig:
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for logging."""
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')} 



@dataclass
class DataConfig(GeneralConfig):
    """Configuration for the data."""
    max_len: int = 32
    train_samples: int = 1000
    val_samples: int = 500
    test_samples: int = 500
    max_mean: float = 100
    all_same_length: bool = True
    data_seed: int = 42


@dataclass
class TrainingConfig(GeneralConfig):
    """Configuration for the training."""
    batch_size: int = 128
    epochs: int = 500
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    early_stopping_patience: int = -1 # -1 means no early stopping


@dataclass 
class ExperimentConfig(GeneralConfig):
    # Logging and output
    log_parent_dir_name: str = "runs"
    experiment_name: str = "transformer_sanity_check"
    save_model: bool = True
    log_interval: int = 10  # Log every N batches
    logger_name: str | None = "tensorboard"

    model_seed: int = 123


# @dataclass
# class TransformerConfig(Config) :
#     """Configuration for the Transformer Classifier sanity check."""
#         # Training parameters
    

        