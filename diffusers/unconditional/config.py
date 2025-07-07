import json

from typing import Tuple
from dataclasses import asdict, dataclass

from mypt.shortcuts import P


@dataclass
class Config: 
    def save(self, filepath: P) -> None:
        """Save the config to a JSON file"""
        # Convert the config to a dictionary
        config_dict = asdict(self)
        
        # Convert tuples to lists for JSON serialization
        for key, value in config_dict.items():
            if isinstance(value, tuple):
                config_dict[key] = list(value)
        
        # Save to JSON file
        with open(filepath, 'w') as f:
            json.dump(config_dict, f, indent=4)

    @classmethod
    def load(cls, filepath: P) -> 'Config':
        """Load the config from a JSON file"""
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        
        # Convert lists back to tuples
        for key, value in config_dict.items():
            if isinstance(value, list):
                config_dict[key] = tuple(value)
        
        # Create a new instance of TrainConfig with the loaded values
        return cls(**config_dict)


# a configuration for the model arguments
@dataclass
class ModelConfig(Config):
    input_shape: Tuple[int, int, int] = (1, 28, 28)


@dataclass
class OptimizerConfig(Config):
    learning_rate: float = 1e-3
    # weight_decay: float = 0.0
    # lr_warmup_steps: int = 100

@dataclass
class TrainingConfig(Config):
    train_batch_size: int = 4
    val_batch_size: int = 4
    num_epochs: int = 25
    seed: int = 42

