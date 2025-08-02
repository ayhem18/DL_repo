import json

from typing import List, Tuple, Optional
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
    input_shape: Tuple[int, int, int] = (3, 128, 128)
    model_type: str = "diffusers"

@dataclass
class OptimizerConfig(Config):
    learning_rate: float = 1e-4
    num_warmup_steps: int = 250
    max_grad_norm: float = 1.0

class TrainingConfig(Config):
    dataset: str = "butterflies"
    train_batch_size: int = 64
    val_batch_size: int = 16

    num_epochs: int = 100
    val_per_epoch: int = 5
    validation_timesteps: Optional[list] = [10, 50, 100, 250, 500, 750, 999]
    timestep_bins: Optional[list] = [50, 250, 500, 1000]
    loss_thresholds: Optional[list] = [0.1, 0.1, 0.075]
    timesteps_sampler_type: str = "curriculum"

