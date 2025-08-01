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

    # lr scheduler: every epoch
    # lr: 1e-3 works for batch sizes of 8 to 16, training diverges for batch sizes of 32 and above 
    
    # lr scheduler: every step
    # let's see how it goes

    learning_rate: float = 1e-4
    num_warmup_steps: int = 250
    # weight_decay: float = 0.0
    # lr_warmup_steps: int = 100

class TrainingConfig(Config):
    num_epochs: int = 50
    val_per_epoch: int = 5
    max_grad_norm: float = 1.0
    validation_timesteps: Optional[list] = [10, 50, 100, 250, 500, 750, 999]
    timestep_bins: Optional[list] = [50, 250, 500, 1000]
    loss_thresholds: Optional[list] = [0.1, 0.05, 0.02, 0.01]
    timesteps_sampler_type: str = "curriculum"

