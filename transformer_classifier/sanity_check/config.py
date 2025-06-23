from typing import Dict, Any
from dataclasses import dataclass



@dataclass
class TransformerConfig:
    """Configuration for the Transformer Classifier sanity check."""
    
    # Data parameters
    max_len: int = 32
    train_samples: int = 1000
    val_samples: int = 1000
    test_samples: int = 1000
    all_same_length: bool = True
    max_mean: float = 3.0
    data_seed: int = 42
    
    # Model parameters
    d_model: int = 64
    num_heads: int = 4
    key_dim: int = 16
    value_dim: int = 16
    num_transformer_blocks: int = 2
    num_classification_layers: int = 2
    dropout: float = 0.1
    pooling: str = "mean"  # Options: "cls", "mean", "max", "last"
    
    # Training parameters
    batch_size: int = 64
    epochs: int = 10
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    early_stopping_patience: int = 10
    model_seed: int = 123
    
    # Logging and output
    log_parent_dir_name: str = "runs"
    experiment_name: str = "transformer_sanity_check"
    save_model: bool = True
    log_interval: int = 10  # Log every N batches
    
    # Logger parameters
    logger_name: str | None = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for logging."""
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')} 