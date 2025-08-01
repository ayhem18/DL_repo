import torch
from abc import ABC, abstractmethod

class AbstractTimeStepsSampler(ABC):
    """Abstract base class for timestep samplers."""
    def __init__(self, num_train_timesteps: int):
        self.num_train_timesteps = num_train_timesteps

    @abstractmethod
    def sample(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Sample a batch of timesteps."""
        pass

    @abstractmethod
    def update(self, *args, **kwargs) -> None:
        """Update the sampler."""
        pass

class UniformTimeStepsSampler(AbstractTimeStepsSampler):
    """Samples timesteps uniformly."""
    def sample(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.randint(
            0,
            self.num_train_timesteps,
            (batch_size,),
            device=device,
            dtype=torch.int64
        )

    def update(self, *args, **kwargs) -> None:
        """
        the uniform sampler does need to be updated
        """
        pass

class LogTimeStepsSampler(AbstractTimeStepsSampler):
    """Samples timesteps using a log-uniform distribution."""
    def sample(self, batch_size: int, device: torch.device) -> torch.Tensor:
        log_min = torch.log(torch.tensor(1e-1))
        log_max = torch.log(torch.tensor(float(self.num_train_timesteps - 1)))
        log_timesteps = torch.rand(batch_size, device=device) * (log_max - log_min) + log_min
        timesteps = torch.exp(log_timesteps).long()
        return torch.clamp(timesteps, 0, self.num_train_timesteps - 1)

    def update(self, *args, **kwargs) -> None:
        """
        the log sampler does not need to be updated
        """
        pass




def set_timesteps_sampler(sampler_type: str, num_train_timesteps: int, **kwargs) -> AbstractTimeStepsSampler:
    """Factory function to get a timestep sampler."""
    if sampler_type == "uniform":
        return UniformTimeStepsSampler(num_train_timesteps, **kwargs)
    elif sampler_type == "log":
        return LogTimeStepsSampler(num_train_timesteps, **kwargs)
    else:
        raise ValueError(f"Invalid sampler type: {sampler_type}")
