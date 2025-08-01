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
        # log_min is 0 since exp(0) = 1 and we want to sample from 1 to num_train_timesteps - 1 (since 0 is the first timestep and there is no learning if the the model is asked to predict the identity function)
        log_max = torch.log(torch.tensor(float(self.num_train_timesteps - 1)))
        log_timesteps = torch.rand(batch_size, device=device) * log_max 
        timesteps = torch.exp(log_timesteps).long()
        return torch.clamp(timesteps, 1, self.num_train_timesteps - 1)

    def update(self, *args, **kwargs) -> None:
        """
        the log sampler does not need to be updated
        """
        pass




def set_timesteps_sampler(sampler_type: str, num_train_timesteps: int, **kwargs) -> AbstractTimeStepsSampler:
    """Factory function to get a timestep sampler."""
    # TODO: add some checks to make sure that the sampler receives the correct arguments
    # for example, a bin-based sampler should receive the bin boundaries as an argument 
    # an adaptive sampler might need some other arguments (to be defined later xD )
    if sampler_type == "uniform":
        return UniformTimeStepsSampler(num_train_timesteps, **kwargs)
    elif sampler_type == "log":
        return LogTimeStepsSampler(num_train_timesteps, **kwargs)
    else:
        raise ValueError(f"Invalid sampler type: {sampler_type}")
