import torch
import torch.nn.functional as F
from abc import ABC, abstractmethod
from typing import List, Dict, Union, Tuple


class AbstractTimeStepsSampler(ABC):
    """Abstract base class for timestep samplers."""
    def __init__(self, num_train_timesteps: int):
        self.num_train_timesteps = num_train_timesteps

    @abstractmethod
    def sample(self, batch_size: int, device: Union[torch.device, str]) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Sample a batch of timesteps."""
        pass

    def update(self, *args, **kwargs) -> None:
        """Optional method to update sampler state based on training progress."""
        pass


class UniformTimeStepsSampler(AbstractTimeStepsSampler):
    """Samples timesteps uniformly."""
    def sample(self, batch_size: int, device: Union[torch.device, str]) -> Tuple[torch.Tensor, Dict[str, float]]:
        timesteps = torch.randint(
            0,
            self.num_train_timesteps,
            (batch_size,),
            device=device,
            dtype=torch.int64
        )
        return timesteps, {}


class LogTimeStepsSampler(AbstractTimeStepsSampler):
    """Samples timesteps using a log-uniform distribution."""
    def sample(self, batch_size: int, device: Union[torch.device, str]) -> Tuple[torch.Tensor, Dict[str, float]]:
        log_min = torch.log(torch.tensor(1e-1))
        log_max = torch.log(torch.tensor(float(self.num_train_timesteps)))
        log_timesteps = torch.rand(batch_size, device=device) * (log_max - log_min) + log_min
        timesteps = torch.exp(log_timesteps).long()
        return torch.clamp(timesteps, 0, self.num_train_timesteps - 1), {}

    def update(self, *args, **kwargs) -> None:
        """The log sampler does not need to be updated."""
        pass


class CurriculumTimeStepsSampler(AbstractTimeStepsSampler):
    """
    Implements a curriculum learning and loss-aware sampling strategy.
    
    1. Curriculum: Starts with the highest-noise bin and progressively activates
       lower-noise bins as the model meets performance thresholds.
    2. Loss-Aware Sampling: Within the active bins, it samples timesteps from bins
       with higher losses more frequently.
    """
    def __init__(self, 
                num_train_timesteps: int, 
                bins: List[int], 
                thresholds: List[float]):
        super().__init__(num_train_timesteps)
        
        if len(bins) != len(thresholds) + 1:
            raise ValueError("the number of thresholds must be less than the number of bins by 1. The first bin is automatically activated (doesn't need a threhsold)")

        self.bin_boundaries = torch.tensor([0] + bins)
        self.num_bins = len(bins)

        self.bin_labels = [f'{self.bin_boundaries[i]}-{self.bin_boundaries[i+1]-1}' for i in range(self.num_bins)]

        self.thresholds = thresholds        
        self.active_bin_idx = self.num_bins - 1

        self.last_epoch_losses = {} 

    def sample(self, batch_size: int, device: Union[torch.device, str]) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Samples timesteps based on loss distribution across active bins."""
        active_indices = list(range(self.active_bin_idx, self.num_bins))
        active_labels = [self.bin_labels[i] for i in active_indices]
        
        active_losses = torch.tensor([self.last_epoch_losses.get(label, 1.0) for label in active_labels], device=device)
        
        bin_probabilities = F.softmax(active_losses, dim=0)
        bin_probs_log = {label: prob.item() for label, prob in zip(active_labels, bin_probabilities)}

        all_timestep_probs = torch.zeros(self.num_train_timesteps, device=device)
        
        max_high = float('-inf')

        for i, bin_idx in enumerate(active_indices):
            low = self.bin_boundaries[bin_idx].item()
            high = self.bin_boundaries[bin_idx + 1].item()
            bin_size = high - low
            
            max_high = max(max_high, high)
            
            if bin_size == 0:
                raise ValueError(f"The bin size is 0 for bin {bin_idx}")
            
            per_timestep_prob = bin_probabilities[i] / bin_size
            all_timestep_probs[low:high] = per_timestep_prob

        if max_high != self.num_train_timesteps:
            raise ValueError(f"The highest bin boundary is {max_high} but the number of timesteps is {self.num_train_timesteps}")
        
        sampled_timesteps = torch.multinomial(all_timestep_probs, num_samples=batch_size, replacement=True)
        
        return sampled_timesteps, bin_probs_log


    def update(self, loss_per_bin: Dict[str, float]) -> None:
        """Updates active bins and stores losses for the next epoch."""
        # iterate through the loss_per_bin and make sure that the keys are the bin labels
        for bin_label, loss in loss_per_bin.items():
            if bin_label not in self.bin_labels:
                raise ValueError(f"The bin label {bin_label} is not in the list of bin labels")
            self.last_epoch_losses[bin_label] = loss

        if self.active_bin_idx == 0:
            return

        current_stage_label = self.bin_labels[self.active_bin_idx]
        current_stage_threshold = self.thresholds[self.active_bin_idx - 1]
        
        loss_for_current_stage = self.last_epoch_losses.get(current_stage_label, float('inf'))

        if loss_for_current_stage < current_stage_threshold:
            print(f"\nINFO: Curriculum stage complete. Bin '{current_stage_label}' reached loss {loss_for_current_stage:.4f} (threshold: {current_stage_threshold}).")
            self.active_bin_idx -= 1
            newly_activated_label = self.bin_labels[self.active_bin_idx]
            print(f"INFO: Activating next bin: '{newly_activated_label}'.\n")
            if self.active_bin_idx == 0:
                print("INFO: All timestep bins are now active.")


def set_timesteps_sampler(sampler_type: str, num_train_timesteps: int, **kwargs) -> AbstractTimeStepsSampler:
    """Factory function to get a timestep sampler."""
    if sampler_type == "uniform":
        return UniformTimeStepsSampler(num_train_timesteps)
    elif sampler_type == "log":
        return LogTimeStepsSampler(num_train_timesteps)
    elif sampler_type == "curriculum":
        if 'bins' not in kwargs or 'thresholds' not in kwargs:
            raise ValueError("'bins' and 'thresholds' must be provided for curriculum sampler.")
        return CurriculumTimeStepsSampler(num_train_timesteps, **kwargs)
    else:
        raise ValueError(f"Invalid sampler type: {sampler_type}")
