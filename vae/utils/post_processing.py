from typing import Any
import torch


class PostProcessor:
    def __init__(self, enabled: bool, min_action: float = -1, max_action: float = 1) -> None:
        min_action = self.convert_to_float(min_action)
        max_action = self.convert_to_float(max_action)
        assert min_action <= max_action
        
        self.min_action = min_action
        self.max_action = max_action
        self.span = self.max_action - self.min_action

        self.enabled = enabled
        self.call_func = self.resacale if self.enabled else lambda x: x

    def convert_to_float(self, value: str):
        try:
            return float(value)
        except ValueError:
            ValueError(f"cast of {value} to float failed")

    def resacale(self, x: torch.Tensor) -> torch.Tensor:
        f"""applies tanh function on x and rescales the range
        from [-1, 1] to [{self.min_action}, {self.max_action}]

        Args:
            x (torch.Tensor): tensor to rescale

        Returns:
            torch.Tensor: rescaled tensro
        """
        x = torch.tanh(x)

        x = x + 1  # ranging now from 0 to 2
        x = x / 2  # ranging now from 0 to 1
        x = x * self.span # ranging now from 0 to self.max_action - self.min_action
        x = x + self.min_action  # ranging now from self.min_action to self.max_action

        return x 
    
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.call_func(x)
