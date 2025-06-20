from typing import List

import torch
from torch import Tensor, nn
from torch.nn import Linear, Module, Sequential


class FeedForwardNetwork(Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        architecture: List[int] = None,
        activation_function: str = "ReLU",
        device: str | torch.device = "cpu",
        final_activation: str = None,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._input_dim = input_dim
        self._output_dim = output_dim

        self._final_activation_cls = (
            getattr(nn, final_activation) if final_activation is not None else None
        )
        self._activation_function_cls = getattr(nn, activation_function)
        self._linear = self._create_linear_unit(architecture).to(device)

    def _create_linear_unit(self, architecture: List[int] = None) -> Sequential:
        """creates a linear unit specified with architecture and self._activation_function_type

        Args:
            architecture (List[int]): dimension of linear layers

        Returns:
            Sequential: sequential linear unit
        """
        # input layer
        if architecture is None or (
            isinstance(architecture, list) and len(architecture) == 0
        ):
            layer = Linear(self._input_dim, self._output_dim)
            if self._final_activation_cls is not None:
                layer = Sequential(layer, self._final_activation_cls())
            return layer

        layers = [
            Linear(self._input_dim, int(architecture[0])),
            self._activation_function_cls(),
        ]
        # add hidden layers
        for idx in range(len(architecture) - 1):
            layers.extend(
                [
                    Linear(int(architecture[idx]), int(architecture[idx + 1])),
                    self._activation_function_cls(),
                ]
            )
        # output layer
        layers.append(Linear(architecture[-1], self._output_dim))

        if self._final_activation_cls is not None:
            layers.append(self._final_activation_cls())

        sequence = Sequential(*layers)
        return sequence

    def forward(self, x: Tensor) -> Tensor:
        return self._linear(x)
