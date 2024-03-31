import torch
from typing import Union
from typing import List
from typing import Optional
import torch.nn as nn

Activation = Union[str, nn.Module]

_str_to_activation = {
    'relu': nn.ReLU(),
    'tanh': nn.Tanh(),
    'leaky_relu': nn.LeakyReLU(),
    'sigmoid': nn.Sigmoid(),
    'selu': nn.SELU(),
    'softplus': nn.Softplus(),
    'identity': nn.Identity(),
}

class MLPFactory:

    @staticmethod
    def build_mlp(
            input_size: int,
            output_size: int,
            hidden_size: int,
            n_layers: int,
            activation: Activation = 'tanh',
            output_activation: Activation = 'identity',
            device: Optional[str] = "cpu"
    ) -> nn.Sequential:
        """

        Args:
            input_size: Dimension of input tensor
            output_size: Dimension of output tensor
            hidden_size: Hidden dimension of neural network
            n_layers: Number of hidden layers
            activation: activation function between hidden layers
            output_activation: output activation function
            device: device

        Returns:

            A multi-layer perceptron with parameters as above

        """

        if isinstance(activation, str):
            activation = _str_to_activation[activation]
        if isinstance(output_activation, str):
            output_activation = _str_to_activation[output_activation]

        layers: List = []
        _input_size: int = input_size

        for _ in range(n_layers):
            layers.append(nn.Linear(_input_size, hidden_size))
            layers.append(activation)
            _input_size = hidden_size
        layers.append(nn.Linear(_input_size, output_size))
        layers.append(output_activation)

        mlp: nn.Sequential = nn.Sequential(*layers)
        mlp.to(device)

        return mlp
