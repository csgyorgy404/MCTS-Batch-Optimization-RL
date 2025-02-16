import torch
import torch.nn as nn
from typing import Optional
from torch.optim import Adam, SGD, Adagrad


class Model(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features,
        out_features,
        hidden_activation="relu",
        out_activation=None,
        optimizer_type="adam",
        lr=1e-3,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.out_features = out_features
        self.hidden_activation = hidden_activation
        self.out_activation = out_activation
        self.flatten = nn.Flatten()

        self.model = self._create_model()
        self.optimizer = self._get_optimizer(optimizer_type, lr)

    def _get_optimizer(self, optimizer_type: str, lr: float) -> torch.optim.Optimizer:
        """Returns the optimizer for training the model.

        Args:
            optimizer_type (str): The type of optimizer to use. Options: 'adam', 'sgd', or 'adagrad'.
            lr (float): The learning rate for the optimizer.

        Returns:
            torch.optim.Optimizer: An optimizer corresponding to the specified type.

        Raises:
            NotImplemented: If the optimizer type is not implemented.
        """
        if optimizer_type == "adam":
            return Adam(self.model.parameters(), lr=lr)
        elif optimizer_type == "sgd":
            return SGD(self.model.parameters(), lr=lr)
        elif optimizer_type == "adagrad":
            return Adagrad(self.model.parameters(), lr=lr)
        else:
            print(f"{optimizer_type}, optimizer is not implemented")
            raise NotImplemented

    def _get_activation(self, activation_type: Optional[str]) -> nn.Module:
        """Returns the activation function module based on the activation type.

        Args:
            activation_type (Optional[str]): The activation type. Options: 'relu', 'tanh', 'sigmoid',
                'softmax', 'leakyrelu' or None.

        Returns:
            nn.Module: The activation function module.

        Raises:
            NotImplementedError: If the activation type is not implemented.
        """
        if activation_type == "relu":
            return nn.ReLU()
        elif activation_type == "tanh":
            return nn.Tanh()
        elif activation_type == "sigmoid":
            return nn.Sigmoid()
        elif activation_type == "softmax":
            return nn.Softmax(dim=-1)
        elif activation_type == "leakyrelu":
            return nn.LeakyReLU()
        elif activation_type is None:
            return nn.Identity()
        else:
            print(f"No {activation_type} activation is implemented")
            raise NotImplementedError

    def _create_model(self) -> nn.Module:
        """Creates and returns a Sequential model composed of linear layers and activation functions.

        Returns:
            nn.Module: The constructed Sequential model.
        """
        activation = self._get_activation(self.hidden_activation)

        layers = []

        start_dim = self.in_features

        for n in self.hidden_features:
            layers.extend([nn.Linear(start_dim, n), activation])
            start_dim = n

        layers.extend([nn.Linear(start_dim, self.out_features), self._get_activation(self.out_activation)])

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Performs the forward pass of the model.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The model output.
        """
        if len(x.shape) == 3:
            x = self.flatten(x)
        return self.model(x)
