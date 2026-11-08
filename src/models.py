"""PyTorch Deep Learning Architectures for Financial Time-Series Forecasting.

Upgrades the original 4-layer stacked recurrent architecture [50, 60, 80, 120]
with progressive dropout [0.2, 0.3, 0.4, 0.5] from Ali Azizi's academic study
into modular, GPU-ready PyTorch nn.Module classes.
"""

from typing import Sequence
import torch
import torch.nn as nn


class StackedRecurrentForecaster(nn.Module):
    """4-layer stacked Recurrent Neural Network (LSTM, GRU, or Vanilla RNN) in PyTorch.

    Preserves the signature [50, 60, 80, 120] hidden-unit progression and
    [0.2, 0.3, 0.4, 0.5] progressive dropout schedule across layers.
    """

    def __init__(
        self,
        input_dim: int = 5,
        hidden_dims: Sequence[int] = (50, 60, 80, 120),
        dropouts: Sequence[float] = (0.2, 0.3, 0.4, 0.5),
        cell_type: str = "LSTM",
        use_dropout: bool = True,
    ) -> None:
        super().__init__()
        if len(hidden_dims) != len(dropouts):
            raise ValueError("hidden_dims and dropouts must have the same length.")

        self.cell_type = cell_type.upper()
        self.use_dropout = use_dropout
        cell_map = {
            "LSTM": nn.LSTM,
            "GRU": nn.GRU,
            "RNN": nn.RNN,
        }
        if self.cell_type not in cell_map:
            raise ValueError(f"Unsupported cell_type: {cell_type}. Choose from {list(cell_map.keys())}.")

        rnn_cls = cell_map[self.cell_type]
        self.layers = nn.ModuleList()
        self.dropouts = nn.ModuleList()

        in_features = input_dim
        for hidden_size, drop_prob in zip(hidden_dims, dropouts):
            self.layers.append(
                rnn_cls(
                    input_size=in_features,
                    hidden_size=hidden_size,
                    num_layers=1,
                    batch_first=True,
                )
            )
            self.dropouts.append(nn.Dropout(drop_prob if use_dropout else 0.0))
            in_features = hidden_size

        self.head = nn.Linear(hidden_dims[-1], 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x
        for rnn_layer, drop_layer in zip(self.layers, self.dropouts):
            out, _ = rnn_layer(out)
            out = drop_layer(out)
        last_step = out[:, -1, :]
        return self.head(last_step)


class TemporalMLPForecaster(nn.Module):
    """Multi-Layer Perceptron (MLP) baseline over flattened 28-day lookback windows."""

    def __init__(
        self,
        seq_len: int = 28,
        input_dim: int = 5,
        hidden_dims: Sequence[int] = (50, 60, 80, 120),
        dropouts: Sequence[float] = (0.2, 0.3, 0.4, 0.5),
        use_dropout: bool = True,
    ) -> None:
        super().__init__()
        layers = [nn.Flatten()]
        in_features = seq_len * input_dim
        for hidden_size, drop_prob in zip(hidden_dims, dropouts):
            layers.append(nn.Linear(in_features, hidden_size))
            layers.append(nn.ReLU())
            if use_dropout and drop_prob > 0:
                layers.append(nn.Dropout(drop_prob))
            in_features = hidden_size
        layers.append(nn.Linear(in_features, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def build_model(
    architecture: str = "LSTM",
    seq_len: int = 28,
    input_dim: int = 5,
    use_dropout: bool = True,
) -> nn.Module:
    """Factory helper to instantiate LSTM, GRU, RNN, or MLP models."""
    arch = architecture.upper()
    if arch in ("LSTM", "GRU", "RNN"):
        return StackedRecurrentForecaster(
            input_dim=input_dim,
            cell_type=arch,
            use_dropout=use_dropout,
        )
    if arch == "MLP":
        return TemporalMLPForecaster(
            seq_len=seq_len,
            input_dim=input_dim,
            use_dropout=use_dropout,
        )
    raise ValueError(f"Unknown architecture '{architecture}'. Supported: LSTM, GRU, RNN, MLP.")
