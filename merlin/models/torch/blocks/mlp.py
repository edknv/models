from typing import List, Optional, Sequence

import torch
from torch import nn

from merlin.models.torch.block import Block
from merlin.models.torch.schema import Schema, output_schema
from merlin.models.torch.transforms.agg import Concat, MaybeAgg
from merlin.models.torch.utils.llama_utils import find_multiple


class MLPBlock(Block):
    """
    Multi-Layer Perceptron (MLP) Block with custom options for activation, normalization,
    dropout.

    Parameters
    ----------
    units : Sequence[int]
        Sequence of integers specifying the dimensions of each linear layer.
    activation : Callable, optional
        Activation function to apply after each linear layer. Default is ReLU.
    normalization : Union[str, nn.Module], optional
        Normalization method to apply after the activation function.
        Supported options are "batch_norm" or any custom `nn.Module`.
        Default is None (no normalization).
    dropout : Optional[float], optional
        Dropout probability to apply after the normalization.
        Default is None (no dropout).
    pre_agg: nn.Module, optional
        Whether to apply the aggregation function before the MLP layers,
        when a dictionary is passed as input.
        Default is MaybeAgg(Concat()).

    Examples
    --------
    >>> mlp = MLPBlock([128, 64], activation=nn.ReLU, normalization="batch_norm", dropout=0.5)
    >>> input_tensor = torch.randn(32, 100)  # batch_size=32, feature_dim=100
    >>> output = mlp(input_tensor)
    >>> print(output.shape)
    torch.Size([32, 64])  # batch_size=32, output_dim=64 (from the last layer of MLP)
    >>> features = {"a": torch.randn(32, 100), "b": torch.randn(32, 100)}
    >>> output = mlp(features)
    torch.Size([32, 64])  # batch_size=32, output_dim=64 (from the last layer of MLP)

    Raises
    ------
    ValueError
        If the normalization parameter is not supported.
    """

    def __init__(
        self,
        units: Sequence[int],
        activation=nn.ReLU,
        normalization=None,
        dropout: Optional[float] = None,
        pre_agg: Optional[nn.Module] = None,
    ):
        modules: List[nn.Module] = [pre_agg or MaybeAgg(Concat())]

        if not isinstance(units, list):
            units = list(units)
        self.out_features = units[-1]

        for dim in units:
            modules.append(nn.LazyLinear(dim))
            if activation is not None:
                modules.append(activation if isinstance(activation, nn.Module) else activation())

            if normalization:
                if normalization == "batchnorm":
                    modules.append(nn.LazyBatchNorm1d())
                elif isinstance(normalization, nn.Module):
                    modules.append(normalization)
                else:
                    raise ValueError(f"Normalization {normalization} not supported")

            if dropout:
                if isinstance(dropout, nn.Module):
                    modules.append(dropout)
                else:
                    modules.append(nn.Dropout(dropout))

        super().__init__(*modules)


class PositionwiseFeedForward(nn.Module):
    """Position-wise Feed-Forward network as proposed in [1]
    References
    ----------
    [1] Position-wise Feed-Forward
    """

    def __init__(
        self,
        embedding_dim: int,
        intermediate_dim: Optional[int] = None,
        bias: bool = False,
        activation=nn.ReLU,
    ):
        super().__init__()

        if intermediate_dim is None:
            hidden_dim = 4 * embedding_dim
            intermediate_dim = int(2 * hidden_dim / 3)
            intermediate_dim = find_multiple(intermediate_dim, 256)

        self.weights_1 = nn.Linear(embedding_dim, intermediate_dim, bias=bias)
        self.weights_2 = nn.Linear(embedding_dim, intermediate_dim, bias=bias)
        self.projection = nn.Linear(intermediate_dim, embedding_dim, bias=bias)
        self.activation = activation if isinstance(activation, nn.Module) else activation()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = self.projection(self.activation(self.weights_1(inputs)) * self.weights_2(inputs))
        return outputs


@output_schema.register(nn.LazyLinear)
@output_schema.register(nn.Linear)
@output_schema.register(MLPBlock)
@output_schema.register(PositionwiseFeedForward)
def _output_schema_block(module: nn.LazyLinear, inputs: Schema):
    return output_schema.tensors(torch.ones((1, module.out_features), dtype=float))
