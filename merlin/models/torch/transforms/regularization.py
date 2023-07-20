import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization [1]
    References
    ----------
    [1] Zhang and Sennrich, Root Mean Square Layer Normalization
        https://arxiv.org/abs/1910.07467
    """

    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        rms = tensor.to(torch.float32).square().mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (tensor * rms).to(tensor.dtype) * self.scale
