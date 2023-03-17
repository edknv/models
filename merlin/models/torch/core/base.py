from torch import nn

from merlin.models.torch.utils.torch_utils import apply_module


class Block(nn.Module):
    """
    A custom PyTorch module that applies pre and post processing to the input.

    Parameters
    ----------
    pre : nn.Module, optional
        The module to apply before the main processing.
    post : nn.Module, optional
        The module to apply after the main processing.
    """

    def __init__(self, pre=None, post=None):
        super().__init__()
        self.pre = pre
        self.post = post

    def __call__(self, inputs, *args, **kwargs):
        """
        Apply the pre and post processing modules if available, then call the forward method.

        Parameters
        ----------
        inputs : torch.Tensor
            The input tensor.
        *args : Any
            Additional positional arguments.
        **kwargs : Any
            Additional keyword arguments.

        Returns
        -------
        torch.Tensor
            The output tensor.
        """

        if self.pre is not None:
            inputs = apply_module(self.pre, inputs, *args, **kwargs)
            outputs = super().__call__(inputs)
        else:
            outputs = super().__call__(inputs, *args, **kwargs)

        if self.post is not None:
            outputs = self.post(outputs)

        return outputs

    def forward(self, x):
        return x


class TabularBlock(Block):
    """
    A custom PyTorch module that applies pre and post
    processing and an aggregation function to the input.

    Parameters
    ----------
    pre : nn.Module, optional
        The module to apply before the main processing.
    post : nn.Module, optional
        The module to apply after the main processing.
    aggregation : Callable, optional
        The function to apply on the output tensor.
    """

    def __init__(self, pre=None, post=None, aggregation=None):
        super().__init__(pre=pre, post=post)
        self.aggregation = aggregation

    def __call__(self, inputs, *args, **kwargs):
        """
        Apply the pre and post processing modules if available,
        call the forward method, and apply the aggregation function
        if available.

        Parameters
        ----------
        inputs : torch.Tensor
            The input tensor.
        *args : Any
            Additional positional arguments.
        **kwargs : Any
            Additional keyword arguments.

        Returns
        -------
        torch.Tensor
            The output tensor.
        """

        outputs = super().__call__(inputs, *args, **kwargs)

        if self.aggregation is not None:
            outputs = self.aggregation(outputs)

        return outputs

    def forward(self, x):
        return x


class NoOp(nn.Module):
    def forward(self, inputs):
        return inputs
