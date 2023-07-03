from typing import Optional

from merlin.models.torch.block import Block, ParallelBlock
from merlin.models.torch.blocks.cross import CrossBlock
from merlin.models.torch.blocks.mlp import MLPBlock
from merlin.models.torch.inputs.tabular import TabularInputBlock
from merlin.models.torch.models.base import Model
from merlin.models.torch.outputs.tabular import TabularOutputBlock
from merlin.models.torch.transforms.agg import Concat, MaybeAgg
from merlin.schema import Schema


def DCNModel(
    schema: Schema,
    depth: int = 1,
    deep_block: Optional[Block] = None,
    stacked: bool = True,
    input_block: Optional[Block] = None,
    output_block: Optional[Block] = None,
) -> Model:
    """
    The Deep & Cross Network (DCN) architecture as proposed in Wang, et al. [1]

    Parameters
    ----------
    schema : Schema
        The schema to use for selection.
    depth : int, optional
        Number of cross-layers to be stacked, by default 1
    deep_block : Block, optional
        The `Block` to use as the deep part of the model (typically a `MLPBlock`)
    stacked : bool
        Whether to use the stacked version of the model or the parallel version.
    input_block : Block, optional
        The `Block` to use as the input layer. If None, a default `TabularInputBlock` object
        is instantiated, that creates the embedding tables for the categorical features
        based on the schema. The embedding dimensions are inferred from the features
        cardinality. For a custom representation of input data you can instantiate
        and provide a `TabularInputBlock` instance.

    Returns
    -------
    Model
        An instance of Model class representing the fully formed DCN.

    Example usage
    -------------
    >>> model = mm.DCNModel(
    ...    schema,
    ...    depth=2,
    ...    deep_block=mm.MLPBlock([256, 64]),
    ...    output_block=BinaryOutput(ColumnSchema("target")))
    >>> trainer = pl.Trainer()
    >>> model.initialize(dataloader)
    >>> trainer.fit(model, dataloader)

    References
    ----------
    [1]. Wang, Ruoxi, et al. "DCN V2: Improved deep & cross network and
       practical lessons for web-scale learning to rank systems." Proceedings
       of the Web Conference 2021. 2021. https://arxiv.org/pdf/2008.13535.pdf
    """
    if deep_block is None:
        deep_block = MLPBlock([512, 256])

    if input_block is None:
        input_block = TabularInputBlock(schema, init="defaults")

    if output_block is None:
        output_block = TabularOutputBlock(schema, init="defaults")

    if stacked:
        model = Model(input_block, CrossBlock(depth), deep_block, output_block)
    else:
        model = Model(
            input_block,
            ParallelBlock({"cross": CrossBlock(depth), "deep": deep_block}).append(
                MaybeAgg(Concat())
            ),
            output_block,
        )

    return model
