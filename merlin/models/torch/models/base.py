#
# Copyright (c) 2023, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
from typing import Dict, List, Optional, Tuple, Union

import pytorch_lightning as pl
import torch
from torch import nn

from merlin.dataloader.torch import Loader
from merlin.io import Dataset
from merlin.models.torch.batch import Batch, sample_batch
from merlin.schema import Schema
from merlin.models.torch.outputs.base import ModelOutput


class Model(pl.LightningModule):
    """Merlin Model class

    Parameters
    ----------
    """
    def __init__(
        self,
        *blocks: nn.Module,
        schema: Optional[Schema] = None,
        pre=None,
        post=None,
        optimizer=torch.optim.Adam,
    ):
        """Initializes `Model` class"""
        super().__init__()
        self.schema = schema
        self.blocks = nn.ModuleList(blocks)
        self.optimizer = optimizer

    def initialize(self, data: Loader):
        return initialize(self, data)

    def forward(self, inputs: Union[torch.Tensor, Dict[str, torch.Tensor]], batch: Optional[Batch] = None):
        outputs = inputs
        for block in self.blocks:
            outputs = block(outputs, batch=batch)
        return outputs

    def training_step(self, batch, batch_idx):
        del batch_idx
        inputs, targets = batch
        predictions = self(inputs)
        loss = compute_loss(predictions, targets)
        return loss

    def configure_optimizers(self):
        return self.optimizer(self.parameters())

    def first(self) -> nn.Module:
        return self.blocks[0]

    def last(self) -> nn.Module:
        return self.blocks[-1]

    def input_schema(self) -> Schema:
        if self.schema:
            return self.schema

        input_schemas = []
        for child in module_utils.get_all_children(self):
            if hasattr(child, "input_schema"):
                input_schemas.append(child.input_schema)

        if not input_schemas:
            raise ValueError("No input schema found")

        return reduce(lambda a, b: a + b, input_schemas)

    def output_schema(self) -> Schema:
        output_schemas = []
        for child in module_utils.get_all_children(self):
            if hasattr(child, "output_schema"):
                output_schemas.append(child.output_schema)

        if not output_schemas:
            raise ValueError("No output schema found")

        return reduce(lambda a, b: a + b, output_schemas)


def compute_loss(
    predictions: Union[torch.Tensor, Dict[str, torch.Tensor]],
    targets: Union[torch.Tensor, Dict[str, torch.Tensor]],
    model_outputs: Tuple[ModelOutput],
) -> torch.Tensor:
    """
    Update targets with model_out.target for each model_out
    """
    if isinstance(predictions, torch.Tensor) and isinstance(targets, torch.Tensor):
        if len(model_outputs) != 1:
            raise RuntimeError("Multiple outputs but only one target was provided.")
        return model_outputs[0].loss(predictions, targets)

    loss = torch.tensor(0.0)
    for model_out in model_outputs:
        name = model_out.output_schema.first.name
        # TODO: How to handle task weights?
        loss += model_out.loss(predictions[name], targets[name])

    return loss / len(model_outputs)


def compute_metrics(
    predictions: Union[torch.Tensor, Dict[str, torch.Tensor]],
    targets: Union[torch.Tensor, Dict[str, torch.Tensor]],
    model_outputs: Tuple[ModelOutput],
) -> Dict[str, Tuple[torch.Tensor, ...]]:
    """ """
    if isinstance(predictions, torch.Tensor) and isinstance(targets, torch.Tensor):
        if len(model_outputs) != 1:
            raise RuntimeError("Multiple outputs but only one target was provided.")
        return tuple(tuple(m(predictions, targets) for m in model_outputs[0].metrics))

    metrics = {}
    for model_out in model_outputs:
        curr_metrics = []
        name = model_out.output_schema.first.name
        for curr_metric in model_out.metrics:
            curr_metrics.append(curr_metric(predictions[name], targets[name]))
        metrics[name] = tuple(curr_metrics)

    return metrics


def initialize(module, data: Loader):
    if isinstance(data, (Loader, Dataset)):
        module.double()  # TODO: Put in data-loader PR to standardize on float-32
        batch = sample_batch(data, batch_size=1, shuffle=False)
    else:
        batch = data

    #module.to(get_device(batch))
    return module(batch)


def get_device(data):
    if isinstance(data, torch.Tensor):
        device = data.device
    elif isinstance(data, tuple):
        device = data[0].device
    elif isinstance(data, dict):
        for d in data.values():
            if isinstance(d, torch.Tensor):
                device = d.device
                break
    else:
        raise ValueError(f"Unsupported data type {type(data)}")

    return device
