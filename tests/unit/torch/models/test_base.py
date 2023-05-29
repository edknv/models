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
import pytorch_lightning as pl
import pytest
import torch
from torch import nn

from merlin.dataloader.torch import Loader
import merlin.models.torch as mm
from merlin.models.torch.utils import module_utils
from merlin.models.torch.models.base import compute_loss, compute_metrics
from merlin.schema import ColumnSchema, Schema


class TestModel:
    def test_init_default(self):
        model = mm.Model(mm.Block(), mm.Block())

        assert isinstance(model, mm.Model)
        assert len(model.blocks) == 2
        assert model.schema is None
        #assert model.pre is None
        #assert model.post is None
        assert model.optimizer is torch.optim.Adam

    def test_init_schema(self):
        schema = Schema([ColumnSchema("foo")])
        model = mm.Model(mm.Block(), mm.Block(), schema=schema)

        assert len(model.blocks) == 2
        assert model.schema.first.name == "foo"

    def test_identity(self):
        model = mm.Model(mm.Block(), mm.Block())
        inputs = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

        outputs = module_utils.module_test(model.to_torchscript(), inputs)

        assert torch.equal(inputs, outputs)

    def test_first(self):
        model = mm.Model(mm.Block(name="a"), mm.Block(name="b"), mm.Block(name="c"))
        assert model.first()._name == "a"

    def test_last(self):
        model = mm.Model(mm.Block(name="a"), mm.Block(name="b"), mm.Block(name="c"))
        assert model.last()._name == "c"

    #def test_train_classification(self, music_streaming_data):
    #    schema = music_streaming_data.schema.without(["user_genres", "like", "item_genres"])
    #    music_streaming_data.schema = schema
    #    click_column = schema.select_by_name("click").first

    #    model = mm.Model(
    #        #mm.ParallelBlock(),
    #        mm.Concat(),
    #        mm.BinaryOutput(click_column),
    #        schema=schema,
    #    )

    #    trainer = pl.Trainer(max_epochs=1)

    #    with Loader(
    #        music_streaming_data,
    #        batch_size=16,
    #        shuffle=False,
    #    ) as loader:
    #        model.initialize(loader)
    #        trainer.fit(model, loader)


class TestComputeLoss:
    def test_tensor_inputs(self):
        predictions = torch.randn(2, 1)
        targets = torch.randint(2, (2, 1), dtype=torch.float32)
        model_outputs = [mm.BinaryOutput()]
        loss = compute_loss(predictions, targets, model_outputs)
        expected = nn.BCEWithLogitsLoss()(predictions, targets)
    
        assert isinstance(loss, torch.Tensor)
        assert torch.allclose(loss, expected)
    
    
    def test_dict_inputs(self):
        predictions = {"a": torch.randn(2, 1)}
        targets = {"a": torch.randint(2, (2, 1), dtype=torch.float32)}
        model_outputs = (mm.BinaryOutput(ColumnSchema("a")), )
        loss = compute_loss(predictions, targets, model_outputs)
        expected = nn.BCEWithLogitsLoss()(predictions["a"], targets["a"])
    
        assert isinstance(loss, torch.Tensor)
        assert torch.allclose(loss, expected)
    
    
    def test_mixed_inputs(self):
        predictions = {"a": torch.randn(2, 1)}
        targets = torch.randint(2, (2, 1), dtype=torch.float32)
        model_outputs = (mm.BinaryOutput(ColumnSchema("a")), )
        loss = compute_loss(predictions, targets, model_outputs)
        expected = nn.BCEWithLogitsLoss()(predictions["a"], targets)
    
        assert isinstance(loss, torch.Tensor)
        assert torch.allclose(loss, expected)
    
    
    def test_multiple_outputs_raises_error(self):
        predictions = torch.randn(2, 1)
        targets = torch.randint(2, (2, 1), dtype=torch.float32)
        model_outputs = (
            mm.BinaryOutput(ColumnSchema("a")),
            mm.BinaryOutput(ColumnSchema("b")),
        )
    
        with pytest.raises(RuntimeError):
            loss = compute_loss(predictions, targets, model_outputs)


def test_compute_loss_single_model_output():
    predictions = {"foo": torch.randn(2, 1)}
    targets = {"foo": torch.randint(2, (2, 1), dtype=torch.float32)}
    model_outputs = [mm.BinaryOutput(ColumnSchema("foo"))]
    loss = compute_loss(predictions, targets, model_outputs)
    expected = nn.BCEWithLogitsLoss()(predictions["foo"], targets["foo"])

    assert isinstance(loss, torch.Tensor)
    assert torch.allclose(loss, expected)
