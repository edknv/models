import os

import cupy
import pytest
import pytorch_lightning as pl
from lightning_fabric import Fabric

import merlin.models.torch as mm
from merlin.loader.torch import Loader


class TestMultiGPU:
    @pytest.mark.multigpu
    def test_multi_gpu(self, music_streaming_data):
        fabric = Fabric(devices=2, strategy="ddp")
        fabric.launch()
        # Fabric creates the LOCAL_RANK environment variable.
        rank = int(os.environ["LOCAL_RANK"])

        schema = music_streaming_data.schema
        data = music_streaming_data.repartition(2)

        model = mm.Model(
            mm.TabularInputBlock(schema, init="defaults"),
            mm.MLPBlock([5]),
            mm.BinaryOutput(schema["click"]),
        )
        model.initialize(data)

        trainer = pl.Trainer(max_epochs=3, devices=2)
        loader = Loader(
            data,
            batch_size=2,
            shuffle=False,
            global_size=2,
            global_rank=int(os.environ["LOCAL_RANK"]),
        )
        trainer.fit(model, loader)

        # 100 rows total / 2 devices -> 50 rows per device
        # 50 rows / 2 per batch -> 25 steps per device
        assert trainer.num_training_batches == 25
