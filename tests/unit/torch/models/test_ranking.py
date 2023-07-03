import pytest
import pytorch_lightning as pl

import merlin.models.torch as mm
from merlin.dataloader.torch import Loader
from merlin.models.torch.batch import sample_batch
from merlin.models.torch.utils import module_utils


class TestDCNModel:
    @pytest.mark.parametrize("deep_block", [None, mm.MLPBlock([4, 2])])
    @pytest.mark.parametrize("stacked", [True, False])
    def test_train_dcn_with_lightning_trainer(
        self,
        music_streaming_data,
        deep_block,
        stacked,
        batch_size=16,
        depth=2,
    ):
        schema = music_streaming_data.schema.select_by_name(
            ["item_id", "user_id", "user_age", "item_genres", "click"]
        )
        music_streaming_data.schema = schema

        model = mm.DCNModel(schema, depth=depth, deep_block=deep_block, stacked=stacked)

        trainer = pl.Trainer(max_epochs=1, devices=1)

        with Loader(music_streaming_data, batch_size=batch_size) as train_loader:
            model.initialize(train_loader)
            trainer.fit(model, train_loader)

        assert trainer.logged_metrics["train_loss"] > 0.0

        batch = sample_batch(music_streaming_data, batch_size)
        outputs = module_utils.module_test(model, batch)

        assert len(outputs["click"]) == len(batch.targets["default"])
