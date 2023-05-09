#
# Copyright (c) 2021, NVIDIA CORPORATION.
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
import os

import numpy as np
import pytest
import tensorflow as tf

import merlin.models.tf as ml
from merlin.core.compat import cudf
from merlin.datasets.synthetic import generate_data
from merlin.io import Dataset
from merlin.models.tf.core.aggregation import ElementWiseMultiply
from merlin.models.tf.utils import testing_utils
from merlin.schema import Tags


def test_matrix_factorization_block(music_streaming_data: Dataset):
    mf = ml.QueryItemIdsEmbeddingsBlock(music_streaming_data.schema, dim=128)

    outputs = mf(ml.sample_batch(music_streaming_data, batch_size=100, include_targets=False))

    assert "query" in outputs
    assert "item" in outputs


def test_matrix_factorization_embedding_export(music_streaming_data: Dataset, tmp_path):
    import pandas as pd

    from merlin.models.tf.core.aggregation import CosineSimilarity

    mf = ml.MatrixFactorizationBlock(
        music_streaming_data.schema, dim=128, aggregation=CosineSimilarity()
    )
    mf = ml.MatrixFactorizationBlock(music_streaming_data.schema, dim=128, aggregation="cosine")
    model = ml.Model(mf, ml.BinaryClassificationTask("like"))
    model.compile(optimizer="adam")

    model.fit(music_streaming_data, batch_size=50, epochs=5)

    item_embedding_parquet = str(tmp_path / "items.parquet")
    mf.export_embedding_table(Tags.ITEM_ID, item_embedding_parquet, gpu=False)

    df = mf.embedding_table_df(Tags.ITEM_ID, gpu=False)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 10001
    assert os.path.exists(item_embedding_parquet)

    # Test GPU export if available
    if cudf:
        user_embedding_parquet = str(tmp_path / "users.parquet")
        mf.export_embedding_table(Tags.USER_ID, user_embedding_parquet, gpu=True)
        assert os.path.exists(user_embedding_parquet)
        df = mf.embedding_table_df(Tags.USER_ID, gpu=True)
        assert isinstance(df, cudf.DataFrame)
        assert len(df) == 10001


def test_elementwisemultiply():
    emb1 = np.random.uniform(-1, 1, size=(5, 10))
    emb2 = np.random.uniform(-1, 1, size=(5, 10))
    x = ElementWiseMultiply()({"emb1": tf.constant(emb1), "emb2": tf.constant(emb2)})

    assert np.mean(np.isclose(x.numpy(), np.multiply(emb1, emb2))) == 1
    assert x.numpy().shape == (5, 10)
