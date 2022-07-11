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

import pytest
import tensorflow as tf

import merlin.models.tf as ml
from merlin.io import Dataset
from merlin.models.config.schema import FeatureCollection
from merlin.models.tf import FeatureContext
from merlin.models.tf.blocks.core.base import PredictionOutput
from merlin.schema import Tags


@pytest.mark.parametrize("ignore_last_batch_on_sample", [True, False])
def test_item_retrieval_scorer(ignore_last_batch_on_sample):
    batch_size = 10
    inbatch_sampler = ml.InBatchSampler()

    item_retrieval_scorer = ml.ItemRetrievalScorer(
        samplers=[inbatch_sampler],
        sampling_downscore_false_negatives=False,
        context=ml.ModelContext(),
    )

    users_embeddings = tf.random.uniform(shape=(batch_size, 5), dtype=tf.float32)
    items_embeddings = tf.random.uniform(shape=(batch_size, 5), dtype=tf.float32)
    positive_items = tf.random.uniform(shape=(10,), minval=1, maxval=100, dtype=tf.int32)

    # First batch
    output_scores1 = item_retrieval_scorer.call_outputs(
        PredictionOutput(
            {
                "query": users_embeddings,
                "item": items_embeddings,
            },
            targets=positive_items,
        ),
        training=True,
    ).predictions
    expected_num_samples_inbatch = batch_size + 1
    tf.assert_equal(tf.shape(output_scores1)[0], batch_size)
    # Number of negatives plus one positive
    tf.assert_equal(tf.shape(output_scores1)[1], expected_num_samples_inbatch)


""" @tf.function convert `call_outputs` of the ItemRetrievalScorer to graph ops,
In graph-model the exception is not raised.
or this test, we need to be able track the exceptions in graph mode.

def test_item_retrieval_scorer_cached_sampler_no_result_first_batch():
    batch_size = 10

    # CachedCrossBatchSampler is the only sampler here and with ignore_last_batch_on_sample=True
    # for the first batch no sample will be returned, which should raise an exception
    cached_batches_sampler = ml.CachedCrossBatchSampler(
        capacity=batch_size * 2,
        ignore_last_batch_on_sample=True,
    )

    item_retrieval_scorer = ml.ItemRetrievalScorer(
        samplers=[cached_batches_sampler], sampling_downscore_false_negatives=False
    )

    users_embeddings = tf.random.uniform(shape=(batch_size, 5), dtype=tf.float32)
    items_embeddings = tf.random.uniform(shape=(batch_size, 5), dtype=tf.float32)

    with pytest.raises(Exception) as excinfo:
        _ = item_retrieval_scorer({"query": users_embeddings, "item": items_embeddings}, {})
    assert "No negative items where sampled from samplers" in str(excinfo.value)
"""


def test_item_retrieval_scorer_no_sampler():
    with pytest.raises(Exception) as excinfo:
        users_embeddings = tf.random.uniform(shape=(10, 5), dtype=tf.float32)
        items_embeddings = tf.random.uniform(shape=(10, 5), dtype=tf.float32)
        positive_items = tf.random.uniform(shape=(10,), minval=1, maxval=100, dtype=tf.int32)
        item_retrieval_scorer = ml.ItemRetrievalScorer(
            samplers=[], sampling_downscore_false_negatives=False, context=ml.ModelContext()
        )
        item_retrieval_scorer.call_outputs(
            PredictionOutput(
                {"query": users_embeddings, "item": items_embeddings}, targets=positive_items
            ),
            training=True,
        )
    assert "At least one sampler is required by ItemRetrievalScorer for negative sampling" in str(
        excinfo.value
    )


def test_item_retrieval_scorer_downscore_false_negatives():
    batch_size = 10

    cached_batches_sampler = ml.InBatchSampler()

    # Adding item id to the context
    item_ids = tf.random.uniform(shape=(batch_size,), minval=1, maxval=10000000, dtype=tf.int32)
    features = FeatureCollection(None, {"item_id": item_ids})
    feature_context = FeatureContext(features)

    FALSE_NEGATIVE_SCORE = -100_000_000.0
    item_retrieval_scorer = ml.ItemRetrievalScorer(
        samplers=[cached_batches_sampler],
        sampling_downscore_false_negatives=True,
        sampling_downscore_false_negatives_value=FALSE_NEGATIVE_SCORE,
    )

    users_embeddings = tf.random.uniform(shape=(batch_size, 5), dtype=tf.float32)
    items_embeddings = tf.random.uniform(shape=(batch_size, 5), dtype=tf.float32)

    outputs = item_retrieval_scorer.call_outputs(
        PredictionOutput({"query": users_embeddings, "item": items_embeddings}, {}),
        training=True,
        feature_context=feature_context,
    )
    output_scores = outputs.predictions

    output_neg_scores = output_scores[:, 1:]

    diag_mask = tf.eye(tf.shape(output_neg_scores)[0], dtype=tf.bool)
    tf.assert_equal(output_neg_scores[diag_mask], FALSE_NEGATIVE_SCORE)
    tf.assert_equal(
        tf.reduce_all(
            tf.not_equal(
                output_neg_scores[tf.math.logical_not(diag_mask)],
                tf.constant(FALSE_NEGATIVE_SCORE, dtype=output_neg_scores.dtype),
            )
        ),
        True,
    )


def test_item_retrieval_scorer_only_positive_when_not_training():
    batch_size = 10

    item_retrieval_scorer = ml.ItemRetrievalScorer(
        samplers=[ml.InBatchSampler()],
        sampling_downscore_false_negatives=False,
        context=ml.ModelContext(),
    )

    users_embeddings = tf.random.uniform(shape=(batch_size, 5), dtype=tf.float32)
    items_embeddings = tf.random.uniform(shape=(batch_size, 5), dtype=tf.float32)

    # Calls with training=False, so that only the positive item is scored
    output_scores = item_retrieval_scorer(
        {"query": users_embeddings, "item": items_embeddings}, training=False
    )
    tf.assert_equal(
        (int(tf.shape(output_scores)[0]), int(tf.shape(output_scores)[1])), (batch_size, 1)
    )


@pytest.mark.parametrize("run_eagerly", [True])
@pytest.mark.parametrize("weight_tying", [True, False])
@pytest.mark.parametrize("sampled_softmax", [True, False])
def test_last_item_prediction_task(
    sequence_testing_data: Dataset,
    run_eagerly: bool,
    weight_tying: bool,
    sampled_softmax: bool,
):
    inputs = ml.InputBlock(
        sequence_testing_data.schema,
        aggregation="concat",
        seq=False,
        max_seq_length=4,
        masking="clm",
        split_sparse=True,
    )
    loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    task = ml.NextItemPredictionTask(
        schema=sequence_testing_data.schema,
        masking=True,
        weight_tying=weight_tying,
        sampled_softmax=sampled_softmax,
        logits_temperature=0.5,
    )

    model = ml.Model(inputs, ml.MLPBlock([64]), task)
    model.compile(optimizer="adam", run_eagerly=run_eagerly, loss=loss)
    losses = model.fit(sequence_testing_data, batch_size=50, epochs=2)

    assert len(losses.epoch) == 2
    for metric in losses.history.keys():
        assert type(losses.history[metric]) is list

    batch = ml.sample_batch(
        sequence_testing_data, batch_size=50, include_targets=False, to_dense=True
    )
    out = model({k: tf.cast(v, tf.int64) for k, v in batch.items()})
    assert out.shape[-1] == 51997


@pytest.mark.parametrize("run_eagerly", [True, False])
@pytest.mark.parametrize("ignore_last_batch_on_sample", [True, False])
def test_retrieval_task_inbatch_default_sampler(
    music_streaming_data: Dataset, run_eagerly, ignore_last_batch_on_sample
):
    batch_size = 100
    music_streaming_data.schema = music_streaming_data.schema.remove_by_tag(Tags.TARGET)
    two_tower = ml.TwoTowerBlock(music_streaming_data.schema, query_tower=ml.MLPBlock([512, 256]))

    batch_inputs = ml.sample_batch(
        music_streaming_data, batch_size=batch_size, include_targets=False
    )

    model = ml.Model(
        two_tower,
        ml.ItemRetrievalTask(music_streaming_data.schema, logits_temperature=2),
    )

    model.compile(optimizer="adam", run_eagerly=run_eagerly, loss="bpr")

    for _ in range(1, 4):
        output = model(batch_inputs, training=True)
        features = FeatureCollection(model.schema, model.as_dense(batch_inputs))
        feature_context = FeatureContext(features)
        output = (
            model.prediction_tasks[0]
            .pre.call_outputs(
                PredictionOutput(output, {}), training=True, feature_context=feature_context
            )
            .predictions
        )
        expected_num_samples_inbatch = batch_size

        tf.assert_equal(tf.shape(output)[0], batch_size)
        # Number of negatives plus one positive
        tf.assert_equal(tf.shape(output)[1], expected_num_samples_inbatch + 1)
