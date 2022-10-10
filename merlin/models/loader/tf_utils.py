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

try:
    import horovod.tensorflow as hvd
except ImportError:
    hvd = None


# Pin GPU to be used to process local rank (one GPU per process)
# TODO: Setting the environment variable CUDA_VISIBLE_DEVICES is a last resort
# because tensorflow's functions for setting devices such as
# `tf.config.set_visible_devices` does not seem to work as expected.
# It should be replaced with a better method when we find one.
# This must be executed before importing tensorflow.
if hvd:
    hvd.init()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(hvd.local_rank())

import tensorflow as tf
from packaging import version
from tensorflow.python.feature_column import feature_column_v2 as fc

from merlin.core.dispatch import HAS_GPU
from merlin.models.loader.utils import device_mem_size


def configure_tensorflow(memory_allocation=None, device=None):
    total_gpu_mem_mb = device_mem_size(kind="total", cpu=(not HAS_GPU)) / (1024**2)

    if memory_allocation is None:
        memory_allocation = os.environ.get("TF_MEMORY_ALLOCATION", 0.5)

    if float(memory_allocation) < 1:
        memory_allocation = total_gpu_mem_mb * float(memory_allocation)
    memory_allocation = int(memory_allocation)
    assert memory_allocation < total_gpu_mem_mb

    if device is None:
        device = int(os.environ.get("TF_VISIBLE_DEVICE", 0))
    tf_devices = tf.config.list_physical_devices("GPU")
    if HAS_GPU and len(tf_devices) == 0:
        raise ImportError("TensorFlow is not configured for GPU")
    if HAS_GPU:
        for gpu in tf_devices:
            tf.config.experimental.set_memory_growth(gpu, True)

    # versions using TF earlier than 2.3.0 need to use extension
    # library for dlpack support to avoid memory leak issue
    __TF_DLPACK_STABLE_VERSION = "2.3.0"
    if version.parse(tf.__version__) < version.parse(__TF_DLPACK_STABLE_VERSION):
        try:
            from tfdlpack import from_dlpack
        except ModuleNotFoundError as e:
            message = "If using TensorFlow < 2.3.0, you must install tfdlpack-gpu extension library"
            raise ModuleNotFoundError(message) from e

    else:
        from tensorflow.experimental.dlpack import from_dlpack

    return from_dlpack


def _get_parents(column):
    """
    recursive function for finding the feature columns
    that supply inputs for a given `column`. If there are
    none, returns the column. Uses sets so is not
    deterministic.
    """
    if isinstance(column.parents[0], str):
        return set([column])
    parents = set()
    for parent in column.parents:
        parents |= _get_parents(parent)
    return parents


def get_dataset_schema_from_feature_columns(feature_columns):
    """
    maps from a list of TensorFlow `feature_column`s to
    lists giving the categorical and continuous feature
    names for a dataset. Useful for constructing NVTabular
    Workflows from feature columns
    """
    base_columns = set()
    for column in feature_columns:
        base_columns |= _get_parents(column)

    cat_names, cont_names = [], []
    for column in base_columns:
        if isinstance(column, fc.CategoricalColumn):
            cat_names.append(column.name)
        else:
            cont_names.append(column.name)

    cat_names = sorted(cat_names)
    cont_names = sorted(cont_names)
    return cat_names, cont_names
