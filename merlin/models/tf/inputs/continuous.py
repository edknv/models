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

from typing import List, Optional, Sequence, Union

import tensorflow as tf

from merlin.models.tf.core.base import BlockType
from merlin.models.tf.core.combinators import SequentialBlock
from merlin.models.tf.core.tabular import (
    TABULAR_MODULE_PARAMS_DOCSTRING,
    Filter,
    TabularAggregationType,
    TabularBlock,
)
from merlin.models.utils.doc_utils import docstring_parameter
from merlin.schema import Schema, Tags


@tf.keras.utils.register_keras_serializable(package="merlin.models")
class Continuous(Filter):
    """Filters (keeps) only the continuous features.

    Parameters
    ----------
    inputs : Optional[Union[Sequence[str], Union[Schema, Tags]]], optional
        Indicates how the continuous features should be identified to be filtered.
        It accepts a schema, a column schema tag or a list with the feature names.
        If None (default), it looks for columns with the CONTINUOUS tag in the column schema.
    """

    def __init__(
        self, inputs: Optional[Union[Sequence[str], Union[Schema, Tags]]] = None, **kwargs
    ):
        if inputs is None:
            inputs = Tags.CONTINUOUS
        self.supports_masking = True
        super().__init__(inputs, **kwargs)


def ContinuousProjection(
    schema: Schema,
    projection: tf.keras.layers.Layer,
) -> SequentialBlock:
    """Concatenates the continuous features and combines them
    using a layer

    Parameters
    ----------
    schema : Schema
        Schema that includes the continuous features
    projection : tf.keras.layers.Layer
        Layer that will be used to combine the continuous features
    """
    return SequentialBlock(Continuous(schema, aggregation="concat"), projection)


@docstring_parameter(tabular_module_parameters=TABULAR_MODULE_PARAMS_DOCSTRING)
@tf.keras.utils.register_keras_serializable(package="merlin.models")
class ContinuousFeatures(TabularBlock):
    """Input block for continuous features.

    Parameters
    ----------
    features: List[str]
        List of continuous features to include in this module.
    {tabular_module_parameters}
    """

    def __init__(
        self,
        features: List[str],
        pre: Optional[BlockType] = None,
        post: Optional[BlockType] = None,
        aggregation: Optional[TabularAggregationType] = None,
        schema: Optional[Schema] = None,
        name: Optional[str] = None,
        **kwargs
    ):
        kwargs["is_input"] = kwargs.get("is_input", True)
        super().__init__(
            pre=pre, post=post, aggregation=aggregation, schema=schema, name=name, **kwargs
        )
        self.filter_features = Filter(features)

    @classmethod
    def from_features(cls, features, **kwargs):
        return cls(features, **kwargs)

    def call(self, inputs, *args, **kwargs):
        cont_features = self.filter_features(inputs)
        cont_features = {
            k: tf.expand_dims(v, -1) if len(v.shape) == 1 else v for k, v in cont_features.items()
        }
        return cont_features

    def compute_call_output_shape(self, input_shapes):
        cont_features_sizes = self.filter_features.compute_output_shape(input_shapes)
        cont_features_sizes = {
            k: tf.TensorShape(list(v) + [1]) if len(v) == 1 else v
            for k, v in cont_features_sizes.items()
        }
        return cont_features_sizes

    def get_config(self):
        config = super().get_config()

        config["features"] = self.filter_features.feature_names

        return config

    def _get_name(self):
        return "ContinuousFeatures"

    def repr_ignore(self) -> List[str]:
        return ["filter_features"]

    def repr_extra(self):
        return ", ".join(sorted(self.filter_features.feature_names))
