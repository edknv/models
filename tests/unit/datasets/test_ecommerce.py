import os
from pathlib import Path

import pytest

import merlin.io
from merlin.datasets import ecommerce
from merlin.datasets.synthetic import generate_data

# This is the path to the raw ali-ccp dataset
MAYBE_ALICCP_DATA = os.environ.get("DATA_PATH_ALICCP", None)

MAYBE_DATA_DIR = os.environ.get("INPUT_DATA_DIR", None)


def test_synthetic_aliccp_data():
    dataset = generate_data("aliccp", 100)

    assert isinstance(dataset, merlin.io.Dataset)
    assert dataset.num_rows == 100
    assert len(dataset.schema) == 27
    assert dataset.compute()["click"].sum() > 0


def test_synthetic_aliccp_raw_data(tmp_path):
    dataset = generate_data("aliccp-raw", 100)

    assert isinstance(dataset, merlin.io.Dataset)
    assert dataset.num_rows == 100
    assert len(dataset.schema) == 25
    assert sorted(dataset.to_ddf().compute().columns) == [
        "click",
        "conversion",
        "item_brand",
        "item_category",
        "item_id",
        "item_intention",
        "item_shop",
        "position",
        "user_age",
        "user_brands",
        "user_categories",
        "user_consumption_1",
        "user_consumption_2",
        "user_gender",
        "user_geography",
        "user_group",
        "user_id",
        "user_intentions",
        "user_is_occupied",
        "user_item_brands",
        "user_item_categories",
        "user_item_intentions",
        "user_item_shops",
        "user_profile",
        "user_shops",
    ]

    ecommerce.transform_aliccp((dataset, dataset), tmp_path)

    output_files = set(sorted(tmp_path.glob("*/*")))
    assert {
        Path(f"{tmp_path}/train/.merlin"),
        Path(f"{tmp_path}/train/_file_list.txt"),
        Path(f"{tmp_path}/train/_metadata"),
        Path(f"{tmp_path}/train/_metadata.json"),
        Path(f"{tmp_path}/train/part_0.parquet"),
        Path(f"{tmp_path}/train/schema.pbtxt"),
        Path(f"{tmp_path}/valid/.merlin"),
        Path(f"{tmp_path}/valid/_file_list.txt"),
        Path(f"{tmp_path}/valid/_metadata"),
        Path(f"{tmp_path}/valid/_metadata.json"),
        Path(f"{tmp_path}/valid/part_0.parquet"),
        Path(f"{tmp_path}/valid/schema.pbtxt"),
        Path(f"{tmp_path}/workflow/categories"),
        Path(f"{tmp_path}/workflow/metadata.json"),
        Path(f"{tmp_path}/workflow/workflow.pkl"),
    }.issubset(output_files)

    metadata_files = tmp_path.glob("*/*.merlin/*")
    assert sorted(metadata_files) == [
        Path(f"{tmp_path}/train/.merlin/schema.json"),
        Path(f"{tmp_path}/valid/.merlin/schema.json"),
    ]


@pytest.mark.skipif(
    MAYBE_ALICCP_DATA is None,
    reason="ALI-CCP data is not available, pass it through env variable $DATA_PATH_ALICCP",
)
def test_get_alliccp():
    data_path = MAYBE_ALICCP_DATA

    nvt_workflow = ecommerce.default_aliccp_transformation(add_target_encoding=False)
    train, valid = ecommerce.get_aliccp(
        data_path, nvt_workflow=nvt_workflow, transformed_name="raw_transform", overwrite=True
    )

    assert isinstance(train, merlin.io.Dataset)
    assert isinstance(valid, merlin.io.Dataset)


@pytest.mark.skipif(
    MAYBE_ALICCP_DATA is None,
    reason="ALI-CCP data is not available, pass it through env variable $DATA_PATH_ALICCP",
)
def test_prepare_alliccp(tmp_path):
    data_path = MAYBE_ALICCP_DATA

    ecommerce.prepare_alliccp(data_path, file_size=50, max_num_rows=100, output_dir=tmp_path)
    output_files = list(tmp_path.glob("*/*"))

    assert len(output_files) == 2
    assert all(f.name.endswith(".parquet") for f in output_files)


@pytest.mark.skipif(
    MAYBE_ALICCP_DATA is None,
    reason="ALI-CCP data is not available, pass it through env variable $DATA_PATH_ALICCP",
)
def test_transform_alliccp(tmp_path):
    data_path = MAYBE_ALICCP_DATA

    ecommerce.transform_aliccp(data_path, tmp_path)
    output_files = list(tmp_path.glob("*/*"))

    assert len(output_files) == 10


def test_synthetic_booking_data():
    dataset = generate_data("booking.com", 100)

    assert isinstance(dataset, merlin.io.Dataset)
    assert dataset.num_rows == 100
    assert len(dataset.schema) == 11


@pytest.mark.skipif(
    MAYBE_DATA_DIR is None,
    reason="No data-dir available, pass it through env variable $INPUT_DATA_DIR",
)
def test_get_booking():
    data_path = os.path.join(MAYBE_DATA_DIR, "booking_test")

    train, valid = ecommerce.get_booking(data_path, overwrite=True)

    assert isinstance(train, merlin.io.Dataset)
    assert isinstance(valid, merlin.io.Dataset)
