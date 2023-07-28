import os
import sys
import requests
from pathlib import Path
import zipfile
from typing import Optional, Union, Tuple
import sqlite3
import re
import json

import numpy as np
import pandas as pd

import merlin.io
from merlin.datasets import BASE_PATH
from merlin.core.dispatch import get_lib
from merlin.models.tokenizers import Tokenizer
from merlin.models.tokenizers.sentencepiece import SentencePieceTokenizer, require_sentencepiece
from merlin.models.utils.nvt_utils import require_nvt
from merlin.schema import ColumnSchema, Schema
import merlin.dtypes as md


GDRIVE_FILE_ID = "1TqleXec_OykOYFREKKtschzY29dUcVAQ"
FILE_NAME = "spider.zip"
DEFUALT_TOKENIZER_PATH = "llama/tokenizer.model"


def get_spider(
    path: Optional[Union[str, Path]] = None,
    overwrite: bool = False,
    tokenizer: Optional[Tokenizer] = None,
    transformed_name: str = "transformed",
    # nvt_workflow: Optional[Workflow] = None,
    **kwargs,
) -> Tuple[merlin.io.Dataset, merlin.io.Dataset]:
    """ """
    require_nvt()

    if tokenizer is None:
        tokenizer = load_sentencepiece_tokenizer()

    if path is None:
        path = Path(BASE_PATH) / "spider"
    else:
        path = Path(path)

    download_spider(path)

    spider_path = path / "spider"

    #preprocess_spider_tables(spider_path)
    preprocess_spider_questions(spider_path)

    raw_path = path
    nvt_path = raw_path / transformed_name
    # transform_spider(raw_path, nvt_path, tokenizer=tokenizer, nvt_workflow=nvt_workflow)


def download_spider(path: Path):
    """Automatically download the spider dataset.

    Parameters
    ----------
    path (Path): Output path.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    file_path = path / FILE_NAME
    if file_path.exists():
        return

    local_filename = str(file_path)
    download_file_from_google_drive(GDRIVE_FILE_ID, local_filename)

    with zipfile.ZipFile(local_filename, "r") as zip_ref:
        zip_ref.extractall(path)


def preprocess_spider_questions(path: Path):
    path = Path(path)

    train_spider_json = path / "train_spider.json"
    train_others_json = path / "train_others.json"

    with open(train_spider_json) as f:
        train_spider = json.load(f)
    with open(train_others_json) as f:
        train_others = json.load(f)

    train = train_spider + train_others

    for _, entry in enumerate(train):
        query = entry["query"]
        db_name = entry["db_id"]
        conn = _load_sql_database(path, db_name)
        answer = pd.read_sql_query(query, conn)
        entry["answer"] = answer.loc[0].values[-1]
        print(answer)
        print(entry["answer"])
        if _ > 5:
            break

    return train


def _load_sql_database(path: Path, db_name: str):
    path = Path(path)
    db_path = path / "database" / db_name
    conn = sqlite3.connect(db_path / f"{db_name}.sqlite")
    conn.text_factory = lambda b: b.decode(errors="ignore")
    return conn


def preprocess_spider_tables(path: Path) -> None:
    path = Path(path)

    databases = [f.name for f in os.scandir(path / "database") if f.is_dir()]

    for db in databases:
        db_path = path / "database" / db
        conn = sqlite3.connect(db_path / f"{db}.sqlite")
        conn.text_factory = lambda b: b.decode(errors="ignore")
        table_names = pd.read_sql_query(
            "SELECT tbl_name FROM sqlite_master where type='table';", conn
        )
        table_names = table_names["tbl_name"].values.tolist()
        for table in table_names:
            parquet_path = db_path / f"{table}.parquet"
            if parquet_path.exists():
                continue

            df = pd.read_sql_query(f"SELECT * FROM {table}", conn)
            dtypes = pd.read_sql_query(f"SELECT name, type FROM pragma_table_info('{table}')", conn)

            schema = Schema()
            for _, row in dtypes.iterrows():
                col_name = row["name"]
                dtype = row["type"].lower()
                if any(dtype.startswith(s) for s in ["varchar", "char", "text"]):
                    dtype = md.string
                elif any(dtype.startswith(s) for s in ["int", "number", "mediumint"]):
                    dtype = md.int32
                elif dtype.startswith("bigint"):
                    dtype = md.int64
                elif dtype in ["smallint", "year"]:
                    dtype = md.int16
                elif dtype == "smallint unsigned":
                    dtype = md.uint16
                elif dtype == "tinyint unsigned":
                    dtype = md.uint8
                elif dtype in ["date", "datetime", "timestamp"]:
                    dtype = md.datetime64
                elif any(dtype.startswith(s) for s in ["decimal", "float", "real", "numeric"]):
                    dtype = md.float32
                elif dtype == "double":
                    dtype = md.float64
                elif dtype in ["bool", "boolean", "bit"]:
                    dtype = md.boolean
                else:
                    dtype = md.unknown
                col_schema = ColumnSchema(name=col_name, dtype=dtype)
                schema += Schema([col_schema])

            df = df.replace(r"^\s*$", np.nan, regex=True)
            df = df.replace("NULL", np.nan)
            df = df.replace("inf", np.nan)
            df = df.replace("nil", np.nan)

            if db == "bike_1":
                # schema says the zip_code column is an integer column, but
                # there are some strings with zip+4 format.
                if col_name == "zip_code":
                    df[col_name] = df[col_name].apply(
                        lambda x: int(x.split("-", 1)[0]) if isinstance(x, str) else x
                    )
                # A bad value somewhere in the bike_1 dataset.
                df = df.replace(r"^T$", np.nan, regex=True)

            dataset = merlin.io.Dataset(df, schema=schema, engine="parquet")
            dataset.to_parquet(parquet_path)


def transform_spider(
    data: Union[str, Path, Tuple[merlin.io.Dataset, merlin.io.Dataset]],
    output_path: Union[str, Path],
    tokenizer: Tokenizer,
    nvt_workflow=None,
    **kwargs,
) -> None:
    nvt_workflow = nvt_workflow or default_spider_transformation()
    workflow_fit_transform(nvt_workflow, _train, _valid, str(output_path), **kwargs)


def default_spider_transformation() -> None:
    ...


def load_sentencepiece_tokenizer() -> SentencePieceTokenizer:
    require_sentencepiece()
    from sentencepiece import SentencePieceProcessor

    processor = SentencePieceProcessor(model_file="../llama/tokenizer.model")
    tokenizer = SentencePieceTokenizer(processor=processor)

    return tokenizer


def download_file_from_google_drive(file_id, destination):
    URL = "https://docs.google.com/uc?export=download&confirm=1"

    session = requests.Session()

    response = session.get(URL, params={"id": file_id}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {"id": file_id, "confirm": token}
        response = session.get(URL, params=params, stream=True)

    save_response_content(response, destination)


def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith("download_warning"):
            return value

    return None


def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)


def main():
    get_spider()


if __name__ == "__main__":
    main()
