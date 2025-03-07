import shutil

import pytest
from testbook import testbook

from tests.conftest import REPO_ROOT

pytest.importorskip("transformers")
utils = pytest.importorskip("merlin.systems.triton.utils")

TRITON_SERVER_PATH = shutil.which("tritonserver")


@pytest.mark.skipif(not TRITON_SERVER_PATH, reason="triton server not found")
@testbook(
    REPO_ROOT / "examples/usecases/transformers-next-item-prediction.ipynb",
    timeout=720,
    execute=False,
)
@pytest.mark.notebook
def test_next_item_prediction(tb, tmpdir):
    tb.inject(
        f"""
        import os, random
        from datetime import datetime, timedelta
        from merlin.datasets.synthetic import generate_data
        ds = generate_data('booking.com-raw', 10000)
        df = ds.compute()
        def generate_date():
            date = datetime.today()
            if random.randint(0, 1):
                date -= timedelta(days=7)
            return date
        df['checkin'] = [generate_date() for _ in range(df.shape[0])]
        df['checkout'] = [generate_date() for _ in range(df.shape[0])]
        df.to_csv('{tmpdir}/train_set.csv')
        """
    )
    tb.cells[4].source = tb.cells[4].source.replace("get_booking('/workspace/data')", "")
    tb.cells[4].source = tb.cells[4].source.replace(
        "read_csv('/workspace/data/train_set.csv'", f"read_csv('{tmpdir}/train_set.csv'"
    )
    tb.cells[31].source = tb.cells[31].source.replace("epochs=5", "epochs=1")
    tb.cells[37].source = tb.cells[37].source.replace("/workspace/ensemble", f"{tmpdir}/ensemble")
    tb.execute_cell(list(range(0, 38)))

    with utils.run_triton_server(f"{tmpdir}/ensemble", grpc_port=8001):
        tb.execute_cell(list(range(38, len(tb.cells))))

    tb.inject(
        """
        logits_count = predictions.shape[1]
        """
    )
    tb.execute_cell(len(tb.cells) - 1)

    cardinality = tb.ref("cardinality")
    logits_count = tb.ref("logits_count")
    assert logits_count == cardinality
