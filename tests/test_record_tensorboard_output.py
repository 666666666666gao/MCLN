import os

from utils import record_tensorboard


def test_tensorboard_writers_are_scoped_to_requested_output(tmp_path, monkeypatch):
    writer_paths = []

    class FakeSummaryWriter(object):
        def __init__(self, path):
            writer_paths.append(path)

    monkeypatch.setattr(record_tensorboard, "SummaryWriter", FakeSummaryWriter)

    record_tensorboard.TensorBoard(str(tmp_path), distributed_rank=0)

    assert writer_paths == [
        os.path.join(str(tmp_path), "tensorboard/train"),
        os.path.join(str(tmp_path), "tensorboard/val"),
    ]
