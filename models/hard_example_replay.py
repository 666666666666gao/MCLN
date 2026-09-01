from __future__ import print_function

import hashlib
import json
import math
from pathlib import Path

import torch
import torch.distributed as dist
from torch.utils.data import Sampler


HARD_EXAMPLE_REPLAY_SCHEMA = "mcln-hard-example-replay-v1"
HARD_EXAMPLE_REPLAY_CRITERIA = {
    "default_top1_iou_lte": 0.25,
    "default_topk": 5,
    "topk_oracle_iou_gt": 0.25,
}


def _sha256_bytes(raw):
    return hashlib.sha256(raw).hexdigest()


def _load_manifest(path, expected_sha256):
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise ValueError(
            "hard-example replay manifest does not exist: {}".format(path)
        )
    if not isinstance(expected_sha256, str) or len(expected_sha256) != 64:
        raise ValueError(
            "hard-example replay requires an exact 64-character SHA-256"
        )
    raw = path.read_bytes()
    actual_sha256 = _sha256_bytes(raw)
    if actual_sha256 != expected_sha256.lower():
        raise ValueError(
            "hard-example replay manifest SHA-256 changed: {}".format(
                actual_sha256
            )
        )
    try:
        manifest = json.loads(raw.decode("utf-8"))
    except Exception as exc:
        raise ValueError(
            "hard-example replay manifest is not valid UTF-8 JSON"
        ) from exc
    if not isinstance(manifest, dict):
        raise ValueError("hard-example replay manifest must be a JSON object")
    return path, actual_sha256, manifest


def _validate_manifest(manifest):
    if manifest.get("schema") != HARD_EXAMPLE_REPLAY_SCHEMA:
        raise ValueError("unsupported hard-example replay manifest schema")
    if manifest.get("criteria") != HARD_EXAMPLE_REPLAY_CRITERIA:
        raise ValueError("hard-example replay criteria changed")
    if manifest.get("repeat_count") != 1:
        raise ValueError("hard-example replay repeat_count must be exactly 1")
    dataset = manifest.get("dataset")
    if not isinstance(dataset, str) or not dataset:
        raise ValueError("hard-example replay dataset must be non-empty")
    for key in ("base_dataset_size", "joint_dataset_size", "hard_count"):
        value = manifest.get(key)
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError("invalid hard-example replay {}".format(key))
    if manifest["base_dataset_size"] > manifest["joint_dataset_size"]:
        raise ValueError("base dataset cannot exceed joint dataset")
    records = manifest.get("hard_examples")
    if not isinstance(records, list) or len(records) != manifest["hard_count"]:
        raise ValueError("hard-example replay record count changed")
    indices = []
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("hard-example replay records must be objects")
        index = record.get("dataset_index")
        if not isinstance(index, int) or isinstance(index, bool):
            raise ValueError("hard-example replay dataset_index must be int")
        if not 0 <= index < manifest["base_dataset_size"]:
            raise ValueError("hard-example replay dataset_index is out of range")
        if not isinstance(record.get("scan_id"), str):
            raise ValueError("hard-example replay scan_id must be a string")
        if "target_id" not in record:
            raise ValueError("hard-example replay record lacks target_id")
        indices.append(index)
    if indices != sorted(indices) or len(set(indices)) != len(indices):
        raise ValueError(
            "hard-example replay indices must be unique and sorted"
        )
    return indices


def _annotation_value(annotation, key):
    if not isinstance(annotation, dict):
        raise ValueError("training annotations must be dictionaries")
    if key not in annotation:
        raise ValueError("training annotation lacks {}".format(key))
    return annotation[key]


def _validate_dataset_binding(dataset, manifest):
    if len(dataset) != manifest["joint_dataset_size"]:
        raise ValueError(
            "hard-example replay joint dataset size changed: {} != {}".format(
                len(dataset), manifest["joint_dataset_size"]
            )
        )
    annotations = getattr(dataset, "annos", None)
    if not isinstance(annotations, list) or len(annotations) != len(dataset):
        raise ValueError(
            "hard-example replay requires the bound annotation list"
        )
    base_size = manifest["base_dataset_size"]
    dataset_name = manifest["dataset"]
    for index in range(base_size):
        annotation_dataset = str(
            _annotation_value(annotations[index], "dataset")
        )
        if annotation_dataset != dataset_name:
            raise ValueError(
                "hard-example replay base annotation order changed at {}"
                .format(index)
            )
    for record in manifest["hard_examples"]:
        index = record["dataset_index"]
        annotation = annotations[index]
        if str(_annotation_value(annotation, "scan_id")) != record["scan_id"]:
            raise ValueError(
                "hard-example replay scan identity changed at {}".format(index)
            )
        if str(_annotation_value(annotation, "target_id")) != str(
            record["target_id"]
        ):
            raise ValueError(
                "hard-example replay target identity changed at {}".format(
                    index
                )
            )


class HardExampleReplayDistributedSampler(Sampler):
    """Shuffle every training row once and replay each fixed hard row once."""

    def __init__(
        self,
        dataset,
        manifest_path,
        expected_manifest_sha256,
        batch_size,
        num_replicas=None,
        rank=None,
        seed=0,
    ):
        if num_replicas is None:
            num_replicas = dist.get_world_size() if dist.is_initialized() else 1
        if rank is None:
            rank = dist.get_rank() if dist.is_initialized() else 0
        if not isinstance(num_replicas, int) or num_replicas <= 0:
            raise ValueError("num_replicas must be positive")
        if not isinstance(rank, int) or not 0 <= rank < num_replicas:
            raise ValueError("rank is outside num_replicas")
        if not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if not isinstance(seed, int) or isinstance(seed, bool):
            raise ValueError("seed must be an integer")

        manifest_path, manifest_sha256, manifest = _load_manifest(
            manifest_path, expected_manifest_sha256
        )
        hard_indices = _validate_manifest(manifest)
        _validate_dataset_binding(dataset, manifest)

        self.dataset = dataset
        self.dataset_size = len(dataset)
        self.hard_indices = tuple(hard_indices)
        self.batch_size = batch_size
        self.num_replicas = num_replicas
        self.rank = rank
        self.seed = seed
        self.epoch = 0
        requested_size = self.dataset_size + len(self.hard_indices)
        global_batch_size = self.batch_size * self.num_replicas
        self.total_size = int(
            math.ceil(float(requested_size) / global_batch_size)
            * global_batch_size
        )
        self.padding_size = self.total_size - requested_size
        self.num_samples = self.total_size // self.num_replicas
        self.manifest_path = str(manifest_path)
        self.manifest_sha256 = manifest_sha256
        self.manifest = manifest

    def __iter__(self):
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        base = torch.randperm(self.dataset_size, generator=generator)
        hard_tensor = torch.tensor(self.hard_indices, dtype=torch.long)
        hard_order = hard_tensor[
            torch.randperm(len(hard_tensor), generator=generator)
        ]
        pieces = [base, hard_order]
        if self.padding_size:
            repeats = int(
                math.ceil(float(self.padding_size) / len(hard_tensor))
            )
            pieces.append(hard_order.repeat(repeats)[: self.padding_size])
        combined = torch.cat(pieces)
        combined = combined[
            torch.randperm(self.total_size, generator=generator)
        ]
        rank_indices = combined[self.rank : self.total_size : self.num_replicas]
        if len(rank_indices) != self.num_samples:
            raise RuntimeError("hard-example replay sampler size drifted")
        return iter(rank_indices.tolist())

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def summary(self):
        return {
            "base_dataset_size": self.manifest["base_dataset_size"],
            "hard_count": len(self.hard_indices),
            "joint_dataset_size": self.dataset_size,
            "manifest_path": self.manifest_path,
            "manifest_sha256": self.manifest_sha256,
            "padding_size": self.padding_size,
            "per_rank_samples": self.num_samples,
            "repeat_count": 1,
            "total_size": self.total_size,
        }
