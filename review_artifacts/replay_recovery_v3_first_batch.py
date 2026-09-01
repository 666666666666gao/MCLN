from __future__ import print_function

import hashlib
import itertools
import json
import os
import pathlib
import random
import stat
import sys
from types import SimpleNamespace


AUDIT_ROOT = pathlib.Path(
    "/root/autodl-tmp/DATA_ROOT/output/network_v99_baseline_gt/nr3d/audit/"
    "nr3d_fpr_tv_e57_e58_b100_b16x1_recovery_v3"
)
CODE_ROOT = AUDIT_ROOT / "consumed_snapshot" / "code"
INPUT_ROOT = AUDIT_ROOT / "consumed_snapshot" / "inputs"
RUN_ROOT = (
    AUDIT_ROOT
    / "runtime_output"
    / "nr3d"
    / "nr3d_fpr_tv_e57_e58_b100_b16x1_recovery_v3"
    / "1788147434"
)
CONFIG = RUN_ROOT / "config.json"
CODE_MANIFEST = CODE_ROOT / "CODE_MANIFEST.json"
INPUT_MANIFEST = INPUT_ROOT / "INPUT_MANIFEST.json"
OUTPUT = pathlib.Path(
    "/root/mcln_fpr_collate_fix_review_b046bab/"
    "recovery_v3_first_batch_replay_receipt_v1.json"
)
EXPECTED_CONFIG_SHA256 = (
    "9c04246f7de1a0314def0feb8520338a55c60908cb7558d2fb9e3b45f0af291b"
)
EXPECTED_CODE_MANIFEST_SHA256 = (
    "63ea6ea0509144129882198a91f7af0ff6fac6bc4a84030a2402fe1c6a100823"
)
EXPECTED_INPUT_MANIFEST_SHA256 = (
    "7eac4a1686916780a9c7171522c1dbefb03653f95849ae367a6bf33836ec46b5"
)
EXPECTED_OLD_MAIN_UTILS_SHA256 = (
    "df1780a6ed0c8678759f33d060ad1e0aff25f39b6787f2bc0536540bd2da1ea5"
)
EPOCH = 58
BATCH_SIZE = 16


def read_regular(path, expected_sha):
    flags = os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(str(path), flags)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise SystemExit("input is not a regular file: " + str(path))
        chunks = []
        digest = hashlib.sha256()
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
            digest.update(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if (
            before.st_dev != after.st_dev
            or before.st_ino != after.st_ino
            or before.st_size != after.st_size
            or before.st_mtime_ns != after.st_mtime_ns):
        raise SystemExit("input changed while reading: " + str(path))
    actual_sha = digest.hexdigest()
    if actual_sha != expected_sha:
        raise SystemExit("input SHA changed: " + str(path))
    return b"".join(chunks), actual_sha


config_raw, config_sha = read_regular(CONFIG, EXPECTED_CONFIG_SHA256)
code_manifest_raw, code_manifest_sha = read_regular(
    CODE_MANIFEST, EXPECTED_CODE_MANIFEST_SHA256
)
input_manifest_raw, input_manifest_sha = read_regular(
    INPUT_MANIFEST, EXPECTED_INPUT_MANIFEST_SHA256
)
config = json.loads(config_raw.decode("utf-8"))
code_manifest = json.loads(code_manifest_raw.decode("utf-8"))
input_manifest = json.loads(input_manifest_raw.decode("utf-8"))
main_record = code_manifest.get("files", {}).get("main_utils.py")
if (
        not isinstance(main_record, dict)
        or main_record.get("sha256") != EXPECTED_OLD_MAIN_UTILS_SHA256):
    raise SystemExit("replay code manifest does not bind the failed main_utils")
if input_manifest.get("code_manifest_sha256") != code_manifest_sha:
    raise SystemExit("replay input snapshot is bound to another code snapshot")
for key, expected in {
        "dataset": ["nr3d"],
        "test_dataset": "nr3d",
        "batch_size": BATCH_SIZE,
        "rng_seed": 0,
        "joint_det": True,
        "butd_cls": True,
        "start_epoch": EPOCH,
        "max_epoch": EPOCH,
        "use_parent_relative_text_verifier": True,
        "use_sacr_source": False,
        "use_sacr_score_refiner": False,
}.items():
    if type(config.get(key)) is not type(expected) or config.get(key) != expected:
        raise SystemExit("replay config changed: " + key)

os.chdir(str(CODE_ROOT))
sys.path.insert(0, str(CODE_ROOT))
sys.path.insert(1, str(CODE_ROOT / "pointnet2"))

import numpy as np
import torch
from torch.utils.data._utils.collate import default_collate
from torch.utils.data.distributed import DistributedSampler

from main_utils import STRUCTURED_COLLATE_KEYS, joint_det_structured_collate
from src.joint_det_dataset import Joint3DDataset


args = SimpleNamespace(**config)
dataset_dict = {"nr3d": 1, "scannet": 10}
train_dataset = Joint3DDataset(
    dataset_dict=dataset_dict,
    test_dataset=args.test_dataset,
    split="train",
    use_color=args.use_color,
    use_height=args.use_height,
    overfit=False,
    data_path=args.data_root,
    detect_intermediate=args.detect_intermediate,
    use_multiview=args.use_multiview,
    butd=args.butd,
    butd_gt=args.butd_gt,
    butd_cls=args.butd_cls,
    augment_det=args.augment_det,
    skip_missing_superpoints=args.skip_missing_superpoints,
    use_sacr_source=True,
    legacy_scene_graph_cache_path=getattr(args, "legacy_scene_graph_cache", ""),
    legacy_scene_graph_cache_strict=getattr(
        args, "legacy_scene_graph_cache_strict", False
    ),
    legacy_scene_graph_cache_expected_target_selection=getattr(
        args, "legacy_scene_graph_cache_expected_target_selection", ""
    ),
    legacy_scene_graph_cache_expected_sha256=getattr(
        args, "legacy_scene_graph_cache_expected_sha256", ""
    ),
)
if len(train_dataset) != 44909:
    raise SystemExit("replay train dataset length changed")

sampler = DistributedSampler(
    train_dataset,
    num_replicas=1,
    rank=0,
    shuffle=True,
    seed=0,
    drop_last=False,
)
sampler.set_epoch(EPOCH)
first_batch_indices = list(itertools.islice(iter(sampler), BATCH_SIZE))
if len(first_batch_indices) != BATCH_SIZE:
    raise SystemExit("replay sampler returned an incomplete first batch")

# Match DataLoader's first worker seed and the reviewed seed_worker(worker_id=0).
loader_generator = torch.Generator()
loader_generator.manual_seed(0)
base_seed = int(
    torch.empty((), dtype=torch.int64).random_(generator=loader_generator).item()
)
torch.set_num_threads(1)
torch.manual_seed(base_seed)
worker_seed = base_seed % (2 ** 32)
np.random.seed(worker_seed)
random.seed(worker_seed)
np.random.seed(np.random.get_state()[1][0])

samples = [train_dataset[index] for index in first_batch_indices]
structured_lengths = {}
for key in sorted(STRUCTURED_COLLATE_KEYS):
    if key not in samples[0]:
        continue
    values = [sample[key] for sample in samples]
    lengths = [len(value) for value in values]
    structured_lengths[key] = lengths
variable_structured_lengths = {
    key: lengths
    for key, lengths in structured_lengths.items()
    if len(set(lengths)) > 1
}
if not variable_structured_lengths:
    raise SystemExit("first replay batch did not contain variable structured fields")

try:
    default_collate(samples)
except Exception as error:
    default_exception_type = type(error).__name__
    default_exception_message = str(error)
else:
    raise SystemExit("default_collate unexpectedly accepted the first replay batch")
if (
        default_exception_type != "RuntimeError"
        or default_exception_message
        != "each element in list of batch should be of equal size"):
    raise SystemExit("first replay batch failed with another error")

structured_batch = joint_det_structured_collate(samples)
if set(structured_batch) != set(samples[0]):
    raise SystemExit("structured collate changed the batch key set")
for key in STRUCTURED_COLLATE_KEYS:
    if key in samples[0] and structured_batch[key] != [
            sample[key] for sample in samples]:
        raise SystemExit("structured collate changed a structured field: " + key)

payload = {
    "schema": "mcln-fpr-tv-first-batch-collate-replay-v1",
    "audit_root": str(AUDIT_ROOT),
    "config": {"path": str(CONFIG), "sha256": config_sha},
    "code_manifest": {
        "path": str(CODE_MANIFEST),
        "sha256": code_manifest_sha,
    },
    "input_manifest": {
        "path": str(INPUT_MANIFEST),
        "sha256": input_manifest_sha,
    },
    "failed_main_utils_sha256": EXPECTED_OLD_MAIN_UTILS_SHA256,
    "dataset_length": len(train_dataset),
    "epoch": EPOCH,
    "batch_size": BATCH_SIZE,
    "sampler": {
        "type": "DistributedSampler",
        "num_replicas": 1,
        "rank": 0,
        "shuffle": True,
        "seed": 0,
        "drop_last": False,
    },
    "worker_id": 0,
    "worker_base_seed": base_seed,
    "first_batch_indices": first_batch_indices,
    "structured_lengths": structured_lengths,
    "variable_structured_lengths": variable_structured_lengths,
    "default_collate": {
        "failed": True,
        "exception_type": default_exception_type,
        "exception_message": default_exception_message,
    },
    "structured_collate": {"succeeded": True},
    "conclusion": "first_batch_fails_before_training_loop_body",
}
raw = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
descriptor = os.open(
    str(OUTPUT), os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC, 0o444
)
try:
    offset = 0
    while offset < len(raw):
        offset += os.write(descriptor, raw[offset:])
    os.fsync(descriptor)
finally:
    os.close(descriptor)
directory_descriptor = os.open(str(OUTPUT.parent), os.O_RDONLY)
try:
    os.fsync(directory_descriptor)
finally:
    os.close(directory_descriptor)
print("first_batch_collate_replay=pass receipt=" + str(OUTPUT))
