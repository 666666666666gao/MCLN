"""One G0 role: protected averaged weights, fresh matched AdamW, train only.

Run from an immutable old/fixed source snapshot. No production training entry
point is modified and no checkpoint-saving function is called.
"""

import argparse
import ast
import copy
import csv
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.nr3d_view_pair_contract import (
    CHECKPOINT_SHA, digest_ids, is_view_dependent, split_rows, validate_census,
)


def file_sha(path):
    digest = hashlib.sha256()
    with open(str(path), "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    with open(str(path), "x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")


def read_train_rows(data_root):
    train_list = Path("data/meta_data/nr3d_train_scans.txt")
    csv_path = data_root / "refer_it_3d/nr3d.csv"
    if file_sha(train_list) != "df7d28238ede7225d4f53617c58203c217be003e3314548fe1df7ca9c4d27508":
        raise ValueError("train scene-list identity drift")
    if file_sha(csv_path) != "5de4f1b47130803c88f7c57903e7b6df5473f1b903c32cf28d06fa9c25996a67":
        raise ValueError("Nr3D CSV identity drift")
    scenes = set(ast.literal_eval(train_list.read_text()))
    with open(str(csv_path)) as stream:
        return [row for row in csv.DictReader(stream) if row["scan_id"] in scenes]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--role", choices=("old", "fixed"), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    opt = parser.parse_args()
    opt.output.mkdir(parents=True, exist_ok=False)
    data_root = Path("/root/autodl-tmp/DATA_ROOT")
    source_root = Path(__file__).resolve().parents[1]
    source_manifest = json.loads((source_root / "g0_source_manifest.json").read_text())
    for relative, expected in source_manifest["files"].items():
        if file_sha(source_root / relative) != expected:
            raise ValueError("source drift: " + relative)
    if file_sha(opt.checkpoint) != CHECKPOINT_SHA:
        raise ValueError("protected checkpoint identity drift")
    raw_rows = read_train_rows(data_root)
    partitions = split_rows(raw_rows)
    census = validate_census(raw_rows, partitions)
    print("G0 input census PASS", json.dumps(census), flush=True)

    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    from torch.utils.data.distributed import DistributedSampler
    from torch.nn.parallel import DistributedDataParallel
    from main_utils import parse_option
    from train_dist_mod import TrainTester
    from src.joint_det_dataset import Joint3DDataset
    from src.grounding_evaluator import GroundingEvaluator

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.cuda.set_device(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.distributed.init_process_group(backend="nccl", init_method="env://")
    checkpoint = torch.load(str(opt.checkpoint), map_location="cpu")
    if checkpoint["epoch"] != 57 or checkpoint.get("evaluation_only") is not True:
        raise ValueError("expected the protected eval-only E57/E69 average")
    if "optimizer" in checkpoint or "scheduler" in checkpoint:
        raise ValueError("checkpoint payload no longer matches the amended contract")
    saved_argv = sys.argv
    sys.argv = [saved_argv[0]]
    args = parse_option()
    sys.argv = saved_argv
    vars(args).update(vars(checkpoint["config"]))
    args.lr = 1e-5
    args.lr_backbone = 1e-4
    args.source_choice_selector_lr = 1.25e-5
    args.start_epoch = 1
    args.max_epoch = 1
    args.checkpoint_path = str(opt.checkpoint)
    args.checkpoint_metric_retention = False
    args.expected_eval_sample_count = 7151
    args.log_dir = str(opt.output / "runtime")
    args.exp = "g0_view_pair_" + opt.role
    args.print_freq = 100
    args.num_workers = 4
    args.persistent_train_workers = False
    args.batch_size = 16
    args.eval = False
    args.eval_train = False
    args.data_root = str(data_root) + "/"
    if not (args.use_source_choice_selector and args.eval_use_selector_choice_scores
            and args.butd_cls and args.joint_det and not args.use_source_moe):
        raise ValueError("protected model/evaluator protocol drift")
    tester = TrainTester(args)
    model = tester.get_model(args).cuda()
    model = DistributedDataParallel(model, device_ids=[0], broadcast_buffers=False,
                                    find_unused_parameters=True)
    model.load_state_dict(checkpoint["model"], strict=True)
    del checkpoint
    optimizer = tester.get_optimizer(args, model)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[])
    groups = [{"name": group["name"], "lr": group["lr"],
               "parameter_names": group["parameter_names"]}
              for group in optimizer.param_groups]
    if [g["lr"] for g in groups] != [1e-5, 1e-4, 1e-5, 1.25e-5] or optimizer.state:
        raise ValueError("fresh optimizer contract drift")
    print("G0 strict model load PASS; fresh optimizer groups",
          [(g["name"], g["lr"], len(g["parameter_names"])) for g in groups], flush=True)

    class PairDataset(Joint3DDataset):
        def _augment(self, pc, color, rotate):
            pc, color, augmentations = super()._augment(pc, color, rotate)
            self.pair_rotate_allowed = bool(rotate)
            self.pair_large_transform = bool(
                abs(augmentations["theta_z"]) > 45 or
                augmentations.get("yz_flip", False) or augmentations.get("xz_flip", False)
            )
            return pc, color, augmentations

        def __getitem__(self, index):
            self.pair_rotate_allowed = False
            self.pair_large_transform = False
            item = super().__getitem__(index)
            source_id = self.pair_ids[index]
            item["view_pair_id"] = np.int64(source_id)
            # Use the same pre-parser, fixed view groups for both roles.
            item["is_view_dep"] = is_view_dependent(raw_rows[source_id]["utterance"])
            item["view_pair_rotate_allowed"] = self.pair_rotate_allowed
            item["view_pair_large_transform"] = self.pair_large_transform
            return item

    base = PairDataset(
        dataset_dict={"nr3d": 1}, test_dataset="nr3d", split="train",
        data_path=args.data_root, use_color=args.use_color,
        detect_intermediate=args.detect_intermediate, butd_cls=args.butd_cls,
        skip_missing_superpoints=args.skip_missing_superpoints,
    )
    if len(base.annos) != len(raw_rows):
        raise ValueError("runtime annotations do not match raw census")
    for anno, raw in zip(base.annos, raw_rows):
        if anno["scan_id"] != raw["scan_id"] or anno["target_id"] != int(raw["target_id"]):
            raise ValueError("runtime annotation ordering drift")
    datasets = {}
    for name, ids in partitions.items():
        dataset = copy.copy(base)
        dataset.annos = [base.annos[index] for index in ids]
        dataset.pair_ids = ids
        dataset.augment = name == "fit"
        datasets[name] = dataset

    def seed_worker(worker_id):
        torch.set_num_threads(1)
        seed = torch.initial_seed() % 2**32
        np.random.seed(seed)
        random.seed(seed)

    loaders = {}
    for name in ("fit", "holdout"):
        sampler = DistributedSampler(datasets[name], shuffle=name == "fit")
        sampler.set_epoch(58)
        loaders[name] = DataLoader(
            datasets[name], batch_size=16, sampler=sampler, num_workers=4,
            pin_memory=True, drop_last=False, worker_init_fn=seed_worker,
            generator=torch.Generator().manual_seed(0), prefetch_factor=2,
        )
    preflight = {"schema": "mcln-nr3d-view-pair-v2", "role": opt.role,
                 "checkpoint_sha256": CHECKPOINT_SHA,
                 "optimizer_initialization": "fresh_adamw_zero_moments",
                 "optimizer_groups": groups, "scheduler_milestones": [],
                 "source_manifest_sha256": file_sha(source_root / "g0_source_manifest.json"),
                 "census": census, "fit_batches": len(loaders["fit"]),
                 "holdout_batches": len(loaders["holdout"]),
                 "formal_validation_dataset_constructed": False}
    write_json(opt.output / "preflight.json", preflight)
    criterion, set_criterion = tester.get_criterion(args)
    if opt.preflight_only:
        # One fit batch, real loss and backward; no optimizer step or heldout read.
        batch = tester._to_gpu(next(iter(loaders["fit"])))
        model.train()
        outputs = model(tester._get_inputs(batch))
        outputs.update(batch)
        loss, outputs = tester._compute_loss(outputs, criterion, set_criterion, args)
        tester._validated_batch_loss_values(loss, outputs, optimizer)
        loss.backward()
        tester._reject_nonfinite_optimizer_gradients(optimizer, loss, model=model)
        write_json(opt.output / "smoke.json", {
            "finite_loss": float(loss.detach()), "backward": True,
            "optimizer_steps": 0, "heldout_batches": 0, "weights_written": 0,
        })
        print("G0 zero-step fit-only smoke PASS", flush=True)
        torch.distributed.destroy_process_group()
        return

    class ObservedLoader:
        def __init__(self, loader):
            self.loader = loader
            self.ids = []
            self.rotate_allowed = 0
            self.large_transform = 0

        def __len__(self):
            return len(self.loader)

        def __iter__(self):
            for batch in self.loader:
                self.ids.extend(batch["view_pair_id"].tolist())
                self.rotate_allowed += int(batch["view_pair_rotate_allowed"].sum())
                self.large_transform += int(batch["view_pair_large_transform"].sum())
                yield batch

    observed = ObservedLoader(loaders["fit"])
    started = time.time()
    training = tester.train_one_epoch(58, observed, model, criterion, set_criterion,
                                      optimizer, scheduler, args)
    training.update({"sample_count": len(observed.ids),
                     "sample_identity_sha256": digest_ids(observed.ids),
                     "sample_order_sha256": digest_ids(observed.ids, ordered=True),
                     "optimizer_steps": scheduler.last_epoch,
                     "rotate_allowed_rows": observed.rotate_allowed,
                     "large_transform_rows": observed.large_transform,
                     "elapsed_seconds": time.time() - started})
    if (training["sample_identity_sha256"] != census["fit"]["identity_sha256"] or
            training["optimizer_steps"] != 1611 or training["batch_count"] != 1611):
        raise ValueError("complete fit epoch identity/step mismatch")
    write_json(opt.output / "training.json", training)

    class PairEvaluator(GroundingEvaluator):
        def _record_position_subgroups(self, outputs, bid, threshold, found):
            super()._record_position_subgroups(outputs, bid, threshold, found)
            source_id = int(outputs["view_pair_id"][bid])
            key = "hit025" if threshold == 0.25 else "hit050"
            row_results.setdefault(source_id, {
                "id": source_id, "view_dependent": bool(outputs["is_view_dep"][bid]),
            })[key] = bool(found[0])

    row_results = {}
    evaluator = PairEvaluator(prefixes=["last_"], topks=[1],
                              filter_non_gt_boxes=True,
                              eval_use_selector_choice_scores=True)
    # Re-seed evaluation: old/fixed training augmentation consumes different RNG.
    random.seed(1000)
    np.random.seed(1000)
    torch.manual_seed(1000)
    torch.cuda.manual_seed_all(1000)
    model.eval()
    with torch.no_grad():
        for index, batch in enumerate(loaders["holdout"]):
            _, outputs = tester._main_eval_branch(
                index, batch, loaders["holdout"], model, {}, criterion, set_criterion, args
            )
            evaluator.evaluate_bbox_by_pos_align(outputs, "last_")
    rows = [row_results[index] for index in sorted(row_results)]
    if digest_ids(row["id"] for row in rows) != census["holdout"]["identity_sha256"]:
        raise ValueError("heldout identity mismatch")
    for relative, expected in source_manifest["files"].items():
        if file_sha(source_root / relative) != expected:
            raise ValueError("postflight source drift: " + relative)
    if file_sha(opt.checkpoint) != CHECKPOINT_SHA or list(opt.output.rglob("*.pth")):
        raise ValueError("checkpoint/output preservation failed")
    receipt = dict(preflight, training=training, rows=rows, weights_written=0,
                   elapsed_seconds=time.time() - started, status="complete")
    write_json(opt.output / "receipt.json", receipt)
    print("G0 role COMPLETE", opt.role, len(rows), flush=True)
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
