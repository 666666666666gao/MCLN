"""Observe the native protected Nr3D evaluation once, without model updates.

The input manifest pins an existing immutable model tree, historical CLI/config,
the protected averaged checkpoint, and this observer. No experimental head is
loaded. Reproduction and diagnostic validity are reported separately.
"""

import argparse
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time


def file_sha(path):
    digest = hashlib.sha256()
    with open(str(path), "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    with open(str(path), "x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    opt = parser.parse_args()
    manifest = json.loads(opt.manifest.read_text())
    addon = opt.manifest.parent
    source = Path(manifest["model_source"])
    for relative, expected in manifest["files"].items():
        assert file_sha(addon / relative) == expected, relative
    source_manifest = source / "g0_source_manifest.json"
    assert file_sha(source_manifest) == manifest["source_manifest_sha256"]
    for relative, expected in json.loads(source_manifest.read_text())["files"].items():
        assert file_sha(source / relative) == expected, relative
    checkpoint_path = Path(manifest["checkpoint"])
    assert file_sha(checkpoint_path) == manifest["checkpoint_sha256"]
    output = addon / "results"
    output.mkdir(exist_ok=False)
    os.chdir(str(source))
    sys.path.insert(0, str(source))
    import scripts
    scripts.__path__ = [str(addon / "scripts")] + list(scripts.__path__)
    import torch
    from main_utils import parse_option, prepare_source_moe_gate_checkpoint_config
    from train_dist_mod import TrainTester
    from scripts.nr3d_candidate_contract import diagnose_root_candidates

    sys.argv = [str(source / "train_dist_mod.py")] + manifest["eval_argv"]
    args = prepare_source_moe_gate_checkpoint_config(parse_option())
    historical_config = json.loads((addon / "historical_config.json").read_text())
    config_changes = {key: {"historical": value, "current": vars(args)[key]}
                      for key, value in historical_config.items()
                      if vars(args)[key] != value}
    assert set(config_changes) <= {"checkpoint_path", "log_dir", "exp"}, config_changes
    assert args.eval and not args.eval_train and args.batch_size == 16
    assert args.expected_eval_sample_count == 7899 and args.num_workers == 4
    assert args.butd_cls and not args.butd and not args.butd_gt
    assert args.eval_use_selector_choice_scores and not args.use_source_moe
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    torch.cuda.set_device(args.local_rank)
    torch.distributed.init_process_group(
        backend="nccl", init_method="env://", timeout=datetime.timedelta(seconds=5400))
    assert torch.distributed.get_world_size() == 1
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
    assert checkpoint["epoch"] == 57 and checkpoint["evaluation_only"] is True
    assert "optimizer" not in checkpoint and "scheduler" not in checkpoint
    expected_state = {(name[7:] if name.startswith("module.") else name): value
                      for name, value in checkpoint["model"].items()}

    def assert_exact_state(model):
        actual = model.module.state_dict()
        assert set(actual) == set(expected_state)
        for name, value in actual.items():
            assert torch.equal(value.detach().cpu(), expected_state[name]), name

    class ObservedTester(TrainTester):
        def get_loaders(self, current_args):
            train_loader, test_loader = super().get_loaders(current_args)
            assert train_loader is None and len(test_loader.dataset) == 7899
            assert test_loader.dataset.split == "val" and not test_loader.dataset.augment
            assert list(test_loader.sampler) == list(range(7899))
            self.annos = test_loader.dataset.annos
            self.observed_rows = []
            self.raw_sizes = None
            self.sa_indices = {}
            self.row_stream = open(str(output / "rows.jsonl"), "x")
            return train_loader, test_loader

        def _build_grounding_evaluator(self, current_args, prefixes):
            self.observed_evaluator = super()._build_grounding_evaluator(current_args, prefixes)
            return self.observed_evaluator

        def _main_eval_branch(self, *branch_args):
            statistics, end_points = super()._main_eval_branch(*branch_args)
            rows = diagnose_root_candidates(end_points, self.observed_evaluator)
            root_masks = end_points["gt_masks"][:, 0].bool()
            global_indices = self.sa_indices[1].long()
            centers = {"sa1": root_masks.gather(1, global_indices).sum(1).tolist()}
            for level in (2, 3, 4):
                global_indices = global_indices.gather(1, self.sa_indices[level].long())
                centers["sa{}".format(level)] = root_masks.gather(1, global_indices).sum(1).tolist()
            seed_indices = end_points["seed_inds"].long()
            query_indices = seed_indices.gather(1, end_points["query_points_sample_inds"].long())
            centers["fp2_seeds"] = root_masks.gather(1, seed_indices).sum(1).tolist()
            centers["kps_queries"] = root_masks.gather(1, query_indices).sum(1).tolist()
            for batch_row, row in enumerate(rows):
                index = len(self.observed_rows)
                anno = self.annos[index]
                assert end_points["scan_ids"][batch_row] == anno["scan_id"]
                assert int(end_points["target_id"][batch_row]) == anno["target_id"]
                row.update(id=index, scan_id=anno["scan_id"], target_id=anno["target_id"],
                           target_name=end_points["target_name"][batch_row],
                           raw_token_count=len(anno["utterance"].split()),
                           distractor_count=len(anno["distractor_ids"]),
                           is_view_dep=bool(end_points["is_view_dep"][batch_row]),
                           target_sampled_center_counts={key: value[batch_row]
                                                        for key, value in centers.items()},
                           raw_nonpositive_size_query_count=self.raw_sizes[0][batch_row],
                           raw_prediction_size_min=self.raw_sizes[1][batch_row])
                row["rec_query_seed_inside_target"] = (
                    None if row["rec_selection"] is None else bool(root_masks[
                        batch_row, query_indices[batch_row, row["rec_selection"]["query"]]]))
                self.row_stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
                self.observed_rows.append(row)
            self.row_stream.flush()
            return statistics, end_points

        @torch.no_grad()
        def evaluate_one_epoch(self, epoch, test_loader, model, criterion, set_criterion, current_args):
            assert_exact_state(model)

            def capture_raw_sizes(module, inputs, outputs):
                sizes = outputs["last_pred_size"]
                self.raw_sizes = ((sizes <= 0).any(-1).sum(-1).tolist(),
                                  sizes.amin(dim=(1, 2)).tolist())

            def capture_sampling(level):
                def hook(module, inputs, outputs):
                    self.sa_indices[level] = outputs[2]
                return hook

            handles = [model.module.register_forward_hook(capture_raw_sizes)]
            for level in (1, 2, 3, 4):
                handles.append(getattr(model.module.backbone_net, "sa{}".format(level))
                               .register_forward_hook(capture_sampling(level)))
            self.started = time.time()
            metrics = super().evaluate_one_epoch(epoch, test_loader, model, criterion,
                                                 set_criterion, current_args)
            self.elapsed = time.time() - self.started
            for handle in handles:
                handle.remove()
            self.row_stream.close()
            assert_exact_state(model)
            self.native_metrics = metrics
            return metrics

    tester = ObservedTester(args)
    write_json(output / "preflight.json", {
        "schema": "mcln-nr3d-official-candidate-audit-v1",
        "manifest_sha256": file_sha(opt.manifest),
        "config_changes": config_changes, "runtime_log_dir": args.log_dir,
        "historical_source_recovered": False,
        "source_identity": "immutable_current_source_reproduction_check",
        "optimizer_steps": 0, "checkpoint_writes": 0,
    })
    tester.main(args)
    rows = tester.observed_rows
    assert len(rows) == 7899
    metrics = tester.native_metrics
    summary = {"sample_count": len(rows)}
    for suffix, threshold in (("025", .25), ("050", .5)):
        summary["rec_hits" + suffix] = sum(row["rec_selection"] is not None
            and row["rec_selection"]["box_iou"] > threshold for row in rows)
        summary["mask_hits" + suffix] = sum(row["mask_selection"]["mask_iou"] > threshold for row in rows)
        assert summary["rec_hits" + suffix] == sum(
            group["hits" + suffix] for group in metrics["position_subgroups"].values())
        assert summary["mask_hits" + suffix] == metrics["mask"]["hits" + suffix]
    summary["mask_iou_sum"] = sum(row["mask_selection"]["mask_iou"] for row in rows)
    assert math.isclose(summary["mask_iou_sum"], metrics["mask"]["iou_sum"], abs_tol=1e-8, rel_tol=0)
    historical = json.loads((addon / "historical_metrics.json").read_text())
    match = {
        "rec_hits" + suffix: summary["rec_hits" + suffix] == sum(
            group["hits" + suffix] for group in historical["position_subgroups"].values())
        for suffix in ("025", "050")}
    match.update({"mask_hits" + suffix: summary["mask_hits" + suffix] == historical["mask"]["hits" + suffix]
                  for suffix in ("025", "050")})
    match["mask_iou_sum"] = math.isclose(summary["mask_iou_sum"], historical["mask"]["iou_sum"],
                                        abs_tol=1e-8, rel_tol=0)
    for relative, expected in json.loads(source_manifest.read_text())["files"].items():
        assert file_sha(source / relative) == expected, relative
    assert file_sha(checkpoint_path) == manifest["checkpoint_sha256"]
    write_json(output / "receipt.json", {
        "status": "complete", "summary": summary, "native_metrics": metrics,
        "native_row_parity": True, "historical_metric_match": match,
        "historical_metrics_reproduced": all(match.values()),
        "historical_source_recovered": False, "elapsed_seconds": tester.elapsed,
        "rows_sha256": file_sha(output / "rows.jsonl"),
        "manifest_sha256": file_sha(opt.manifest), "protected_state_unchanged": True,
        "optimizer_steps": 0, "checkpoint_writes": 0, "formal_promotion": False,
    })
    print("OFFICIAL_PATH_DIAGNOSTIC_COMPLETE", json.dumps(summary), flush=True)
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
