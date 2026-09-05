"""One fixed fit-batch audit of object memory already consumed by MCLN."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import sys


def file_sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    opt = parser.parse_args()
    root = opt.manifest.parent
    manifest = json.loads(opt.manifest.read_text())
    source = Path(manifest["model_source"])
    assert file_sha(source / "g0_source_manifest.json") == manifest["source_manifest_sha256"]
    for relative, expected in json.loads((source / "g0_source_manifest.json").read_text())["files"].items():
        assert file_sha(source / relative) == expected, relative
    assert file_sha(Path(__file__)) == manifest["observer_sha256"]
    checkpoint_path = Path(manifest["checkpoint"])
    assert file_sha(checkpoint_path) == manifest["checkpoint_sha256"]
    os.chdir(str(source))
    sys.path.insert(0, str(source))
    import numpy as np
    import torch
    from main_utils import parse_option
    from train_dist_mod import TrainTester
    from src.joint_det_dataset import Joint3DDataset
    from models.losses import _iou3d_par, box_cxcyczwhd_to_xyzxyz
    from models.rec_evaluator_filter import build_detector_overlap_valid
    from models.source_choice_adapter import compute_default_source_scores
    from scripts.run_nr3d_view_pair_role import read_train_rows

    raw_rows = read_train_rows(Path("/root/autodl-tmp/DATA_ROOT"))
    salt = "MCLN-NR3D-PAIR-READOUT-V1-20260905"
    fit_ids = [index for index, row in enumerate(raw_rows) if int(hashlib.sha256(
        (salt + "\0" + row["scan_id"]).encode()).hexdigest()[:8], 16) % 5 != 0]
    row_ids = fit_ids[:4]
    assert len(fit_ids) == 26747
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
    sys.argv = [sys.argv[0]]
    args = parse_option()
    vars(args).update(vars(checkpoint["config"]))
    model = TrainTester.get_model(args).cuda().eval()
    state = {key[7:]: value for key, value in checkpoint["model"].items()}
    model.load_state_dict(state, strict=True)
    model.requires_grad_(False)
    assert args.butd_cls and model.butd and not args.butd and not args.butd_gt

    class FixedFit(Joint3DDataset):
        def _scene_graph_parse(self, annos):
            annos[:] = [annos[index] for index in row_ids]
            super()._scene_graph_parse(annos)

    dataset = FixedFit(dataset_dict={"nr3d": 1}, test_dataset="nr3d", split="train",
                       data_path="/root/autodl-tmp/DATA_ROOT/", use_color=True,
                       detect_intermediate=True, butd_cls=True, skip_missing_superpoints=True)
    dataset.augment = False
    assert len(dataset) == 4
    for anno, row_id in zip(dataset.annos, row_ids):
        assert anno["scan_id"] == raw_rows[row_id]["scan_id"]
        assert anno["target_id"] == int(raw_rows[row_id]["target_id"])
    batch = TrainTester._to_gpu(next(iter(torch.utils.data.DataLoader(dataset, batch_size=4))))
    inputs = TrainTester._get_inputs(batch)
    inputs["train"] = False
    captured = {}
    original_decoder_forward = model.decoder[-1].forward

    def capture_decoder_inputs(*arguments, **keywords):
        captured["memory"] = keywords["detected_feats"]
        captured["padding"] = keywords["detected_mask"]
        return original_decoder_forward(*arguments, **keywords)

    def capture_query(module, arguments):
        captured["query"] = arguments[0].transpose(1, 2).detach()

    model.decoder[-1].forward = capture_decoder_inputs
    hook = model.x_query.register_forward_pre_hook(capture_query)
    with torch.no_grad():
        outputs = model(inputs)
        object_features = torch.cat([
            model.box_embeddings(inputs["det_boxes"]),
            model.class_embeddings(model.butd_class_embeddings(inputs["det_class_ids"])).transpose(1, 2),
        ], dim=1).transpose(1, 2).contiguous()
    model.decoder[-1].forward = original_decoder_forward
    hook.remove()
    assert object_features.shape == (4, 132, 288)
    assert torch.equal(object_features, captured["memory"])
    assert torch.equal(captured["padding"], ~inputs["det_bbox_label_mask"])
    assert torch.equal(inputs["det_boxes"], batch["all_bboxes"])
    assert torch.equal(inputs["det_bbox_label_mask"], batch["all_bbox_label_mask"])
    assert torch.isfinite(object_features).all()
    assert captured["query"].shape == (4, 256, 288)
    boxes = torch.cat([outputs["last_center"], outputs["last_pred_size"].clamp(min=1e-6)], dim=-1)
    valid = build_detector_overlap_valid(
        boxes, torch.ones(boxes.shape[:2], device=boxes.device, dtype=torch.bool),
        inputs["det_boxes"], inputs["det_bbox_label_mask"], iou_threshold=.25)
    scores = compute_default_source_scores(outputs, inputs)
    order = scores.masked_fill(~valid, -float("inf")).argsort(1, descending=True)[:, :32]
    rows = []
    for index, row_id in enumerate(row_ids):
        active = inputs["det_bbox_label_mask"][index]
        object_ids = active.nonzero().reshape(-1)
        ious = _iou3d_par(box_cxcyczwhd_to_xyzxyz(inputs["det_boxes"][index, object_ids]),
                         box_cxcyczwhd_to_xyzxyz(boxes[index]))[0]
        top = order[index][valid[index, order[index]]]
        full_covered = object_ids[(ious[:, valid[index]] > .25).any(-1)]
        top_covered = object_ids[(ious[:, top] > .25).any(-1)]
        true_classes = batch["all_class_ids"][index, active].long()
        predicted_classes = inputs["det_class_ids"][index, active].long()
        rows.append({"fit_row_id": row_id, "scan_id": raw_rows[row_id]["scan_id"],
                     "target_id": int(raw_rows[row_id]["target_id"]),
                     "input_object_slots": object_ids.tolist(),
                     "query_memory_covered_object_slots": full_covered.tolist(),
                     "target_top32_covered_object_slots": top_covered.tolist(),
                     "predicted_class_ids": predicted_classes.tolist(),
                     "gt_class_ids_audit_only": true_classes.tolist(),
                     "object_class_correct_count": int((predicted_classes == true_classes).sum()),
                     "object_count": int(active.sum()), "legal_query_count": int(valid[index].sum()),
                     "predicted_class_features_match_gt_for_all_objects": bool((predicted_classes == true_classes).all()),
                     "real_text_anchor_recall_measured": False})
    for name, value in model.state_dict().items():
        assert torch.equal(value.detach().cpu(), state[name]), name
    assert all(parameter.grad is None for parameter in model.parameters())
    result = {"schema": "mcln-nr3d-object-reference-input-audit-v1", "rows": rows,
              "fit_row_ids": row_ids, "split_salt": salt, "model_forwards": 1,
              "decoder_memory_matches_reconstructed_features_exactly": True,
              "decoder_padding_matches_input_validity_exactly": True,
              "butd_cls_uses_existing_instance_box_inputs": True,
              "feature_dimensions": {"box_position": 128, "predicted_class": 160, "total": 288},
              "independent_object_appearance_pooling": False,
              "gt_class_ids_used_for_forward": False,
              "checkpoint_state_unchanged": True, "optimizer_steps": 0,
              "heldout_rows": 0, "formal_rows": 0, "new_head_accuracy_evaluated": False,
              "manifest_sha256": file_sha(opt.manifest),
              "checkpoint_sha256": manifest["checkpoint_sha256"]}
    with (root / "receipt.json").open("x") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
