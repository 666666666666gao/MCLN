"""Four frozen real-scene forwards: original/repeat/padded and masked/padded.

Use the first four frozen G0 fit rows, no optimizer, no validation dataset.
"""

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import sys
import types

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--masked-layer-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    opt = parser.parse_args()
    import numpy as np
    import torch
    from main_utils import parse_option
    from train_dist_mod import TrainTester
    from src.joint_det_dataset import Joint3DDataset
    from models.rec_evaluator_filter import build_detector_overlap_valid
    from scripts.run_nr3d_view_pair_role import file_sha, read_train_rows
    from scripts.nr3d_view_pair_contract import CHECKPOINT_SHA, split_rows

    if file_sha(opt.checkpoint) != CHECKPOINT_SHA:
        raise ValueError("checkpoint identity drift")
    data_root = Path("/root/autodl-tmp/DATA_ROOT")
    raw_rows = read_train_rows(data_root)
    row_ids = split_rows(raw_rows)["fit"][:4]
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    sys.argv = [sys.argv[0]]
    args = parse_option()
    checkpoint = torch.load(str(opt.checkpoint), map_location="cpu")
    vars(args).update(vars(checkpoint["config"]))
    model = TrainTester.get_model(args).cuda().eval()
    model.load_state_dict({k[7:]: v for k, v in checkpoint["model"].items()}, strict=True)
    del checkpoint

    class FitFour(Joint3DDataset):
        def _scene_graph_parse(self, annos):
            # Restrict before expensive parser execution; raw ordering is fixed.
            annos[:] = [annos[index] for index in row_ids]
            super()._scene_graph_parse(annos)

    dataset = FitFour(dataset_dict={"nr3d": 1}, test_dataset="nr3d", split="train",
                      data_path=str(data_root) + "/", use_color=True,
                      detect_intermediate=True, butd_cls=True, skip_missing_superpoints=True)
    dataset.augment = False
    for anno, index in zip(dataset.annos, row_ids):
        assert anno["scan_id"] == raw_rows[index]["scan_id"]
        assert anno["target_id"] == int(raw_rows[index]["target_id"])
    batch = TrainTester._to_gpu(next(iter(torch.utils.data.DataLoader(dataset, batch_size=4))))
    inputs = TrainTester._get_inputs(batch)
    inputs["train"] = False
    tokenizer = model.tokenizer

    class ExtraPadding:
        def batch_encode_plus(self, texts, **kwargs):
            original = tokenizer.batch_encode_plus(texts, **kwargs)
            kwargs["padding"] = "max_length"
            kwargs["max_length"] = original["input_ids"].shape[1] + 16
            return tokenizer.batch_encode_plus(texts, **kwargs)

    def forward():
        with torch.no_grad():
            outputs = model(inputs)
            boxes = torch.cat([outputs["last_center"], outputs["last_pred_size"].clamp(min=1e-6)], -1)
            valid = build_detector_overlap_valid(
                boxes, torch.ones(boxes.shape[:2], dtype=torch.bool, device=boxes.device),
                batch["all_detected_boxes"], batch["all_detected_bbox_label_mask"].bool(),
                iou_threshold=.25,
            )
            scores = outputs["selected_source_scores"]
            selected = scores.masked_fill(~valid, -float("inf")).argmax(-1)
            selected = selected.masked_fill(~valid.any(-1), -1)
            return {"boxes": boxes.cpu(), "scores": scores.cpu(),
                    "valid": valid.cpu(), "query": selected.cpu(),
                    "text_mask_logits": [v.detach().cpu() for v in outputs["last_pred_masks"]],
                    "query_mask_logits": [v.detach().cpu() for v in outputs["sp_last_pred_masks"]]}

    original = forward()
    repeat = forward()
    model.tokenizer = ExtraPadding()
    padded = forward()
    module_spec = importlib.util.spec_from_file_location("masked_spatial_layers", str(opt.masked_layer_source))
    masked_module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(masked_module)
    for layer in model.cross_encoder.layers:
        layer.forward = types.MethodType(masked_module.BiEncoderLayer.forward, layer)
    for layer in model.decoder:
        layer.forward = types.MethodType(masked_module.BiDecoderLayer.forward, layer)
    model.tokenizer = tokenizer
    masked = forward()
    model.tokenizer = ExtraPadding()
    masked_padded = forward()

    def compare(before, after):
        return {
            "query_before": before["query"].tolist(), "query_after": after["query"].tolist(),
            "query_changes": int((before["query"] != after["query"]).sum()),
            "valid_query_bit_changes": int((before["valid"] != after["valid"]).sum()),
            "box_max_abs_delta": float((before["boxes"] - after["boxes"]).abs().max()),
            "score_max_abs_delta": float((before["scores"] - after["scores"]).abs().max()),
            "text_mask_logit_max_abs_delta": max(float((a-b).abs().max()) for a,b in zip(before["text_mask_logits"], after["text_mask_logits"])),
            "query_mask_logit_max_abs_delta": max(float((a-b).abs().max()) for a,b in zip(before["query_mask_logits"], after["query_mask_logits"])),
        }

    result = {"schema": "mcln-real-scene-padding-intervention-v1", "checkpoint_sha256": CHECKPOINT_SHA,
              "script_sha256": file_sha(__file__), "masked_layer_sha256": file_sha(opt.masked_layer_source),
              "fit_row_ids": row_ids, "same_inputs": True, "appended_padding": 16,
              "optimizer_steps": 0, "formal_validation_evaluated": False, "weights_written": 0,
              "repeat_control": compare(original, repeat), "original_padding": compare(original, padded),
              "masked_padding": compare(masked, masked_padded), "fix_vs_original": compare(original, masked)}
    with open(str(opt.output), "x") as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
