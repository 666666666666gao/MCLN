"""Five frozen real-scene forwards with seed-aligned padding diagnostics.

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


def shared_query_indices(before_ids, after_ids):
    """Match physical input-point IDs, never the position on the Query axis."""
    assert len(set(before_ids)) == len(before_ids)
    assert len(set(after_ids)) == len(after_ids)
    after_position = {point_id: index for index, point_id in enumerate(after_ids)}
    pairs = [(index, after_position[point_id])
             for index, point_id in enumerate(before_ids) if point_id in after_position]
    return [a for a, _ in pairs], [b for _, b in pairs]


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
    from models.mask_fusion import as_query_mask_logits, fuse_query_mask_logits
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
    trace = {}

    def tensor_hook(name):
        def capture(module, arguments, output):
            trace[name] = output.detach().cpu()
        return capture

    def encoder_hook(name):
        def capture(module, arguments, output):
            trace[name + "_visual"] = output[0].detach().cpu()
            trace[name + "_text"] = output[1].detach().cpu()
        return capture

    def swa_hook(module, arguments, output):
        trace["swa_raw"].append(output[2].detach())

    hooks = [model.text_projector.register_forward_hook(tensor_hook("text_projector")),
             model.swa_layers[-1].register_forward_hook(swa_hook)]
    for index, layer in enumerate(model.cross_encoder.layers):
        hooks.append(layer.register_forward_hook(encoder_hook("encoder_{}".format(index))))
    for index, layer in enumerate(model.decoder):
        hooks.append(layer.register_forward_hook(tensor_hook("decoder_{}".format(index))))

    class ExtraPadding:
        def batch_encode_plus(self, texts, **kwargs):
            original = tokenizer.batch_encode_plus(texts, **kwargs)
            kwargs["padding"] = "max_length"
            kwargs["max_length"] = original["input_ids"].shape[1] + 16
            return tokenizer.batch_encode_plus(texts, **kwargs)

    def forward():
        nonlocal trace
        trace = {"swa_raw": []}
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
            point_ids = outputs["seed_inds"].long().gather(
                1, outputs["query_points_sample_inds"].long())
            seed_probability = outputs["seeds_obj_cls_logits"].sigmoid().squeeze(1)
            cutoff = seed_probability.topk(model.num_queries, dim=1)[0][:, -1:]
            seed_topk_ties = {"cutoff": cutoff.squeeze(1).cpu().tolist(),
                             "cutoff_equal_count": (seed_probability == cutoff).sum(1).cpu().tolist(),
                             "sigmoid_one_count": (seed_probability == 1).sum(1).cpu().tolist()}
            fused_masks = [fuse_query_mask_logits(
                as_query_mask_logits(text), as_query_mask_logits(query), alpha).cpu()
                for text, query, alpha in zip(outputs["last_pred_masks"],
                                             outputs["sp_last_pred_masks"],
                                             outputs["adaptive_weights"])]
            swa_token_ids = []
            for row, raw in enumerate(trace.pop("swa_raw")):
                weight = raw.softmax(1)
                weight = torch.where(torch.isnan(weight), torch.zeros_like(weight), weight)
                token_score = (weight * ~outputs["text_attention_mask"][row].unsqueeze(-1)).sum(-1)
                swa_token_ids.append(int(token_score[0].argmax()))
            trace["visual_backbone"] = outputs["fp2_features"].detach().cpu()
            return {"boxes": boxes.cpu(), "scores": scores.cpu(),
                    "valid": valid.cpu(), "query": selected.cpu(),
                    "point_ids": point_ids.cpu(), "seed_ids": outputs["seed_inds"].cpu(),
                    "seed_logits": outputs["seeds_obj_cls_logits"].squeeze(1).cpu(),
                    "seed_topk_ties": seed_topk_ties,
                    "padding_mask": outputs["text_attention_mask"].cpu(),
                    "token_ids": outputs["tokenized"]["input_ids"].cpu(),
                    "swa_token_ids": swa_token_ids, "trace": trace,
                    "fused_mask_logits": fused_masks,
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
    for hook in hooks:
        hook.remove()

    def delta(before, after):
        values = (before - after).abs()
        return {"max_abs": float(values.max()), "mean_abs": float(values.mean())}

    def identity_compare(before, after):
        rows = []
        for row, row_id in enumerate(row_ids):
            ids_a, ids_b = before["point_ids"][row].tolist(), after["point_ids"][row].tolist()
            ia, ib = shared_query_indices(ids_a, ids_b)
            qa, qb = int(before["query"][row]), int(after["query"][row])
            aligned = {"fit_row_id": row_id, "common_query_seeds": len(ia),
                       "removed_point_ids": sorted(set(ids_a) - set(ids_b)),
                       "added_point_ids": sorted(set(ids_b) - set(ids_a)),
                       "query_order_bit_changes": sum(a != b for a, b in zip(ids_a, ids_b)),
                       "selected_point_id_before": ids_a[qa] if qa >= 0 else None,
                       "selected_point_id_after": ids_b[qb] if qb >= 0 else None}
            if ia:
                aligned["common_seed_box_delta"] = delta(before["boxes"][row, ia], after["boxes"][row, ib])
                aligned["common_seed_score_delta"] = delta(before["scores"][row, ia], after["scores"][row, ib])
                aligned["common_seed_valid_bit_changes"] = int((before["valid"][row, ia] != after["valid"][row, ib]).sum())
                aligned["common_seed_fused_mask_delta"] = delta(before["fused_mask_logits"][row][ia], after["fused_mask_logits"][row][ib])
                aligned["common_seed_decoder_delta"] = {
                    name: delta(before["trace"][name][row, ia], after["trace"][name][row, ib])
                    for name in before["trace"] if name.startswith("decoder_")}
            if qa >= 0 and qb >= 0:
                aligned["selected_box_before"] = before["boxes"][row, qa].tolist()
                aligned["selected_box_after"] = after["boxes"][row, qb].tolist()
                aligned["selected_box_delta"] = delta(before["boxes"][row, qa], after["boxes"][row, qb])
                mask_a, mask_b = before["fused_mask_logits"][row][qa], after["fused_mask_logits"][row][qb]
                aligned["selected_fused_mask_delta"] = delta(mask_a, mask_b)
                changed_sp = (mask_a > 0) != (mask_b > 0)
                aligned["selected_mask_superpoint_bit_changes"] = int(changed_sp.sum())
                aligned["selected_mask_input_point_bit_changes"] = int(changed_sp[batch["superpoint"][row].long().cpu()].sum())
            rows.append(aligned)
        valid_a, valid_b = ~before["padding_mask"], ~after["padding_mask"]
        assert torch.equal(before["token_ids"][valid_a], after["token_ids"][valid_b])
        stages = {}
        for name in before["trace"]:
            if name.startswith("decoder_"):
                continue
            a, b = before["trace"][name], after["trace"][name]
            if name == "text_projector" or name.endswith("_text"):
                a, b = a[valid_a], b[valid_b]
            stages[name] = delta(a, b)
        return {"rows": rows, "before_query_stage_deltas": stages,
                "backbone_seed_id_bit_changes": int((before["seed_ids"] != after["seed_ids"]).sum()),
                "seed_logit_delta": delta(before["seed_logits"], after["seed_logits"]),
                "seed_topk_ties_before": before["seed_topk_ties"],
                "seed_topk_ties_after": after["seed_topk_ties"],
                "swa_selected_token_before": before["swa_token_ids"],
                "swa_selected_token_after": after["swa_token_ids"]}

    def compare(before, after):
        return {
            "query_before": before["query"].tolist(), "query_after": after["query"].tolist(),
            "query_changes": int((before["query"] != after["query"]).sum()),
            "valid_query_bit_changes": int((before["valid"] != after["valid"]).sum()),
            "box_max_abs_delta": float((before["boxes"] - after["boxes"]).abs().max()),
            "score_max_abs_delta": float((before["scores"] - after["scores"]).abs().max()),
            "text_mask_logit_max_abs_delta": max(float((a-b).abs().max()) for a,b in zip(before["text_mask_logits"], after["text_mask_logits"])),
            "query_mask_logit_max_abs_delta": max(float((a-b).abs().max()) for a,b in zip(before["query_mask_logits"], after["query_mask_logits"])),
            "identity_trace": identity_compare(before, after),
        }

    result = {"schema": "mcln-real-scene-padding-intervention-v2", "checkpoint_sha256": CHECKPOINT_SHA,
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
