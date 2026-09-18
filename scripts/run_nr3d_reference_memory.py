"""One fixed epoch shared by the four reference-memory/readout controls."""

import argparse
import ast
import copy
import json
from pathlib import Path
import random
import sys
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    opt = parser.parse_args()
    opt.output.mkdir(parents=True, exist_ok=False)
    addon = Path(__file__).resolve().parents[1]
    manifest = json.loads((addon / "input_manifest.json").read_text())
    source = Path(manifest["model_source"])
    sys.path.insert(0, str(source))

    import numpy as np
    import torch
    import models
    import scripts
    models.__path__.insert(0, str(addon / "models"))
    scripts.__path__ = [str(addon / "scripts")] + list(scripts.__path__)
    from models.candidate_edge_adapter import build_candidate_edge_inputs
    from models.candidate_edge_direct_scorer import CandidateEdgeDirectScorer
    from models.candidate_reference_scorer import CandidateReferenceScorer
    from models.losses import _iou3d_par, box_cxcyczwhd_to_xyzxyz
    from models.source_choice_adapter import compute_default_source_scores
    from scripts.nr3d_reference_memory_contract import ARMS, CONTRACT, covered_ranking_loss, decide, split_rows
    from scripts.nr3d_view_pair_contract import CHECKPOINT_SHA, digest_ids
    from scripts.run_nr3d_view_pair_role import file_sha, read_train_rows, write_json
    from main_utils import parse_option
    from train_dist_mod import TrainTester
    from src.joint_det_dataset import Joint3DDataset
    from src.grounding_evaluator import GroundingEvaluator

    def verify_inputs():
        assert file_sha(opt.checkpoint) == CHECKPOINT_SHA
        source_manifest = source / "g0_source_manifest.json"
        assert file_sha(source_manifest) == manifest["source_manifest_sha256"]
        for relative, expected in json.loads(source_manifest.read_text())["files"].items():
            assert file_sha(source / relative) == expected, relative
        for relative, expected in manifest["files"].items():
            assert file_sha(addon / relative) == expected, relative

    verify_inputs()
    assert manifest["contract"] == CONTRACT
    raw_rows = read_train_rows(Path("/root/autodl-tmp/DATA_ROOT"))
    parts = split_rows(raw_rows)
    census = {name: {"rows": len(ids),
                      "scenes": len({raw_rows[i]["scan_id"] for i in ids}),
                      "identity_sha256": digest_ids(ids)} for name, ids in parts.items()}
    assert census == manifest["census"]

    def seed_all(seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    seed_all(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    checkpoint = torch.load(str(opt.checkpoint), map_location="cpu")
    sys.argv = [sys.argv[0]]
    args = parse_option()
    vars(args).update(vars(checkpoint["config"]))
    backbone = TrainTester.get_model(args).cuda().eval()
    protected_state = {key[7:]: value for key, value in checkpoint["model"].items()}
    backbone.load_state_dict(protected_state, strict=True)
    backbone.requires_grad_(False)
    del checkpoint

    def verify_frozen_state():
        for name, value in backbone.state_dict().items():
            assert torch.equal(value.detach().cpu(), protected_state[name]), name
        assert all(parameter.grad is None for parameter in backbone.parameters())

    class ReadoutDataset(Joint3DDataset):
        def __getitem__(self, index):
            result = super().__getitem__(index)
            result["reference_id"] = np.int64(self.reference_ids[index])
            return result

    base = ReadoutDataset(dataset_dict={"nr3d": 1}, test_dataset="nr3d", split="train",
                          data_path="/root/autodl-tmp/DATA_ROOT/", use_color=args.use_color,
                          detect_intermediate=args.detect_intermediate, butd_cls=args.butd_cls,
                          skip_missing_superpoints=args.skip_missing_superpoints)
    assert len(base.annos) == len(raw_rows)
    for anno, row in zip(base.annos, raw_rows):
        assert anno["scan_id"] == row["scan_id"] and anno["target_id"] == int(row["target_id"])
    metadata = {i: {"id": i, "scan_id": row["scan_id"], "target_id": int(row["target_id"]),
                    "raw_token_count": len(ast.literal_eval(row["tokens"])),
                    "distractor_count": len(base.annos[i]["distractor_ids"])}
                for i, row in enumerate(raw_rows)}

    def seed_worker(worker_id):
        torch.set_num_threads(1)
        seed = torch.initial_seed() % 2**32
        random.seed(seed)
        np.random.seed(seed)

    loaders = {}
    for name, ids in parts.items():
        data = copy.copy(base)
        data.annos = [base.annos[i] for i in ids]
        data.reference_ids = ids
        data.augment = False
        loaders[name] = torch.utils.data.DataLoader(
            data, batch_size=CONTRACT["batch_size"], shuffle=name == "fit",
            num_workers=CONTRACT["num_workers"], pin_memory=True,
            worker_init_fn=seed_worker, drop_last=False,
            generator=torch.Generator().manual_seed(CONTRACT["loader_seed"]),
        )
    heads = {}
    for mode, specification in ARMS.items():
        seed_all(CONTRACT["init_seed"])
        head = CandidateReferenceScorer(specification["readout"]).cuda()
        head.spatial.load_state_dict(backbone.decoder[-1].self_attn.state_dict(), strict=True)
        heads[mode] = head
    for mode, head in heads.items():
        for name, value in heads["query_global"].state_dict().items():
            assert torch.equal(value, head.state_dict()[name]), (mode, name)
    for readout in ("global", "pair"):
        for name, value in heads["query_" + readout].state_dict().items():
            assert torch.equal(value, heads["object_" + readout].state_dict()[name]), name
    optimizers = {mode: torch.optim.AdamW(head.parameters(), lr=CONTRACT["lr"],
                                         weight_decay=CONTRACT["weight_decay"])
                  for mode, head in heads.items()}
    captured = {}

    def capture_query(module, arguments):
        captured["query"] = arguments[0].transpose(1, 2).detach()

    hook = backbone.x_query.register_forward_pre_hook(capture_query)
    original_decoder_forward = backbone.decoder[-1].forward

    def capture_reference(*arguments, **keywords):
        captured["object_feats"] = keywords["detected_feats"]
        captured["object_padding"] = keywords["detected_mask"]
        return original_decoder_forward(*arguments, **keywords)

    backbone.decoder[-1].forward = capture_reference

    def forward_batch(batch):
        batch = TrainTester._to_gpu(batch)
        inputs = TrainTester._get_inputs(batch)
        inputs["train"] = False
        with torch.no_grad():
            outputs = backbone(inputs)
            shared = build_candidate_edge_inputs(outputs, inputs, captured["query"],
                                                  top_k=CONTRACT["top_k"])
            references = {
                "query": {"memory_feats": shared["candidate_feats"],
                          "memory_boxes": shared["candidate_boxes"],
                          "memory_valid_mask": shared["valid_query_mask"]},
                "object": {"memory_feats": captured["object_feats"],
                           "memory_boxes": inputs["det_boxes"],
                           "memory_valid_mask": ~captured["object_padding"]},
            }
            roots = torch.cat([batch["center_label"][:, :1, :3], batch["size_gts"][:, :1]], -1)
            ious = torch.stack([_iou3d_par(box_cxcyczwhd_to_xyzxyz(root),
                                         box_cxcyczwhd_to_xyzxyz(boxes))[0][0]
                                for root, boxes in zip(roots, shared["candidate_boxes"])])
            assert torch.isfinite(ious).all()
        return batch, outputs, shared, references, ious

    preflight = {"schema": CONTRACT["schema"], "contract": CONTRACT, "census": census,
                 "checkpoint_sha256": CHECKPOINT_SHA,
                 "input_manifest_sha256": file_sha(addon / "input_manifest.json"),
                 "common_initial_state_equal": True,
                 "parameters_by_arm": {mode: sum(p.numel() for p in head.parameters())
                                       for mode, head in heads.items()},
                 "fit_batches": len(loaders["fit"]), "holdout_batches": len(loaders["holdout"]),
                 "formal_validation_dataset_constructed": False}
    write_json(opt.output / "preflight.json", preflight)
    print("R1 PREFLIGHT", json.dumps(preflight), flush=True)
    started = time.time()
    seen, supervised_rows, skipped = [], 0, 0
    steps = {mode: 0 for mode in heads}
    loss_sums = {mode: 0.0 for mode in heads}
    smoke = {}
    for batch_index, raw_batch in enumerate(loaders["fit"]):
        batch, outputs, shared, references, ious = forward_batch(raw_batch)
        seen.extend(batch["reference_id"].tolist())
        target_ious = ious.gather(1, shared["query_indices"])
        target_valid = shared["valid_query_mask"].gather(1, shared["query_indices"])
        count = int((target_valid & (target_ious > .25)).any(1).sum())
        supervised_rows += count
        if count:
            for mode, head in heads.items():
                optimizer = optimizers[mode]
                optimizer.zero_grad(set_to_none=True)
                result = head(**shared, **references[ARMS[mode]["memory"]])
                if opt.preflight_only and ARMS[mode]["memory"] == "query":
                    legacy = CandidateEdgeDirectScorer(ARMS[mode]["readout"]).cuda()
                    legacy.load_state_dict(head.state_dict(), strict=True)
                    with torch.no_grad():
                        expected = legacy(**shared)
                    torch.testing.assert_close(result["query_scores"], expected["query_scores"], rtol=1e-6, atol=1e-6)
                    del legacy, expected
                loss, actual_count = covered_ranking_loss(result["candidate_logits"], target_ious, target_valid)
                assert actual_count == count and torch.isfinite(loss)
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(head.parameters(), CONTRACT["gradient_clip_norm"])
                assert torch.isfinite(norm), mode
                if not opt.preflight_only:
                    optimizer.step()
                    steps[mode] += 1
                loss_sums[mode] += float(loss.detach()) * count
                smoke[mode] = {"loss": float(loss.detach()), "gradient_norm": float(norm)}
                del result, loss
        else:
            skipped += 1
        del batch, outputs, shared, references, ious, target_ious, target_valid
        if opt.preflight_only:
            assert count > 0, "the fixed first fit batch must exercise supervised backward"
            verify_frozen_state()
            write_json(opt.output / "smoke.json", dict(smoke, optimizer_steps=steps,
                       rows=seen, covered_rows=count, heldout_batches=0, weights_written=0,
                       query_readouts_match_legacy_on_actual_fit_tensors=True,
                       max_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),
                       elapsed_seconds=time.time() - started))
            print("R1 FIT-ONLY ZERO-STEP SMOKE PASS", json.dumps(smoke), flush=True)
            hook.remove()
            backbone.decoder[-1].forward = original_decoder_forward
            verify_inputs()
            return
        if (batch_index + 1) % 100 == 0 or batch_index + 1 == len(loaders["fit"]):
            elapsed = time.time() - started
            print("R1 FIT", json.dumps({"batch": batch_index + 1, "total": len(loaders["fit"]),
                  "rows": len(seen), "covered_rows": supervised_rows, "optimizer_steps": steps,
                  "loss_mean": {mode: value / supervised_rows for mode, value in loss_sums.items()},
                  "elapsed_seconds": elapsed,
                  "eta_fit_seconds": elapsed / (batch_index + 1) * (len(loaders["fit"]) - batch_index - 1)}), flush=True)

    assert digest_ids(seen) == census["fit"]["identity_sha256"]
    assert all(value == len(loaders["fit"]) - skipped for value in steps.values())
    verify_frozen_state()
    training = {"sample_count": len(seen), "sample_order_sha256": digest_ids(seen, ordered=True),
                "covered_rows": supervised_rows, "skipped_batches": skipped, "optimizer_steps": steps,
                "elapsed_seconds": time.time() - started, "weights": {}}
    for mode, head in heads.items():
        path = opt.output / (mode + "_final.pth")
        torch.save({"model": {key: value.detach().cpu() for key, value in head.state_dict().items()},
                    "mode": mode, "contract": CONTRACT, "backbone_sha256": CHECKPOINT_SHA}, str(path))
        training["weights"][mode] = {"name": path.name, "sha256": file_sha(path)}
        head.eval()
    write_json(opt.output / "training.json", training)

    protected_hits = {}

    class ProtectedEvaluator(GroundingEvaluator):
        def _record_position_subgroups(self, outputs, bid, threshold, found):
            index = int(outputs["reference_id"][bid])
            protected_hits.setdefault(index, {})[threshold] = bool(found[0])

    evaluator = ProtectedEvaluator(prefixes=["last_"], topks=[1],
                                   filter_non_gt_boxes=True, eval_use_selector_choice_scores=True)
    seed_all(CONTRACT["eval_seed"])
    rows = []
    eval_started = time.time()
    with (opt.output / "holdout_rows.jsonl").open("x") as row_stream, torch.no_grad():
        for batch_index, raw_batch in enumerate(loaders["holdout"]):
            batch, outputs, shared, references, ious = forward_batch(raw_batch)
            default = compute_default_source_scores(outputs, batch)
            scores = {"protected": outputs["selected_source_scores"], "default": default}
            for mode, head in heads.items():
                scores[mode] = head(**shared, **references[ARMS[mode]["memory"]])["query_scores"]
            outputs.update(batch)
            outputs["last_pred_size"] = shared["candidate_boxes"][..., 3:]
            evaluator.evaluate_bbox_by_pos_align(outputs, "last_")
            point_masks, _ = evaluator._build_mask_point_predictions(outputs, "last_")
            mask_queries = evaluator._resolve_learned_mask_queries(outputs, "last_", 256, ious.device)
            for bid, source_id in enumerate(batch["reference_id"].tolist()):
                valid = shared["valid_query_mask"][bid]
                legal_ids = valid.nonzero().reshape(-1)
                gt_mask = batch["gt_masks"][bid, 0].bool()
                predicted = point_masks[bid].bool()
                mask_ious = ((predicted & gt_mask.unsqueeze(0)).sum(-1).double()
                             / (predicted | gt_mask.unsqueeze(0)).sum(-1).double())
                assert torch.isfinite(mask_ious).all()

                def quality(query):
                    if query is None:
                        return {"query": None, "box_iou": 0.0, "mask_iou": 0.0}
                    return {"query": int(query), "box_iou": float(ious[bid, query]),
                            "mask_iou": float(mask_ious[query])}

                row = dict(metadata[source_id], legal_queries=int(valid.sum()),
                           target_points=int(gt_mask.sum()), scores={}, oracle={})
                for name, values in scores.items():
                    query = int(evaluator._position_top_indices(
                        values[bid:bid + 1], valid.unsqueeze(0), "default_query_axis", 1)[0, 0]) if legal_ids.numel() else None
                    row["scores"][name] = quality(query)
                for threshold in [.25, .5]:
                    assert protected_hits[source_id][threshold] == (row["scores"]["protected"]["box_iou"] > threshold)
                row["protected_mask_query"] = int(mask_queries[bid])
                row["protected_mask_iou"] = float(mask_ious[mask_queries[bid]])
                order = default[bid].argsort(descending=True)
                legal_order = default[bid].masked_fill(~valid, -float("inf")).argsort(descending=True)
                for label, ids in [("before_filter", order), ("after_filter", legal_order[valid[legal_order]])]:
                    row["oracle"][label] = {str(k): float(ious[bid, ids[:k]].max()) if ids.numel() else 0.0
                                              for k in [16, 32, 64, 256]}
                best = int(ious[bid].masked_fill(~valid, -1).argmax()) if legal_ids.numel() else None
                row["legal_box_oracle"] = quality(best)
                row["full_box_oracle"] = quality(int(ious[bid].argmax()))
                row["full_mask_oracle"] = quality(int(mask_ious.argmax()))
                detector_ids = batch["all_detected_bbox_label_mask"][bid].bool().nonzero().reshape(-1)
                detector_ious = _iou3d_par(
                    box_cxcyczwhd_to_xyzxyz(batch["all_detected_boxes"][bid, detector_ids]),
                    box_cxcyczwhd_to_xyzxyz(shared["candidate_boxes"][bid]))[0]
                target_ids = shared["query_indices"][bid]
                target_ids = target_ids[valid[target_ids]]
                row["object_availability_proxy"] = {
                    "detector_objects": int(detector_ids.numel()),
                    "full_256": int((detector_ious > .25).any(-1).sum()),
                    "full_legal": int((detector_ious[:, legal_ids] > .25).any(-1).sum()),
                    "target_top32": int((detector_ious[:, target_ids] > .25).any(-1).sum()),
                    "object_input_slots": int(references["object"]["memory_valid_mask"][bid].sum()),
                    "object_predicted_class_correct": int((batch["all_detected_class_ids"][bid, detector_ids]
                                                           == batch["all_class_ids"][bid, detector_ids]).sum()),
                    "is_text_anchor_ground_truth": False,
                }
                rows.append(row)
                row_stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
            if (batch_index + 1) % 100 == 0 or batch_index + 1 == len(loaders["holdout"]):
                row_stream.flush()
                print("R1 HOLDOUT", json.dumps({"batch": batch_index + 1, "total": len(loaders["holdout"]),
                      "rows": len(rows), "elapsed_seconds": time.time() - eval_started}), flush=True)
    hook.remove()
    backbone.decoder[-1].forward = original_decoder_forward
    assert digest_ids(r["id"] for r in rows) == census["holdout"]["identity_sha256"]
    verify_inputs()
    verify_frozen_state()
    summary = {}
    for mode in scores:
        mask_values = [r["protected_mask_iou"] if mode == "protected" else r["scores"][mode]["mask_iou"] for r in rows]
        summary[mode] = {"rows": len(rows),
                         "rec_hits025": sum(r["scores"][mode]["box_iou"] > .25 for r in rows),
                         "rec_hits050": sum(r["scores"][mode]["box_iou"] > .5 for r in rows),
                         "mask_hits025": sum(v > .25 for v in mask_values),
                         "mask_hits050": sum(v > .5 for v in mask_values),
                         "mask_mean_iou": sum(mask_values) / len(rows)}
    decision = decide(rows)
    write_json(opt.output / "decision.json", decision)
    receipt = dict(preflight, status="complete", training=training, summary=summary,
                   decision=decision, backbone_gradients_absent=True, protected_evaluator_row_parity=True,
                   protected_state_unchanged=True,
                   holdout_rows_sha256=file_sha(opt.output / "holdout_rows.jsonl"),
                   elapsed_seconds=time.time() - started,
                   max_gpu_allocated_bytes=torch.cuda.max_memory_allocated())
    write_json(opt.output / "receipt.json", receipt)
    print("R1 COMPLETE", json.dumps(receipt, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
