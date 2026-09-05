import hashlib
import json
import math
import os
from pathlib import Path
import sys

root = Path(__file__).parent
run = root / "results"
manifest = json.loads((root / "input_manifest.json").read_text())
receipt = json.loads((run / "receipt.json").read_text())
source = Path(manifest["model_source"])
sys.path.insert(0, str(source))
os.chdir(str(source))
from scripts.run_nr3d_view_pair_role import file_sha, read_train_rows, write_json
from scripts.nr3d_view_pair_contract import CHECKPOINT_SHA, digest_ids
assert receipt["status"] == "complete"
assert receipt["input_manifest_sha256"] == file_sha(root / "input_manifest.json")
assert receipt["contract"] == manifest["contract"]
assert receipt["census"] == manifest["census"]
assert receipt["protected_evaluator_row_parity"] and receipt["backbone_gradients_absent"]
source_manifest = source / "g0_source_manifest.json"
assert file_sha(source_manifest) == manifest["source_manifest_sha256"]
for relative, expected in json.loads(source_manifest.read_text())["files"].items():
    assert file_sha(source / relative) == expected, relative
for relative, expected in manifest["files"].items():
    assert file_sha(root / relative) == expected, relative
checkpoint = Path("/root/autodl-tmp/DATA_ROOT/output/network_v99_baseline_gt/nr3d/control/official_rec_monitor/official_best_rec025_epoch_57_0p56652741.pth")
assert file_sha(checkpoint) == CHECKPOINT_SHA
for artifact in receipt["training"]["weights"].values():
    assert file_sha(run / artifact["name"]) == artifact["sha256"]
rows_path = run / "holdout_rows.jsonl"
assert file_sha(rows_path) == receipt["holdout_rows_sha256"]
rows = [json.loads(line) for line in rows_path.read_text().splitlines()]
raw_rows = read_train_rows(Path("/root/autodl-tmp/DATA_ROOT"))
salt = manifest["contract"]["split_salt"]
expected_ids = [i for i, r in enumerate(raw_rows) if int(hashlib.sha256(
    (salt + "\0" + r["scan_id"]).encode()).hexdigest()[:8], 16) % 5 == 0]
assert [r["id"] for r in rows] == expected_ids
assert digest_ids(expected_ids) == receipt["census"]["holdout"]["identity_sha256"]
assert all(r["scan_id"] == raw_rows[r["id"]]["scan_id"] and r["target_id"] == int(raw_rows[r["id"]]["target_id"]) for r in rows)
training = receipt["training"]
assert training["sample_count"] == receipt["census"]["fit"]["rows"]
assert training["optimizer_steps"]["global"] == training["optimizer_steps"]["pair"] == receipt["fit_batches"] - training["skipped_batches"]
assert training == json.loads((run / "training.json").read_text())
summaries = {}
for mode in ["global", "pair", "protected", "default"]:
    boxes = [r["scores"][mode]["box_iou"] for r in rows]
    masks = [r["protected_mask_iou"] if mode == "protected" else r["scores"][mode]["mask_iou"] for r in rows]
    assert all(math.isfinite(v) and 0 <= v <= 1 for v in boxes + masks)
    summaries[mode] = {"rows": len(rows), "rec_hits025": sum(v > .25 for v in boxes),
                       "rec_hits050": sum(v > .5 for v in boxes),
                       "mask_hits025": sum(v > .25 for v in masks),
                       "mask_hits050": sum(v > .5 for v in masks), "mask_mean_iou": sum(masks) / len(rows)}
assert summaries == receipt["summary"]
comparisons = {}
for reference in ["global", "protected"]:
    compared = {}
    for group in ["overall", "long_13plus", "hard_2plus_distractors"]:
        selected = [r for r in rows if group == "overall" or (r["raw_token_count"] >= 13 if group == "long_13plus" else r["distractor_count"] >= 2)]
        compared[group] = {"rows": len(selected)}
        for suffix, threshold in [("025", .25), ("050", .5)]:
            before = [r["scores"][reference]["box_iou"] > threshold for r in selected]
            after = [r["scores"]["pair"]["box_iou"] > threshold for r in selected]
            fixes = sum(not a and b for a, b in zip(before, after))
            breaks = sum(a and not b for a, b in zip(before, after))
            compared[group][suffix] = {"reference_hits": sum(before), "new_hits": sum(after),
                                       "fixes": fixes, "breaks": breaks, "delta_hits": fixes - breaks}
    comparisons[reference] = compared
    assert compared == receipt["decision"]["pair_vs_" + reference]
control, parent = comparisons["global"], comparisons["protected"]
mechanism = (control["overall"]["025"]["delta_hits"] > 0 and control["overall"]["050"]["delta_hits"] >= 0
             and control["long_13plus"]["025"]["delta_hits"] > 0 and control["hard_2plus_distractors"]["025"]["delta_hits"] > 0)
practical = (parent["overall"]["025"]["delta_hits"] > 0 and parent["overall"]["050"]["delta_hits"] >= 0
             and sum(r["scores"]["pair"]["mask_iou"] for r in rows) >= sum(r["protected_mask_iou"] for r in rows))
assert receipt["decision"]["mechanism_screen_pass"] == mechanism
assert receipt["decision"]["practical_screen_pass"] == practical
assert receipt["decision"]["eligible_for_decoder_experiment"] == (mechanism and practical)
assert receipt["decision"]["formal_promotion"] is False
write_json(run / "independent_verification.json", {
    "schema": "mcln-p2-terminal-independent-v1", "status": "pass", "rows": len(rows),
    "input_and_protected_checkpoint_unchanged": True, "final_head_hashes_verified": True,
    "sample_identities_verified": True, "matched_optimizer_steps_verified": True,
    "summary_and_fix_break_arithmetic_verified": True, "registered_decision_verified": True,
    "verification_script_sha256": file_sha(__file__), "receipt_sha256": file_sha(run / "receipt.json")})
print("P2 INDEPENDENT TERMINAL VERIFICATION PASS", len(rows), flush=True)
