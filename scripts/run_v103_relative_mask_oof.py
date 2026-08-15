#!/usr/bin/env python
"""Scene-disjoint train-only OOF gate for V103 relative mask transitions."""

import argparse
import json
from pathlib import Path

import torch

from models.rec_joint_box_mask import (
    LEGACY_MASK_POLICY_INDEX,
    MASK_POLICY_COUNT,
    build_mask_policy_feature_names,
)
from models.rec_relative_mask_policy import (
    ALLOWED_MASK_POLICY_INDICES,
    RelativeMaskTransitionPostprocessor,
    compute_relative_mask_policy_loss,
    select_relative_mask_policy_ensemble,
)
from scripts.cache_scanrefer_mask_policy_features import (
    load_mask_policy_feature_cache,
)
from scripts.run_v102_mask_only_oof import (
    acceptance_gate,
    capture_readonly_file_identity,
    file_sha256,
    fit_fold_normalization,
    metric_delta,
    normalize_batch,
    scene_block_bootstrap_lower_bounds,
    set_deterministic,
    tensor_sha256,
    write_exclusive_json,
    write_exclusive_torch,
)
from scripts.train_scanrefer_joint_box_mask import (
    _compute_baseline_scores,
    _load_all_training_sources,
)
from scripts.train_scanrefer_rec_hierarchical_reranker import (
    AUTHORITATIVE_BACKBONE_PATH,
    capture_immutable_artifact_identities,
)


SCHEMA = "rec-v103-relative-mask-transition-scene-oof-v1"
VERSION = 1
EXPECTED_ROW_COUNT = 36665
EXPECTED_SCENE_COUNT = 562
EXPECTED_FOLD_COUNT = 5
EXPECTED_V101_PREDICTION_SHA256 = (
    "b81664e65d64dad7058f8f252d990d4ab11dd8c00746c64a918bb120b6434c99"
)
HIDDEN_DIM = 128
DROPOUT = 0.1
SEEDS = (0, 1, 2)
EPOCHS = 12
BATCH_SIZE = 128
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-3
GRAD_CLIP_NORM = 1.0
TARGET_TEMPERATURE = 0.25
AGGREGATE_MARGIN = 0.02
BOOTSTRAP_SAMPLES = 10000


def fit_fold_model(geometry_features, variant_valid, mask_features,
                   policy_ious, fit_indices, statistics, device, seed):
    """Train exactly one preregistered seed on four scene folds."""
    if int(seed) not in SEEDS:
        raise ValueError("V103 seed is not preregistered")
    set_deterministic(int(seed), device)
    model = RelativeMaskTransitionPostprocessor(
        hidden_dim=HIDDEN_DIM, dropout=DROPOUT
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    fit_indices = torch.as_tensor(fit_indices, dtype=torch.long).cpu()
    history = []
    for epoch in range(EPOCHS):
        model.train()
        order = fit_indices[torch.randperm(
            int(fit_indices.numel()), generator=generator
        )]
        totals = {name: 0.0 for name in (
            "loss", "delta_iou", "transition025", "transition050",
            "listwise", "regret",
        )}
        batches = 0
        for start in range(0, int(order.numel()), BATCH_SIZE):
            indices = order[start:start + BATCH_SIZE]
            geometry = geometry_features.index_select(0, indices).to(device)
            valid = variant_valid.index_select(0, indices).to(device)
            mask = mask_features.index_select(0, indices).to(device)
            labels = policy_ious.index_select(0, indices).to(device)
            geometry, mask = normalize_batch(
                geometry, mask, valid, statistics
            )
            outputs = model(geometry, mask, valid)
            loss, components = compute_relative_mask_policy_loss(
                outputs,
                labels,
                valid.any(dim=2),
                target_temperature=TARGET_TEMPERATURE,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), GRAD_CLIP_NORM
            )
            optimizer.step()
            totals["loss"] += float(loss.detach().item())
            for name, value in components.items():
                totals[name] += float(value.detach().item())
            batches += 1
        history.append({
            "epoch": epoch + 1,
            **{name: value / float(batches)
               for name, value in totals.items()},
        })
    model.eval().requires_grad_(False)
    return model, history


def predict_seed_outputs(model, geometry_features, variant_valid,
                         mask_features, held_indices, statistics, device):
    """Materialize one held-fold seed prediction without using labels."""
    held_indices = torch.as_tensor(held_indices, dtype=torch.long).cpu()
    delta_iou = []
    probabilities = []
    query_valid = []
    with torch.inference_mode():
        for start in range(0, int(held_indices.numel()), BATCH_SIZE):
            indices = held_indices[start:start + BATCH_SIZE]
            geometry = geometry_features.index_select(0, indices).to(device)
            valid = variant_valid.index_select(0, indices).to(device)
            mask = mask_features.index_select(0, indices).to(device)
            geometry, mask = normalize_batch(
                geometry, mask, valid, statistics
            )
            outputs = model(geometry, mask, valid)
            delta_iou.append(outputs["delta_iou"].cpu())
            probabilities.append(
                outputs["transition_probabilities"].cpu()
            )
            query_valid.append(outputs["query_valid"].cpu())
    return {
        "delta_iou": torch.cat(delta_iou, dim=0).float(),
        "transition_probabilities": torch.cat(
            probabilities, dim=0
        ).float(),
        "query_valid": torch.cat(query_valid, dim=0).bool(),
    }


def _selection_from_seed_outputs(seed_outputs, parents):
    prediction = select_relative_mask_policy_ensemble(
        seed_outputs,
        parents,
        aggregate_margin=AGGREGATE_MARGIN,
    )
    if not torch.equal(
            prediction["selected_parent_positions"], parents):
        raise RuntimeError("V103 changed the frozen REC parent query")
    return prediction


def _validate_inputs(joined_rows, feature_rows, all_state, sidecar,
                     variant_valid):
    if (len(joined_rows) != EXPECTED_ROW_COUNT
            or len(feature_rows) != EXPECTED_ROW_COUNT
            or all_state["features"].shape != (
                EXPECTED_ROW_COUNT, 16 * 7, 179)
            or all_state["valid_mask"].shape != (
                EXPECTED_ROW_COUNT, 16 * 7)
            or all_state["mask_policy_ious"].shape != (
                EXPECTED_ROW_COUNT, 16, MASK_POLICY_COUNT)):
        raise ValueError("V103 train source coverage changed")
    expected_indices = torch.as_tensor(
        sidecar["dataset_indices"], dtype=torch.long
    )
    for index, (joined, feature) in enumerate(zip(joined_rows, feature_rows)):
        identity = (
            int(joined["dataset_index"]),
            str(joined["scan_id"]),
            int(joined["target_id"]),
        )
        if (identity != (
                int(feature["dataset_index"]),
                feature["scan_id"],
                int(feature["target_id"]))
                or identity[0] != int(expected_indices[index].item())
                or not torch.equal(
                    torch.as_tensor(joined["query_indices"]).long(),
                    feature["query_indices"],
                )
                or not torch.equal(
                    torch.as_tensor(joined["candidate_valid"]).bool(),
                    feature["candidate_valid"],
                )):
            raise ValueError("V103 cache row alignment changed")
    feature_valid = torch.stack([
        row["candidate_valid"] for row in feature_rows
    ])
    if not torch.equal(variant_valid.any(dim=2), feature_valid):
        raise ValueError("V103 query validity changed")
    return expected_indices


def main(argv=None):
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--base-cache", required=True)
    parser.add_argument("--geometry-cache", required=True)
    parser.add_argument("--joint-cache", required=True)
    parser.add_argument("--mask-feature-cache", required=True)
    parser.add_argument("--v101-sidecar", required=True)
    parser.add_argument("--parent-checkpoint", required=True)
    parser.add_argument("--geometry-checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--decision-output")
    parser.add_argument("--device", default="cuda:0", choices=("cuda:0",))
    parser.add_argument("--runtime-batch-size", type=int, default=512)
    args = parser.parse_args(argv)
    if args.runtime_batch_size <= 0:
        parser.error("--runtime-batch-size must be positive")
    output = Path(args.output).expanduser().absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError(str(output))
    decision_output = None
    if args.decision_output is not None:
        decision_output = Path(args.decision_output).expanduser().absolute()
        if decision_output.exists() or decision_output.is_symlink():
            raise FileExistsError(str(decision_output))
    device = torch.device(args.device)
    if not torch.cuda.is_available():
        raise ValueError("V103 OOF requires CUDA")

    protected_paths = {
        "backbone": Path(AUTHORITATIVE_BACKBONE_PATH).resolve(),
        "parent": Path(args.parent_checkpoint).expanduser().resolve(),
        "geometry": Path(args.geometry_checkpoint).expanduser().resolve(),
    }
    protected_before = capture_immutable_artifact_identities(protected_paths)
    protected_before["v101_sidecar"] = capture_readonly_file_identity(
        args.v101_sidecar, "v101_sidecar"
    )

    joined_rows, base_manifest, geometry_manifest, joint_manifest = (
        _load_all_training_sources(args)
    )
    all_state = _compute_baseline_scores(
        joined_rows, args, args.device, base_manifest, geometry_manifest
    )
    feature_rows, feature_manifest = load_mask_policy_feature_cache(
        args.mask_feature_cache, require_full=True
    )
    sidecar_path = Path(args.v101_sidecar).expanduser().absolute()
    sidecar = torch.load(sidecar_path, map_location="cpu")
    if (sidecar.get("schema") != "rec-v101-oof-row-decisions-v1"
            or sidecar.get("validation_data_accessed") is not False
            or sidecar.get("prediction_sha256")
            != EXPECTED_V101_PREDICTION_SHA256
            or sidecar.get("row_count") != EXPECTED_ROW_COUNT
            or sidecar.get("scene_count") != EXPECTED_SCENE_COUNT):
        raise ValueError("V101 OOF sidecar provenance changed")

    geometry_features = all_state["features"].reshape(
        EXPECTED_ROW_COUNT, 16, 7, 179
    ).float().contiguous()
    variant_valid = all_state["valid_mask"].reshape(
        EXPECTED_ROW_COUNT, 16, 7
    ).bool().contiguous()
    mask_features = torch.stack([
        row["mask_policy_features"] for row in feature_rows
    ]).float().contiguous()
    policy_ious = all_state["mask_policy_ious"].float().contiguous()
    expected_indices = _validate_inputs(
        joined_rows, feature_rows, all_state, sidecar, variant_valid
    )
    selected_parents = torch.as_tensor(
        sidecar["selected_parent_positions"], dtype=torch.long
    ).contiguous()
    fold_ids = torch.as_tensor(sidecar["fold_ids"], dtype=torch.long)
    scenes = list(all_state["scene_ids"])
    if (len(set(scenes)) != EXPECTED_SCENE_COUNT
            or set(fold_ids.tolist()) != set(range(EXPECTED_FOLD_COUNT))):
        raise ValueError("V103 scene/fold coverage changed")

    selected_policies = torch.full(
        (EXPECTED_ROW_COUNT,), LEGACY_MASK_POLICY_INDEX, dtype=torch.long
    )
    proposal_policies = selected_policies.clone()
    accepted = torch.zeros(EXPECTED_ROW_COUNT, dtype=torch.bool)
    worst_aggregate = torch.zeros(EXPECTED_ROW_COUNT, dtype=torch.float32)
    worst_delta_iou = torch.zeros(EXPECTED_ROW_COUNT, dtype=torch.float32)
    worst_effects = torch.zeros(EXPECTED_ROW_COUNT, 2, dtype=torch.float32)
    predicted_parents = torch.full_like(selected_parents, -1)
    seed_selected_policies = {
        int(seed): selected_policies.clone() for seed in SEEDS
    }
    fold_records = []
    for held in range(EXPECTED_FOLD_COUNT):
        fit_indices = fold_ids.ne(held).nonzero(as_tuple=False).reshape(-1)
        held_indices = fold_ids.eq(held).nonzero(as_tuple=False).reshape(-1)
        fit_scenes = {scenes[index] for index in fit_indices.tolist()}
        held_scenes = {scenes[index] for index in held_indices.tolist()}
        if fit_scenes & held_scenes:
            raise RuntimeError("V103 scene leakage between fit and held fold")
        statistics = fit_fold_normalization(
            geometry_features,
            variant_valid,
            mask_features,
            variant_valid.any(dim=2),
            fit_indices,
        )
        held_seed_outputs = []
        histories = {}
        for seed in SEEDS:
            model, history = fit_fold_model(
                geometry_features,
                variant_valid,
                mask_features,
                policy_ious,
                fit_indices,
                statistics,
                device,
                seed,
            )
            held_seed_outputs.append(predict_seed_outputs(
                model,
                geometry_features,
                variant_valid,
                mask_features,
                held_indices,
                statistics,
                device,
            ))
            histories[int(seed)] = history
            print(json.dumps({
                "completed_fold_seed": {
                    "held_fold": held,
                    "seed": int(seed),
                    "final_epoch": history[-1],
                }
            }, sort_keys=True), flush=True)
            del model
            torch.cuda.empty_cache()

        held_parents = selected_parents.index_select(0, held_indices)
        prediction = _selection_from_seed_outputs(
            held_seed_outputs, held_parents
        )
        selected_policies[held_indices] = prediction[
            "selected_policy_indices"
        ]
        proposal_policies[held_indices] = prediction[
            "proposal_policy_indices"
        ]
        accepted[held_indices] = prediction["accepted"]
        worst_aggregate[held_indices] = prediction[
            "worst_aggregate_gain"
        ]
        worst_delta_iou[held_indices] = prediction["worst_delta_iou"]
        worst_effects[held_indices] = prediction["worst_effects"]
        predicted_parents[held_indices] = prediction[
            "selected_parent_positions"
        ]

        held_rows = torch.arange(int(held_indices.numel()))
        held_labels = policy_ious[held_indices, held_parents]
        before = held_labels[:, LEGACY_MASK_POLICY_INDEX]
        after = held_labels[
            held_rows, prediction["selected_policy_indices"]
        ]
        individual_seed_records = []
        for position, seed in enumerate(SEEDS):
            seed_output = held_seed_outputs[position]
            seed_prediction = _selection_from_seed_outputs(
                [seed_output, seed_output, seed_output], held_parents
            )
            seed_selected_policies[int(seed)][held_indices] = (
                seed_prediction["selected_policy_indices"]
            )
            seed_after = held_labels[
                held_rows, seed_prediction["selected_policy_indices"]
            ]
            individual_seed_records.append({
                "seed": int(seed),
                "accepted_switches": int(
                    seed_prediction["accepted"].sum().item()
                ),
                **metric_delta(before, seed_after),
            })
        fold_record = {
            "held_fold": held,
            "fit_rows": int(fit_indices.numel()),
            "held_rows": int(held_indices.numel()),
            "fit_scenes": len(fit_scenes),
            "held_scenes": len(held_scenes),
            "accepted_switches": int(prediction["accepted"].sum().item()),
            "normalization_sha256": statistics["sha256"],
            "final_epochs": {
                str(seed): histories[int(seed)][-1] for seed in SEEDS
            },
            "individual_seed_diagnostics": individual_seed_records,
            **metric_delta(before, after),
        }
        fold_records.append(fold_record)
        print(json.dumps({
            "completed_fold": fold_record
        }, sort_keys=True), flush=True)
        del held_seed_outputs

    if not torch.equal(predicted_parents, selected_parents):
        raise RuntimeError("V103 changed REC parent identity in OOF")
    rows = torch.arange(EXPECTED_ROW_COUNT)
    selected_labels = policy_ious[rows, selected_parents]
    before = selected_labels[:, LEGACY_MASK_POLICY_INDEX]
    after = selected_labels[rows, selected_policies]
    metrics = metric_delta(before, after)
    individual_seed_metrics = []
    for seed in SEEDS:
        seed_after = selected_labels[rows, seed_selected_policies[int(seed)]]
        individual_seed_metrics.append({
            "seed": int(seed),
            "accepted_switches": int(seed_selected_policies[int(seed)].ne(
                LEGACY_MASK_POLICY_INDEX
            ).sum().item()),
            **metric_delta(before, seed_after),
        })
    bootstrap = scene_block_bootstrap_lower_bounds(
        before,
        after,
        scenes,
        seed=0,
        samples=BOOTSTRAP_SAMPLES,
    )
    rec_identity_digest = tensor_sha256(
        selected_parents, torch.as_tensor(sidecar["selected_indices"])
    )
    rec_identity_digest_after = tensor_sha256(
        predicted_parents, torch.as_tensor(sidecar["selected_indices"])
    )
    gate = acceptance_gate(
        metrics,
        fold_records,
        bootstrap,
        rec_identity_digest == rec_identity_digest_after,
    )
    protected_after = capture_immutable_artifact_identities(protected_paths)
    protected_after["v101_sidecar"] = capture_readonly_file_identity(
        args.v101_sidecar, "v101_sidecar"
    )
    if protected_after != protected_before:
        raise RuntimeError("protected artifacts changed during V103 OOF")

    helper_path = Path(__file__).with_name("run_v102_mask_only_oof.py")
    report = {
        "schema": SCHEMA,
        "version": VERSION,
        "deployable": bool(gate["passed"]),
        "validation_data_accessed": False,
        "inference_uses_ground_truth": False,
        "protocol": {
            "selection": (
                "frozen_v101_parent_then_three_seed_worst_case_"
                "relative_mask_policy"
            ),
            "source_policy_count": MASK_POLICY_COUNT,
            "allowed_policy_indices": list(ALLOWED_MASK_POLICY_INDICES),
            "forbidden_policy_indices": [0, 1, 2, 3],
            "legacy_policy_index": LEGACY_MASK_POLICY_INDEX,
            "aggregate_margin": AGGREGATE_MARGIN,
            "seeds": list(SEEDS),
            "seed_selection": False,
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "gradient_clip_norm": GRAD_CLIP_NORM,
            "target_temperature": TARGET_TEMPERATURE,
            "class_reweighting": False,
            "bootstrap_samples": BOOTSTRAP_SAMPLES,
            "runtime_materialization_batch_size": args.runtime_batch_size,
            "grid_search": False,
        },
        "coverage": {
            "rows": EXPECTED_ROW_COUNT,
            "scenes": EXPECTED_SCENE_COUNT,
            "folds": EXPECTED_FOLD_COUNT,
            "trained_models": EXPECTED_FOLD_COUNT * len(SEEDS),
        },
        "metrics": metrics,
        "folds": fold_records,
        "individual_seed_diagnostics": individual_seed_metrics,
        "bootstrap": bootstrap,
        "gate": gate,
        "diagnostics": {
            "accepted_switches": int(accepted.sum().item()),
            "proposal_changes": int(proposal_policies.ne(
                LEGACY_MASK_POLICY_INDEX
            ).sum().item()),
            "selected_policy_counts": [
                int(value) for value in torch.bincount(
                    selected_policies, minlength=MASK_POLICY_COUNT
                ).tolist()
            ],
            "prediction_sha256": tensor_sha256(
                proposal_policies,
                selected_policies,
                accepted,
                worst_aggregate,
                worst_delta_iou,
                worst_effects,
            ),
            "rec_identity_sha256_before": rec_identity_digest,
            "rec_identity_sha256_after": rec_identity_digest_after,
            "v101_prediction_sha256": EXPECTED_V101_PREDICTION_SHA256,
        },
        "input_sha256": {
            "backbone": all_state["geometry_artifact"].get(
                "checkpoint_sha256"
            ),
            "parent": all_state["parent_artifact_sha256"],
            "geometry": all_state["geometry_artifact_sha256"],
            "v101_sidecar": file_sha256(sidecar_path),
            "base_manifest": file_sha256(
                Path(args.base_cache) / "manifest.json"
            ),
            "geometry_manifest": file_sha256(
                Path(args.geometry_cache) / "manifest.json"
            ),
            "joint_manifest": file_sha256(
                Path(args.joint_cache) / "manifest.json"
            ),
            "mask_feature_manifest": file_sha256(
                Path(args.mask_feature_cache) / "manifest.json"
            ),
        },
        "feature_contract": {
            "mask_feature_names": build_mask_policy_feature_names(),
            "mask_feature_manifest_content_sha256": feature_manifest[
                "content_sha256"
            ],
            "joint_label_manifest_content_sha256": joint_manifest[
                "content_sha256"
            ],
        },
        "source_sha256": {
            "driver": file_sha256(__file__),
            "model": file_sha256(
                Path(__file__).resolve().parents[1]
                / "models" / "rec_relative_mask_policy.py"
            ),
            "shared_oof_helpers": file_sha256(helper_path),
            "feature_cache": file_sha256(
                Path(__file__).with_name(
                    "cache_scanrefer_mask_policy_features.py"
                )
            ),
        },
        "protected_before": protected_before,
        "protected_after": protected_after,
    }

    decision_sha = None
    if decision_output is not None:
        decision_sidecar = {
            "schema": "rec-v103-relative-mask-oof-decisions-v1",
            "version": 1,
            "validation_data_accessed": False,
            "inference_uses_ground_truth": False,
            "row_count": EXPECTED_ROW_COUNT,
            "scene_count": EXPECTED_SCENE_COUNT,
            "fold_count": EXPECTED_FOLD_COUNT,
            "dataset_indices": expected_indices.clone(),
            "scene_ids": list(scenes),
            "fold_ids": fold_ids.clone(),
            "selected_parent_positions": selected_parents.clone(),
            "proposal_policy_indices": proposal_policies.clone(),
            "selected_policy_indices": selected_policies.clone(),
            "single_seed_selected_policy_indices": {
                int(seed): value.clone()
                for seed, value in seed_selected_policies.items()
            },
            "accepted": accepted.clone(),
            "worst_aggregate_gain": worst_aggregate.clone(),
            "worst_delta_iou": worst_delta_iou.clone(),
            "worst_effects": worst_effects.clone(),
            "selected_parent_policy_ious": selected_labels.clone(),
            "before_ious": before.clone(),
            "after_ious": after.clone(),
            "prediction_sha256": report["diagnostics"][
                "prediction_sha256"
            ],
            "rec_identity_sha256": rec_identity_digest,
            "input_sha256": dict(report["input_sha256"]),
            "source_sha256": dict(report["source_sha256"]),
        }
        decision_output.parent.mkdir(parents=True, exist_ok=True)
        decision_sha = write_exclusive_torch(
            decision_output, decision_sidecar
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    output_sha = write_exclusive_json(output, report)
    print(json.dumps({
        "output": str(output),
        "sha256": output_sha,
        "decision_output": (
            str(decision_output) if decision_output is not None else None
        ),
        "decision_sha256": decision_sha,
        "deployable": report["deployable"],
        "metrics": metrics,
        "individual_seed_diagnostics": individual_seed_metrics,
        "bootstrap": bootstrap,
        "gate": gate,
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
