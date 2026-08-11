# ScanRefer REC Mask Geometry Reranker Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (- [ ]) syntax for tracking.

**Goal:** Add a deployable mask-derived geometry scorer to the frozen epoch-71 MCLN and query reranker, then prove on full official ScanRefer validation that last-layer position-alignment Top-1 reaches Acc@0.25 >= 0.60000 and Acc@0.50 >= 0.47000 without inference-time ground truth.

**Architecture:** Replay the frozen MCLN with the authoritative batch size of 12 and write a provenance-bound geometry sidecar for the existing Top-16 query caches. Train one shared pointwise scorer over the flattened 16 queries x 7 geometry variants using 152 base features, 25 geometry features, the frozen parent score, and a parent-Top1 flag (179 dimensions). At runtime, rebuild the same candidates before GT is attached, preserve the existing query-axis path exactly when the selected geometry weight is zero, and otherwise evaluate aligned geometry boxes, scores, and validity on a 112-candidate axis.

**Tech Stack:** Python 3.7, PyTorch 1.10.2, pytest, ScanRefer, frozen MCLN, existing QueryReranker and REC cache infrastructure.

**Repository Constraint:** This directory has no .git metadata. Do not create commits or worktrees. Preserve unrelated files and record fresh verification output at every checkpoint.

**Authoritative Inputs:**

- Backbone checkpoint: /root/autodl-tmp/DATA_ROOT/output/preserved_best/mcln_pair_sweep/mcln_pair_default_rankblend010_2ep_best_acc025_epoch71_0.57993.pth
- Backbone SHA-256: 3e44f4bdad3bd66ad82102032e1cb0241de57d147c0aa1d3eff9736926ef2208
- Parent reranker: /root/autodl-tmp/DATA_ROOT/output/rec_reranker/e71_top16/artifacts/reranker_h256_d010_lr1e3_seed0_final_contract.pth
- Parent SHA-256: f06f8972fdcfbbdcb799df267864ab2ebc9ca8403ff92576e2bbdb0a8c17269b
- Base train cache: /root/autodl-tmp/DATA_ROOT/output/rec_reranker/e71_top16/train (36,665 rows)
- Base val cache: /root/autodl-tmp/DATA_ROOT/output/rec_reranker/e71_top16/val (9,508 rows)
- Extraction contract: batch_size=12, num_workers=2, topk_per_source=8, K=16, G=7, geometry features=25, root-only target IoU.
- Approved design source: docs/superpowers/specs/2026-07-14-scanrefer-rec-reranker-design.md, Fallback 1.
- Audit evidence: /root/autodl-tmp/DATA_ROOT/output/rec_reranker/e71_top16/mask_geometry_audit_train256.

---

## File Map

- Modify models/rec_mask_geometry.py: keep deployable geometry construction separate from target attachment and expose rejection-code projection.
- Create models/rec_geometry_reranker.py: pure 179-dimensional feature construction, flattening, parent-prior ranking, and geometry score blending.
- Create scripts/rec_geometry_cache.py: canonical hashing, base-cache binding, geometry row/manifest validation, atomic sharding, resume, and strict joins.
- Create scripts/cache_scanrefer_rec_mask_geometry.py: deterministic frozen-MCLN sidecar extraction.
- Create scripts/train_rec_geometry_reranker.py: scene-disjoint training, train-only calibration, artifact validation/loading, and offline evaluation helpers.
- Modify main_utils.py: add geometry artifact/evaluation CLI flags.
- Modify train_dist_mod.py: expose parent compact/query scores, load both frozen scorers, build geometry before GT merge, and attach the active geometry candidate tensors.
- Modify src/grounding_evaluator.py: make position alignment operate on an explicitly resolved candidate axis.
- Create tests/test_rec_geometry_reranker.py.
- Create tests/test_rec_geometry_cache.py.
- Create tests/test_cache_scanrefer_rec_mask_geometry.py.
- Create tests/test_train_rec_geometry_reranker.py.
- Create tests/test_grounding_evaluator_rec_geometry.py.
- Modify tests/test_rec_reranker_runtime.py for the parent-output refactor and GT-isolation spy.

### Task 1: Geometry Targets And Rejection-Code Projection

**Files:**
- Modify: models/rec_mask_geometry.py
- Modify: tests/test_rec_mask_geometry.py

- [x] **Step 1: Write failing target-isolation tests**

Add tests which build a deployable geometry batch with shape [B,K,G,6], call a new target attachment helper, and prove that target IoUs are [B,K,G], invalid candidates receive zero IoU, strict threshold labels use >0.25 and >0.50, and model_inputs is unchanged:

    def test_attach_rec_mask_geometry_targets_is_root_only_and_non_mutating():
        original_model_inputs = candidate_batch["model_inputs"]
        geometry = build_rec_mask_geometry_candidates(
            end_points_without_gt, inputs, candidate_batch
        )
        targeted = attach_rec_mask_geometry_targets(
            geometry, end_points_with_two_gt_boxes, root_only=True
        )
        assert targeted["geometry_ious"].shape == geometry["valid_mask"].shape
        assert targeted["threshold_labels"].shape == (
            geometry["valid_mask"].shape + (2,)
        )
        assert candidate_batch["model_inputs"] is original_model_inputs
        assert "geometry_ious" not in geometry

- [x] **Step 2: Write a failing rejection-code projection test**

Construct mask_diagnostics with two source/threshold groups and assert:

    codes = project_variant_rejection_codes(geometry_batch)
    assert codes.shape == geometry_batch["valid_mask"].shape
    assert codes.dtype == torch.int16
    assert torch.equal(codes[..., regressed_index], torch.zeros_like(
        codes[..., regressed_index]
    ))

The helper must select the quantile column named by each variant config. It must reject a missing source group, threshold, quantile, or a code outside int16 range.

- [x] **Step 3: Run the focused tests and verify RED**

Run:

    /root/miniconda3/envs/bdetr/bin/python -m pytest tests/test_rec_mask_geometry.py -q

Expected: import or attribute failures for attach_rec_mask_geometry_targets and project_variant_rejection_codes.

- [x] **Step 4: Implement the two pure helpers**

Add these public functions without changing build_rec_mask_geometry_candidates:

    def attach_rec_mask_geometry_targets(geometry_batch, end_points,
                                         root_only=True):
        boxes = geometry_batch["boxes"]
        valid = geometry_batch["valid_mask"].bool()
        flat_boxes = boxes.reshape(boxes.shape[0], -1, 6)
        gt_boxes = torch.cat([
            end_points["center_label"][..., :3].float(),
            end_points["size_gts"].float(),
        ], dim=-1)
        gt_mask = end_points["box_label_mask"]
        if root_only:
            gt_boxes = gt_boxes[:, :1]
            gt_mask = gt_mask[:, :1]
        ious = compute_query_ious(flat_boxes, gt_boxes, gt_mask).reshape(
            valid.shape
        )
        ious = ious.masked_fill(~valid, 0.0)
        result = dict(geometry_batch)
        result["geometry_ious"] = ious
        result["threshold_labels"] = torch.stack(
            [ious > 0.25, ious > 0.50], dim=-1
        )
        return result

    def project_variant_rejection_codes(geometry_batch):
        # Map each non-regressed variant to the matching
        # source/threshold diagnostic and its declared quantile column.
        # Return a finite CPU/GPU-preserving int16 tensor [B,K,G].

Import compute_query_ious from models.rec_reranker. Keep all GT reads inside the attachment helper.

- [x] **Step 5: Run focused geometry tests**

Run:

    /root/miniconda3/envs/bdetr/bin/python -m pytest tests/test_rec_mask_geometry.py -q

Expected: all tests pass.

### Task 2: Pure Flat Geometry Scoring Contract

**Files:**
- Create: models/rec_geometry_reranker.py
- Create: tests/test_rec_geometry_reranker.py

- [x] **Step 1: Write failing 179-dimensional schema tests**

Use B=2, K=3, G=2, Dq=4, Dg=5 in the small test and assert the general formula, flatten order, and name order:

    flat = build_rec_geometry_model_inputs(
        base_features, geometry_features, parent_scores,
        parent_top1_mask, geometry_valid,
        base_feature_names, geometry_feature_names,
    )
    assert flat["features"].shape == (2, 6, 11)
    assert flat["valid_mask"].shape == (2, 6)
    assert flat["feature_names"] == (
        tuple(base_feature_names)
        + tuple(geometry_feature_names)
        + ("parent_score", "parent_is_deployed_top1")
    )

Add the production assertion 152 + 25 + 2 == 179 and prove query-major, variant-minor order by checking flat index query_idx * G + variant_idx.

- [x] **Step 2: Write failing parent-path and tie tests**

Create tied compact parent scores whose query indices are in a different order from the full query axis. Assert:

    parent = build_deployed_parent_state(
        compact_scores, query_indices, candidate_valid, num_queries=8
    )
    assert torch.equal(parent["top1_query_index"], expected_query_axis_top1)
    assert parent["parent_top1_mask"].shape == compact_scores.shape

Add a flat prior test requiring every valid (query, variant) pair to have a unique rank, all regressed variants to precede non-regressed variants for one query, and query ordering to come from the deployed scattered [B,Q] ordering.

- [x] **Step 3: Write failing zero-weight fallback and invalidity tests**

The score selector must return an explicit parent_path flag at weight zero and must never flatten/re-sort the parent candidate set:

    selected = blend_rec_geometry_scores(
        parent_state, learned_logits, geometry_valid,
        geometry_weight=0.0, regressed_variant_index=0
    )
    assert selected["use_parent_query_axis"] is True
    assert torch.equal(selected["query_scores"], parent_state["query_scores"])
    assert "flat_scores" not in selected

For nonzero weight, assert invalid candidates are -inf, all valid scores are finite, and every row has at least one valid candidate. Add a test where geometry_weight=1.0 and learned logits tie; stable_flat_descending_indices must return the lower flat index first.

- [x] **Step 4: Run tests and verify RED**

Run:

    /root/miniconda3/envs/bdetr/bin/python -m pytest tests/test_rec_geometry_reranker.py -q

Expected: module import failure.

- [x] **Step 5: Implement the pure module**

Define:

    REC_GEOMETRY_MODEL_SCHEMA_VERSION = "rec-geometry-flat-v1"
    FLAT_PARENT_PRIOR_VERSION = \
        "score-desc-query-index-asc-regressed-first-v2"

    def stable_query_descending_order(query_scores):
        # Backend-independent lexicographic order: (-score, query_index).
        ...

    def build_deployed_parent_state(compact_scores, query_indices,
                                    candidate_valid, num_queries):
        query_scores = scatter_candidate_scores(
            compact_scores, query_indices, candidate_valid, num_queries
        )
        query_order = stable_query_descending_order(query_scores)
        top1_query_index = query_order[:, 0]
        parent_top1_mask = (
            query_indices == top1_query_index.unsqueeze(1)
        ) & candidate_valid
        return {
            "compact_scores": compact_scores,
            "query_scores": query_scores,
            "query_indices": query_indices,
            "candidate_valid": candidate_valid,
            "query_order": query_order,
            "top1_query_index": top1_query_index,
            "parent_top1_mask": parent_top1_mask,
        }

    def build_rec_geometry_model_inputs(
            base_features, geometry_features, parent_scores,
            parent_top1_mask, geometry_valid,
            base_feature_names, geometry_feature_names):
        variants = geometry_features.shape[2]
        base = base_features.unsqueeze(2).expand(
            -1, -1, variants, -1
        )
        score = parent_scores[:, :, None, None].expand(
            -1, -1, variants, 1
        )
        top1 = parent_top1_mask[:, :, None, None].expand(
            -1, -1, variants, 1
        ).to(base.dtype)
        features = torch.cat([base, geometry_features, score, top1], dim=-1)
        valid = geometry_valid.bool()
        features = features.reshape(features.shape[0], -1, features.shape[-1])
        valid = valid.reshape(valid.shape[0], -1)
        features = torch.where(
            valid.unsqueeze(-1), features, torch.zeros_like(features)
        )
        return {
            "features": features,
            "valid_mask": valid,
            "feature_names": tuple(base_feature_names)
                + tuple(geometry_feature_names)
                + ("parent_score", "parent_is_deployed_top1"),
        }

    def build_flat_parent_prior(parent_state, geometry_valid,
                                regressed_variant_index):
        query_scores = parent_state["query_scores"]
        query_order = parent_state["query_order"]
        query_ranks = torch.empty_like(query_order)
        rank_values = torch.arange(
            query_scores.shape[1], device=query_scores.device
        ).unsqueeze(0).expand_as(query_order)
        query_ranks.scatter_(1, query_order, rank_values)
        compact_ranks = torch.gather(
            query_ranks, 1, parent_state["query_indices"]
        )
        variants = geometry_valid.shape[2]
        variant_priority = list(range(variants))
        variant_priority.remove(regressed_variant_index)
        variant_priority.insert(0, regressed_variant_index)
        priority_by_index = torch.empty(
            variants, dtype=compact_ranks.dtype, device=compact_ranks.device
        )
        priority_by_index[torch.tensor(
            variant_priority, device=compact_ranks.device
        )] = torch.arange(variants, device=compact_ranks.device)
        order_code = (
            compact_ranks.unsqueeze(-1) * variants + priority_by_index
        )
        return -order_code.reshape(order_code.shape[0], -1).float()

    def blend_rec_geometry_scores(parent_state, learned_logits,
                                  geometry_valid, geometry_weight,
                                  regressed_variant_index):
        weight = float(geometry_weight)
        if weight == 0.0:
            return {
                "use_parent_query_axis": True,
                "query_scores": parent_state["query_scores"],
            }
        valid = geometry_valid.reshape(geometry_valid.shape[0], -1).bool()
        prior = build_flat_parent_prior(
            parent_state, geometry_valid, regressed_variant_index
        )
        flat_scores = (
            (1.0 - weight) * masked_rank_normalize(prior, valid)
            + weight * masked_rank_normalize(learned_logits, valid)
        ).masked_fill(~valid, -float("inf"))
        return {
            "use_parent_query_axis": False,
            "flat_scores": flat_scores,
            "flat_valid_mask": valid,
        }

Use the existing scatter_candidate_scores for the full query state. For nonzero weights, rank-normalize the unique flat parent prior and learned logits, blend them, and retain -inf for invalid entries. Weight zero must return before building any flat ordering.

The deployed parent query order must be reconstructed on the full query axis; do not replace it with compact argmax. Both offline training and runtime must use explicit lexicographic `(-score, query_index)` ordering because PyTorch 1.10 CPU and CUDA `argsort` return different orders for equal finite scores and `-inf` padding. Validation compares the canonical finite subsequence while allowing arbitrary internal order among `-inf` padding. For final flat geometry ordering, add `stable_flat_descending_indices(scores, valid)`, implemented as explicit lexicographic `(-score, flat_index)` ordering per row.

- [x] **Step 6: Run pure scorer tests**

Run:

    /root/miniconda3/envs/bdetr/bin/python -m pytest tests/test_rec_geometry_reranker.py tests/test_rec_reranker.py -q

Expected: all tests pass.

### Task 3: Geometry Sidecar Integrity And Join Layer

**Files:**
- Create: scripts/rec_geometry_cache.py
- Create: tests/test_rec_geometry_cache.py
- Modify: scripts/train_rec_reranker.py
- Modify: tests/test_train_rec_reranker.py

- [ ] **Step 1: Generalize the existing base-cache loader**

Write failing tests for:

    rows, manifest = load_candidate_cache(path, expected_split="val")
    assert manifest["split"] == "val"

Keep load_training_cache(path) as a wrapper that calls expected_split="train". Require completed, full-size caches and retain all existing tensor/identity validation.

- [ ] **Step 2: Write failing canonical hash and base-binding tests**

Tests must prove:

- canonical JSON uses UTF-8, sort_keys=True, compact separators, and allow_nan=False;
- a one-byte base shard mutation changes its SHA and cache-content digest;
- reordered base shard entries are rejected;
- base manifest SHA alone is not accepted without ordered shard SHA-256 values;
- train and val bindings cannot be interchanged.

The public API is:

    def canonical_json_sha256(payload):
        encoded = json.dumps(
            payload, sort_keys=True, separators=(",", ":"),
            ensure_ascii=True, allow_nan=False
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def sha256_file(path):
        digest = hashlib.sha256()
        with Path(path).open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def build_base_cache_binding(cache_dir, expected_split):
        cache_dir = Path(cache_dir).resolve()
        manifest = load_candidate_manifest(cache_dir, expected_split)
        shard_entries = [
            {"name": name, "sha256": sha256_file(cache_dir / name)}
            for name in manifest["shards"]
        ]
        return {
            "path": str(cache_dir),
            "manifest_sha256": canonical_json_sha256(manifest),
            "shards": shard_entries,
            "content_sha256": canonical_json_sha256({
                "manifest": manifest,
                "shards": shard_entries,
            }),
        }

- [ ] **Step 3: Write failing geometry row and manifest tests**

The exact row schema is:

    dataset_index: int
    scan_id: str
    target_id: int
    default_top1_query_index: int
    query_indices: int64[K]
    candidate_valid: bool[K]
    geometry_boxes: float32[K,G,6]
    geometry_valid: bool[K,G]
    evaluator_valid: bool[K,G]
    geometry_features: float32[K,G,25]
    geometry_ious: float32[K,G]
    source_rejection_codes: int16[K,G]

Require finite positive-sized valid boxes, finite features/IoUs, IoUs in [0,1], invalid IoUs equal zero, one unique regressed variant, regressed validity equal candidate_valid, and at least one evaluator-valid candidate per row.

The immutable manifest fields must include:

    geometry_cache_schema_version
    geometry_schema_version
    geometry_feature_names
    ordered variant_names and variant_configs
    min_points and max_point_fraction
    split, dataset_size, source_dataset_size
    candidate_rule and target_iou_policy
    checkpoint path, SHA-256, and epoch
    model_inputs and backbone_config
    extraction_batch_size=12, num_workers=2, shard_size=252
    base_cache_binding
    ScanRefer annotation SHA-256
    panel/audit provenance reference
    filter_non_gt_boxes policy

Mutable fields are complete, sample_count, shard entries (name, row_count, sha256), parity maxima, and cache_content_digest.

- [ ] **Step 4: Write failing atomic/resume tests**

Cover:

- shard_size % extraction_batch_size == 0;
- incomplete sample_count is a multiple of shard_size;
- only a completed cache may end with a short final shard;
- manifested row counts sum to sample_count;
- an unexpected shard is rejected by the strict loader;
- the writer may replace only the exact next unmanifested orphan;
- shard rename happens before manifest publication;
- a failed manifest write removes the just-published shard;
- final complete cache has sample_count == dataset_size == source_dataset_size;
- content digests bind every row-bearing shard.

- [ ] **Step 5: Run cache tests and verify RED**

Run:

    /root/miniconda3/envs/bdetr/bin/python -m pytest tests/test_rec_geometry_cache.py tests/test_train_rec_reranker.py -q

Expected: missing API/module failures.

- [ ] **Step 6: Implement the cache module and loader refactor**

Expose:

    GEOMETRY_CACHE_SCHEMA_VERSION = 1

    def initialize_geometry_cache(output_dir, immutable_metadata,
                                  overwrite=False):
        # Validate immutable metadata, create or verify the building
        # directory, remove only the exact next uncommitted shard, and
        # atomically publish a building manifest with zero committed rows.
        return write_or_resume_building_manifest(
            output_dir, immutable_metadata, overwrite
        )

    def append_geometry_shard(output_dir, manifest, rows):
        validated_rows = validate_geometry_rows(rows, manifest)
        shard_path = write_fsynced_temporary_shard(
            output_dir, manifest, validated_rows
        )
        descriptor = describe_and_publish_shard(shard_path, validated_rows)
        return atomically_append_manifest_descriptor(
            output_dir, manifest, descriptor
        )

    def finalize_geometry_cache(output_dir, manifest, parity_maxima):
        validate_complete_geometry_state(manifest, parity_maxima)
        finalized = build_final_geometry_manifest(manifest, parity_maxima)
        atomically_publish_complete_bundle(output_dir, finalized)
        return finalized

    def load_geometry_cache(cache_dir, expected_split):
        manifest = load_and_validate_geometry_manifest(
            cache_dir, expected_split, require_complete=True
        )
        rows = load_and_validate_geometry_shards(cache_dir, manifest)
        return rows, manifest

    def join_base_and_geometry_rows(base_rows, geometry_rows,
                                    base_manifest, geometry_manifest):
        validate_base_binding(base_manifest, geometry_manifest)
        if len(base_rows) != len(geometry_rows):
            raise ValueError("base and geometry cache row counts differ")
        return [
            validate_and_join_geometry_row(base, geometry, geometry_manifest)
            for base, geometry in zip(base_rows, geometry_rows)
        ]

The join must compare dataset_index, scan_id, target_id, default query, query indices, candidate validity, candidate rule, checkpoint, model inputs, backbone config, target policy, base manifest SHA, and all ordered base shard hashes before returning rows.

- [ ] **Step 7: Run cache and parent regression tests**

Run:

    /root/miniconda3/envs/bdetr/bin/python -m pytest tests/test_rec_geometry_cache.py tests/test_train_rec_reranker.py tests/test_rec_candidate_cache.py -q

Expected: all tests pass.

### Task 4: Deterministic Full Geometry Extractor

**Files:**
- Create: scripts/cache_scanrefer_rec_mask_geometry.py
- Create: tests/test_cache_scanrefer_rec_mask_geometry.py

- [ ] **Step 1: Write failing row-construction tests**

Inject a synthetic fresh candidate batch, cached base rows, and geometry batch. Require the extractor helper to:

- run assert_candidate_cache_parity before accepting geometry;
- canonicalize the unique regressed boxes and IoUs from the base rows;
- preserve fresh mask-derived geometry for non-regressed variants;
- project rejection codes to [B,K,G] int16;
- set evaluator_valid according to the manifest filter policy;
- detach all row tensors to contiguous CPU storage;
- exclude base features, parent scores, GT boxes, threshold labels, and model_inputs from sidecar rows.

- [ ] **Step 2: Write a failing GT-boundary spy**

Use injected callables and assert the sequence:

    end_points = model(inputs)
    candidate_batch = build_rec_candidate_batch(end_points, inputs)
    geometry = build_rec_mask_geometry_candidates(
        end_points, inputs, candidate_batch
    )
    # Only after both deployable builders return:
    targeted_candidates = attach_candidate_targets(
        candidate_batch, end_points_with_gt, root_only=True
    )
    targeted_geometry = attach_rec_mask_geometry_targets(
        geometry, end_points_with_gt, root_only=True
    )

The spy must fail if center_label, size_gts, box_label_mask, gt_masks, or candidate/geometry IoUs are visible to either deployable builder.

- [ ] **Step 3: Write failing CLI and replay tests**

Require:

    --split {train,val}
    --data-root PATH
    --checkpoint PATH
    --base-cache PATH
    --output-dir PATH
    --batch-size 12
    --num-workers 2
    --shard-size 252
    --device cuda:0
    --overwrite

Reject any production batch size other than the base binding's declared replay size. Resume only at a batch-aligned shard boundary. Dataset indices must be sequential from zero and the final DataLoader batch may be short only at dataset end.

- [ ] **Step 4: Run extractor tests and verify RED**

Run:

    /root/miniconda3/envs/bdetr/bin/python -m pytest tests/test_cache_scanrefer_rec_mask_geometry.py -q

Expected: module import failure.

- [ ] **Step 5: Implement extraction**

Reuse TrainTester.get_datasets, TrainTester.get_model, TrainTester._get_inputs, load_checkpoint stripping, build_rec_candidate_batch, attach_candidate_targets, build_rec_mask_geometry_candidates, and audit parity tolerances. Load base rows with load_candidate_cache for the selected split and index them by dataset_index.

Record maximum absolute drift for boxes, candidate IoUs, features, default scores, and contrastive scores. Enforce exact query/valid/default-query identity, box atol/rtol 0.002, and IoU atol/rtol 0.01. Feature and score drift is diagnostic but all values must be finite.

- [ ] **Step 6: Run a one-shard GPU smoke**

Run train and val smoke extractions with the production batch size 12 and shard size 252 into new smoke building directories. Stop after one committed 252-row shard, verify both caches remain incomplete and therefore unavailable to the strict final loader, all tensors meet schema, regressed slices equal base cache bit-for-bit, and resuming continues from dataset index 252. Do not publish a partial smoke as a production-complete cache.

- [ ] **Step 7: Run extractor, geometry, and cache tests**

Run:

    /root/miniconda3/envs/bdetr/bin/python -m pytest tests/test_cache_scanrefer_rec_mask_geometry.py tests/test_rec_geometry_cache.py tests/test_rec_mask_geometry.py tests/test_audit_scanrefer_mask_geometry.py -q

Expected: all tests pass.

### Task 5: Scene-Disjoint Geometry Scorer Training

**Files:**
- Create: scripts/train_rec_geometry_reranker.py
- Create: tests/test_train_rec_geometry_reranker.py

- [ ] **Step 1: Write failing joined-dataset tests**

Create two scenes of synthetic base/geometry rows. Assert:

- deterministic_scene_split keeps every scene wholly in fit or calibration;
- feature statistics consume only fit-scene evaluator-valid flattened candidates;
- before the scene split, the frozen parent model is run once over all joined
  train rows in contiguous dataset-index order using CUDA float32, eval mode,
  no autocast, world size 1, local batch size 12, a natural remainder, and the
  deployed A100 matmul TF32 setting enabled;
- optimizer batch size never changes the materialized parent scores;
- the parent score digest and the complete inference contract are retained in
  the geometry artifact together with the base-cache and parent-artifact SHA;
- parent_top1 is derived after scattering parent compact scores to Q=256;
- geometry IoUs are targets only and never enter model features;
- each batch has features [B,112,179], boxes [B,112,6], valid [B,112], and IoUs [B,112].

- [ ] **Step 2: Write failing calibration and exact-fallback tests**

The evaluator must report parent baseline, each nonzero blend, fix/break counts, and geometry oracle. Weight zero must use the parent scattered query-axis selection, not compact argmax or a flat prior. The chooser must reject a weight if either strict threshold regresses versus the parent baseline, then maximize:

    min(acc025 / 0.60, acc050 / 0.47) + 0.1 * (acc025 + acc050)

Use the existing conservative weight grid:

    (0.0, 0.05, 0.1, 0.2, 0.35, 0.5, 0.65, 0.8, 0.9, 1.0)

- [ ] **Step 3: Write failing artifact tests**

The artifact must reject any mismatch in:

- artifact/model/geometry/base schema versions;
- exact 179 ordered feature names and normalization tensors;
- ordered variants/config, unique regressed index, min/max filters;
- checkpoint SHA/epoch, model inputs, backbone config, and candidate rule;
- actual parent artifact SHA and its structural provenance;
- exact parent inference contract and materialized train parent-score digest;
- flat prior/tie policy and selected weight;
- train base-cache content digest and train geometry-cache digest;
- target IoU and evaluator filter policies;
- fit/calibration scene digest and training arguments.

Do not store val cache hashes, val metrics, or validation-derived choices in the training artifact.

- [ ] **Step 4: Write a failing synthetic learnability test**

Generate geometry features where one dimension identifies the best geometry variant independently of query order. Require one short training run to improve both calibration thresholds over the frozen parent and reproduce identical outputs after artifact reload.

- [ ] **Step 5: Run tests and verify RED**

Run:

    /root/miniconda3/envs/bdetr/bin/python -m pytest tests/test_train_rec_geometry_reranker.py -q

Expected: module import failure.

- [ ] **Step 6: Implement the trainer**

Reuse QueryReranker and compute_rec_reranker_loss over flattened candidates. Use AdamW, gradient clipping 1.0, fit-only normalization, shuffled deterministic fit batches, early stopping on calibration score, and CPU-cloned best state. The public entry points are:

    def load_geometry_training_data(base_cache, geometry_cache,
                                    parent_artifact_path):
        base_rows, base_manifest = load_candidate_cache(
            base_cache, expected_split="train"
        )
        geometry_rows, geometry_manifest = load_geometry_cache(
            geometry_cache, expected_split="train"
        )
        joined = join_base_and_geometry_rows(
            base_rows, geometry_rows, base_manifest, geometry_manifest
        )
        parent_model, parent_artifact = load_reranker_artifact(
            parent_artifact_path, device="cpu"
        )
        return joined, base_manifest, geometry_manifest, (
            parent_model, parent_artifact
        )

    def train_geometry_reranker(
            base_cache, geometry_cache, parent_artifact_path, output,
            split_seed=0, model_seed=0, hidden_dim=256, dropout=0.1,
            lr=1e-3, weight_decay=1e-4, batch_size=256,
            max_epochs=100, patience=10, device="cuda:0"):
        joined, base_manifest, geometry_manifest, parent = (
            load_geometry_training_data(
                base_cache, geometry_cache, parent_artifact_path
            )
        )
        parent_scores = materialize_parent_scores(
            joined, parent, device=device, local_batch_size=12
        )
        fit_rows, calibration_rows = deterministic_scene_split(
            joined, split_seed
        )
        feature_mean, feature_std = compute_geometry_feature_stats(
            fit_rows, parent
        )

        model = QueryReranker(
            input_dim=179, hidden_dim=hidden_dim, dropout=dropout
        ).to(device)
        # Train with compute_rec_reranker_loss, keep the earliest
        # calibration-optimal state, choose a non-regressing blend, and
        # atomically save the fully validated artifact.
        return fit_and_save_geometry_model(
            model, fit_rows, calibration_rows, feature_mean, feature_std,
            parent, base_manifest, geometry_manifest, output,
            model_seed, lr, weight_decay, batch_size, max_epochs, patience,
            device
        )

    def load_geometry_reranker_artifact(path, device="cpu"):
        artifact = torch.load(Path(path).resolve(), map_location="cpu")
        model_config = validate_geometry_artifact(artifact)
        model = QueryReranker(**model_config)
        model.load_state_dict(artifact["model_state_dict"], strict=True)
        model.to(device).eval()
        return model, artifact

The production materialization must happen before `deterministic_scene_split`
and must populate a read-only row-identity cache used by statistics, optimizer
batches, calibration, artifact reload checks, and every robustness sweep. Its
content digest is over ordered row identity plus raw float32 compact-score
bytes. `batch_size=256` remains the geometry optimizer batch and must not be
reused as the parent inference batch. CPU parent inference or CUDA batches
other than 12 are not equivalent on the target A100 because TF32 perturbations
can flip near-equal parent logits before rank blending. For the authoritative
train cache, artifact construction must require local `cuda:0` with matmul
TF32 enabled. A sealed cache may be re-entered only with the exact same row
objects and signature; artifact construction must reconstruct the ordered row
set and score digest from the live cache so missing or mutated scores fail
closed.

- [ ] **Step 7: Run training tests and all pure REC tests**

Run:

    /root/miniconda3/envs/bdetr/bin/python -m pytest tests/test_train_rec_geometry_reranker.py tests/test_rec_geometry_reranker.py tests/test_rec_reranker.py tests/test_train_rec_reranker.py -q

Expected: all tests pass.

### Task 6: Parent Runtime Refactor And Geometry Attachment

**Files:**
- Modify: main_utils.py
- Modify: train_dist_mod.py
- Modify: tests/test_rec_reranker_runtime.py
- Create: tests/test_rec_geometry_runtime.py

- [ ] **Step 1: Write a failing parent-output compatibility test**

Refactor build_rec_reranker_scores behind:

    outputs = build_rec_reranker_outputs(
        end_points, inputs, reranker, artifact
    )
    assert set(outputs) == {
        "candidate_batch", "compact_scores", "query_scores"
    }

Keep build_rec_reranker_scores as a compatibility wrapper returning query_scores. Require bit-identical wrapper output for ordinary and tied score fixtures.

- [ ] **Step 2: Write failing runtime artifact and GT-isolation tests**

Add CLI flags:

    --rec_geometry_reranker_checkpoint PATH
    --eval_use_rec_geometry_reranker_scores

Require the geometry flag to load the frozen parent and geometry artifacts once, validate their actual file hashes/configuration, and call geometry construction before batch_data GT is merged. A spy must prove no GT key is present.

- [ ] **Step 3: Write failing zero/nonzero runtime tests**

At selected weight zero:

- attach rec_reranker_scores [B,Q];
- attach rec_geometry_runtime_mode="parent_query_axis";
- do not call build_rec_mask_geometry_candidates;
- do not attach geometry boxes/scores/validity.

At nonzero weight:

- build geometry exactly once from the already-built candidate_batch;
- build normalized [B,112,179] features;
- run the frozen geometry scorer;
- attach rec_geometry_boxes [B,112,6], rec_geometry_scores [B,112], and rec_geometry_valid_mask [B,112];
- attach rec_geometry_runtime_mode="flat_geometry_axis";
- attach rec_geometry_fallback_index [B], pointing to the deployed parent Top-1 query's canonical regressed flat candidate;
- retain rec_reranker_scores for diagnostics/fallback;
- reject nonfinite scores, malformed boxes, or a row with no valid candidate.

- [ ] **Step 4: Run runtime tests and verify RED**

Run:

    /root/miniconda3/envs/bdetr/bin/python -m pytest tests/test_rec_reranker_runtime.py tests/test_rec_geometry_runtime.py -q

Expected: missing output/refactor/CLI behavior failures.

- [ ] **Step 5: Implement the runtime refactor**

Preserve current parent artifact validation and add geometry validation before the first scored batch. Pass eval_use_rec_geometry_reranker_scores into GroundingEvaluator. Geometry takes precedence only when runtime_mode is flat_geometry_axis and its complete candidate tensor set plus fallback index is attached. Parent mode must not carry partial geometry tensors. Do not attach cached IoUs, threshold labels, rejection codes, or any training-only field at runtime.

- [ ] **Step 6: Run runtime and leakage tests**

Run:

    /root/miniconda3/envs/bdetr/bin/python -m pytest tests/test_rec_reranker_runtime.py tests/test_rec_geometry_runtime.py tests/test_rec_candidate_adapter.py tests/test_rec_mask_geometry.py -q

Expected: all tests pass.

### Task 7: Evaluator Active Candidate Axis

**Files:**
- Modify: src/grounding_evaluator.py
- Create: tests/test_grounding_evaluator_rec_geometry.py
- Modify: tests/test_grounding_evaluator_rec_reranker.py

- [ ] **Step 1: Write failing geometry precedence and shape tests**

Build a fixture where Q-axis parent scores select a wrong regressed box but C-axis geometry scores select a correct mask box. Assert only last_ position alignment uses geometry; semantic alignment and mask evaluation remain unchanged.

Reject mismatched [B,C,6]/[B,C]/[B,C] shapes, NaN/+inf valid scores, nonfinite or nonpositive valid boxes, an out-of-range/invalid fallback index, partial geometry key sets, and rows without a valid candidate.

- [ ] **Step 2: Write failing filter and invalid Top-k tests**

With filter_non_gt_boxes=True, compute detector overlap against active geometry boxes rather than Q-axis boxes and combine it with rec_geometry_valid_mask. Construct fewer than ten valid candidates and prove invalid zero boxes never enter Top-1/5/10. If filtering removes every geometry candidate, fail closed instead of sorting an all--inf row; never silently select an arbitrary zero box.

Geometry Top-k ordering must call `stable_flat_descending_indices` and then take `min(k, valid_count)`. The parent Q-axis path must preserve the existing score tensor but use `stable_query_descending_order` for selection so CPU calibration and CUDA runtime choose the same query when finite maxima tie.

- [ ] **Step 3: Write a failing exact parent fallback test**

Use tied parent maxima and candidate order different from query order. With geometry weight zero (therefore no geometry tensors attached), assert the compatibility wrapper score tensor remains bit-for-bit identical and evaluator selection follows canonical `(-score, query_index)` order on both CPU and CUDA.

- [ ] **Step 4: Run evaluator tests and verify RED**

Run:

    /root/miniconda3/envs/bdetr/bin/python -m pytest tests/test_grounding_evaluator_rec_geometry.py tests/test_grounding_evaluator_rec_reranker.py -q

Expected: geometry axis is unsupported.

- [ ] **Step 5: Implement one active-axis resolver**

Add a private helper returning aligned boxes, scores, and validity:

    def _resolve_position_candidates(self, end_points, prefix,
                                     pred_bbox, default_scores):
        mode = end_points.get("rec_geometry_runtime_mode")
        geometry_keys = (
            "rec_geometry_boxes",
            "rec_geometry_scores",
            "rec_geometry_valid_mask",
            "rec_geometry_fallback_index",
        )
        if prefix == "last_" and mode == "flat_geometry_axis" and all(
                key in end_points for key in geometry_keys):
            boxes = end_points["rec_geometry_boxes"]
            scores = end_points["rec_geometry_scores"]
            valid = end_points["rec_geometry_valid_mask"].bool()
            fallback = end_points["rec_geometry_fallback_index"]
            validate_active_geometry_axis(boxes, scores, valid, fallback)
            return boxes, scores, valid
        if prefix == "last_" and mode == "parent_query_axis":
            reject_partial_geometry_keys(end_points, geometry_keys)
        if prefix == "last_" and self.eval_use_rec_reranker_scores:
            scores = end_points["rec_reranker_scores"]
            valid = torch.isfinite(scores)
            return pred_bbox, scores, valid
        return pred_bbox, default_scores, torch.ones_like(
            default_scores, dtype=torch.bool
        )

Precedence for last_ is geometry tensors, then rec_reranker_scores, then source-choice/default behavior. Keep the old Q-axis operations unchanged when geometry is absent. Apply shape validation, detector filtering, score masking, ordering, and box gathering to the returned active axis.

- [ ] **Step 6: Run evaluator regression tests**

Run:

    /root/miniconda3/envs/bdetr/bin/python -m pytest tests/test_grounding_evaluator_rec_geometry.py tests/test_grounding_evaluator_rec_reranker.py tests/test_grounding_evaluator_source_choice.py -q

Expected: all tests pass.

### Task 8: Full CPU Verification Before Expensive Extraction

**Files:** all files touched above.

- [ ] **Step 1: Compile every changed Python module**

Run:

    /root/miniconda3/envs/bdetr/bin/python -m py_compile models/rec_mask_geometry.py models/rec_geometry_reranker.py scripts/rec_geometry_cache.py scripts/cache_scanrefer_rec_mask_geometry.py scripts/train_rec_geometry_reranker.py main_utils.py train_dist_mod.py src/grounding_evaluator.py

Expected: exit 0.

- [ ] **Step 2: Run all REC and evaluator tests**

Run:

    /root/miniconda3/envs/bdetr/bin/python -m pytest tests/test_rec_reranker.py tests/test_rec_candidate_adapter.py tests/test_rec_candidate_cache.py tests/test_train_rec_reranker.py tests/test_rec_reranker_runtime.py tests/test_grounding_evaluator_rec_reranker.py tests/test_rec_mask_geometry.py tests/test_audit_scanrefer_mask_geometry.py tests/test_rec_geometry_reranker.py tests/test_rec_geometry_cache.py tests/test_cache_scanrefer_rec_mask_geometry.py tests/test_train_rec_geometry_reranker.py tests/test_rec_geometry_runtime.py tests/test_grounding_evaluator_rec_geometry.py -q

Expected: zero failures.

- [ ] **Step 3: Run the complete repository test suite**

Run:

    /root/miniconda3/envs/bdetr/bin/python -m pytest -q

Expected: zero failures. Record any environment-only collection failure separately and do not launch full extraction until all changed-code tests pass.

### Task 9: Produce Full Train And Validation Geometry Sidecars

**Output:**
- /root/autodl-tmp/DATA_ROOT/output/rec_reranker/e71_top16/geometry_train
- /root/autodl-tmp/DATA_ROOT/output/rec_reranker/e71_top16/geometry_val

- [ ] **Step 1: Extract the complete train sidecar**

Run with CUDA device 0, batch size 12, workers 2, shard size 252, and no data augmentation. Do not run train and val extraction concurrently.

Expected: complete=true, sample_count=dataset_size=source_dataset_size=36,665; every shard and base binding hash verifies; parity satisfies exact identity plus declared box/IoU tolerances.

- [ ] **Step 2: Verify train sidecar independently**

Load base and sidecar through their strict public loaders, join all rows, assert exactly 36,665 unique contiguous indices, recompute the cache content digest, and print geometry oracle, variant validity/rejection rates, and maximum parity drift.

- [ ] **Step 3: Extract and verify the complete val sidecar**

Use the identical checkpoint, variant order, batch size, workers, and shard size with split=val.

Expected: complete=true and exactly 9,508 joined rows. Validation labels may be used only for one immutable evaluation record after train-only model selection; they must never enter scorer input or artifact selection.

### Task 10: Train And Select The Geometry Scorer On Train Scenes Only

**Output directory:**
- /root/autodl-tmp/DATA_ROOT/output/rec_reranker/e71_top16/geometry_artifacts

- [ ] **Step 1: Train the primary configuration**

Use seed 0, hidden_dim 256, dropout 0.1, lr 1e-3, weight_decay 1e-4, batch size 256, max 100 epochs, patience 10, and the declared weight grid. Require fit/calibration scenes to reproduce the parent seed-0 90/10 split (506/56 scenes).

- [ ] **Step 2: Run a bounded train-only robustness sweep**

Train only these additional configurations:

- hidden_dim=128, dropout=0.1, lr=1e-3, seed=0
- hidden_dim=256, dropout=0.0, lr=3e-4, seed=0
- hidden_dim=256, dropout=0.1, lr=3e-4, seed=0
- hidden_dim=256, dropout=0.1, lr=1e-3, seed=1

Select one immutable artifact by calibration score subject to no regression at either threshold versus the frozen parent. Do not inspect validation metrics during this choice.

- [ ] **Step 3: Freeze the selected artifact and selection record**

Write a JSON selection record containing every candidate artifact SHA, train cache digests, fit/calibration metrics, fix/break/oracle metrics, selection rule, and winning SHA. The selected .pth must remain unchanged after validation begins.

### Task 11: Full Official Validation And Acceptance Gate

**Output directory:**
- /root/autodl-tmp/DATA_ROOT/output/rec_reranker/e71_top16/geometry_official_val

- [ ] **Step 1: Run one full official validation**

Launch one-GPU distributed evaluation with batch size 12, the exact epoch-71 backbone, the parent artifact, the frozen geometry artifact, --eval_use_rec_reranker_scores, --eval_use_rec_geometry_reranker_scores, use_color, butd, six decoder layers, soft-token loss, contrastive alignment, self-attention, and the checkpoint's source-choice configuration. Explicitly require butd=true, butd_cls=false, and butd_gt=false: both butd_cls and butd_gt substitute GT scene boxes in Joint3DDataset and therefore cannot prove the no-GT objective. Preserve config.json, log.txt, stdout, selected artifact SHA, and code/schema hashes.

Run:

    CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 PYTHONPATH="$PWD:$PWD/pointnet2:$PYTHONPATH" /root/miniconda3/envs/bdetr/bin/python -m torch.distributed.launch --nproc_per_node 1 --master_port 29671 train_dist_mod.py --num_decoder_layers 6 --num_target 256 --model MCLN --use_color --butd --self_attend --detect_intermediate --joint_det --use_soft_token_loss --use_contrastive_align --use_source_choice_selector --source_choice_selector_sources default,default_rank_blend_contrastive010 --source_choice_selector_hidden_dim 288 --skip_missing_superpoints --dataset scanrefer --test_dataset scanrefer --data_root /root/autodl-tmp/DATA_ROOT/ --batch_size 12 --num_workers 2 --print_freq 100 --checkpoint_path /root/autodl-tmp/DATA_ROOT/output/preserved_best/mcln_pair_sweep/mcln_pair_default_rankblend010_2ep_best_acc025_epoch71_0.57993.pth --rec_reranker_checkpoint /root/autodl-tmp/DATA_ROOT/output/rec_reranker/e71_top16/artifacts/reranker_h256_d010_lr1e3_seed0_final_contract.pth --rec_geometry_reranker_checkpoint /root/autodl-tmp/DATA_ROOT/output/rec_reranker/e71_top16/geometry_artifacts/selected_geometry_reranker.pth --eval_use_rec_reranker_scores --eval_use_rec_geometry_reranker_scores --log_dir /root/autodl-tmp/DATA_ROOT/output/rec_reranker/e71_top16/geometry_official_val --exp epoch71_geometry_official --eval

- [ ] **Step 2: Cross-check official and sidecar evaluation**

Evaluate the already-frozen artifact once against geometry_val and write a separate immutable validation record. Official and sidecar Top-1 hit counts must agree within the declared batch/cache parity contract; investigate any difference before accepting the official result.

- [ ] **Step 3: Apply the only completion gate**

Extract the exact official lines:

    last_ position alignment Acc0.25: Top-1: X
    last_ position alignment Acc0.50: Top-1: Y

The project goal is complete only when X >= 0.60000 and Y >= 0.47000 and the runtime configuration proves inference_uses_ground_truth=false. Unit tests, train/calibration accuracy, sidecar metrics, geometry oracle, partial validation, or mask IoU do not satisfy this gate.

- [ ] **Step 4: Preserve the fallback decision**

If either official threshold misses, keep the goal active. Use the frozen validation result only as an evaluation record, not to tune geometry hyperparameters. Proceed to the already-approved REC-specific one-epoch fine-tuning fallback in docs/superpowers/specs/2026-07-14-scanrefer-rec-reranker-design.md with a separate implementation plan.
