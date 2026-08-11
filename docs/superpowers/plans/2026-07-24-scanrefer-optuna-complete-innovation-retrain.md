# ScanRefer Optuna Complete-Innovation Retrain Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement, verify, and launch a resumable 20-trial train-only Optuna search for `MCLN + source-choice selector`, then automatically start the selected 46-epoch full-train continuation.

**Architecture:** Add strict optimizer grouping and loss-scale wiring to the shared trainer, export exact five-metric evaluator receipts, and build train-only fit/calibration dataset views without constructing ScanRefer validation. A pure contract module owns trial feasibility and selection, while separate GPU runners execute two-epoch trials and the long continuation; a small orchestrator persists Optuna state, prunes weights, and launches the long run after a feasible selection.

**Tech Stack:** Python 3.7, PyTorch 1.10.2+cu111, Optuna 4.0.0, torch distributed single-GPU training, pytest, SQLite, ScanRefer, existing MCLN evaluators.

---

## Execution Notes

The repository has no Git metadata and must not be initialized as a repository. Replace every commit checkpoint below with a source snapshot and SHA-256 manifest. Work in the current directory because a Git worktree cannot be created.

Use `/root/miniconda3/envs/bdetr/bin/python` for every Python command. Keep `pretained model/ckpt_epoch_54.pth` and all protected epoch-71 artifacts unchanged except for making epoch-54 mode `0444` immediately before the formal launch.

This plan covers implementation through automatic long-run launch. Rebuilding parent/geometry/joint sidecars is a dependent follow-up plan after a long-run Pareto checkpoint exists; the current runner must emit the exact checkpoint and provenance needed for that follow-up.

## File Map

- Create `models/mcln_training_groups.py`: strict, pure MCLN optimizer parameter classification and group construction.
- Modify `main_utils.py`: expose mask LR/loss CLI flags, use strict groups, and forward both loss scales.
- Modify `src/grounding_evaluator.py`: export exact structured Position and mask metrics from counters.
- Modify `train_dist_mod.py`: return the structured metrics from `evaluate_one_epoch` without changing existing log output.
- Create `scripts/tuning/scanrefer_train_only.py`: authoritative scene split and fit/calibration annotation views.
- Create `scripts/tuning/mcln_optuna_contract.py`: search space, metric validation, feasibility, objective, tie-break, disk and checkpoint rules.
- Create `scripts/tuning/train_mcln_optuna_trial.py`: baseline and exactly-two-epoch train-only GPU runner.
- Create `scripts/tuning/optuna_mcln_complete_retrain.py`: resumable Optuna subprocess orchestration and long-run dispatch.
- Create `scripts/tuning/train_mcln_complete_long.py`: full-train epoch-55-to-100 runner with atomic latest and Pareto retention.
- Create `scripts/tuning/mcln_retrain_provenance.py`: hashes, environment receipt, atomic JSON, and source snapshot.
- Create `scripts/tuning/run_optuna_mcln_complete_retrain20.sh`: stable launcher.
- Create `tests/test_mcln_training_groups.py`: optimizer group contract tests.
- Create `tests/test_mcln_retrain_metrics.py`: exact evaluator export tests.
- Create `tests/test_scanrefer_train_only.py`: split and validation-isolation tests.
- Create `tests/test_mcln_optuna_contract.py`: search, feasibility, recovery and retention tests.
- Create `tests/test_train_mcln_optuna_trial.py`: two-epoch runner contract tests.
- Create `tests/test_optuna_mcln_complete_retrain.py`: study resume, cleanup and dispatch tests.
- Create `tests/test_train_mcln_complete_long.py`: cosine schedule, validation epochs and Pareto tests.
- Create `tests/test_mcln_retrain_provenance.py`: snapshot and input protection tests.
- Modify `docs/REC_3DRES_OPTIMIZATION_LOG.md`: implementation and launch handoff.

### Task 1: Strict Optimizer Groups And Loss Wiring

**Files:**
- Create: `models/mcln_training_groups.py`
- Create: `tests/test_mcln_training_groups.py`
- Modify: `main_utils.py:51-83`
- Modify: `main_utils.py:377-488`
- Modify: `main_utils.py:608-628`

- [ ] **Step 1: Write failing optimizer-group tests**

Create a toy module with ordinary, mask, backbone, frozen text, and selector parameters. Require stable group order `decoder`, `backbone`, `mask_head`, `selector`, exact LRs, no duplicate identities, and complete coverage:

```python
class ToyMCLN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.decoder = torch.nn.Linear(2, 2)
        self.backbone_net = torch.nn.Linear(2, 2)
        self.x_mask = torch.nn.Linear(2, 2)
        self.rel_encoder = torch.nn.Linear(2, 2)
        self.text_encoder = torch.nn.Linear(2, 2)
        self.source_choice_selector = torch.nn.Linear(2, 2)
        for parameter in self.text_encoder.parameters():
            parameter.requires_grad = False


def test_complete_innovation_groups_are_disjoint_and_use_requested_lrs():
    groups = build_mcln_optimizer_param_groups(
        ToyMCLN(), decoder_lr=2e-5, backbone_lr=2e-4,
        selector_lr=7e-4, mask_head_lr_multiplier=4.0,
    )
    assert [group["name"] for group in groups] == [
        "decoder", "backbone", "mask_head", "selector",
    ]
    assert [group["lr"] for group in groups] == [2e-5, 2e-4, 8e-5, 7e-4]
    identities = [id(p) for group in groups for p in group["params"]]
    assert len(identities) == len(set(identities))
    expected = {id(p) for p in ToyMCLN().parameters() if p.requires_grad}
    assert len(identities) == len(expected)
```

Use one model instance for both `groups` and `expected` in the actual test. Add rejection tests for zero/negative/non-finite LR, missing selector parameters when selector is required, and an intentionally duplicated parameter path.

- [ ] **Step 2: Run the tests and verify the module is missing**

Run: `/root/miniconda3/envs/bdetr/bin/python -m pytest tests/test_mcln_training_groups.py -q`

Expected: collection fails with `ModuleNotFoundError: No module named 'models.mcln_training_groups'`.

- [ ] **Step 3: Implement strict parameter classification**

Create the module with this public contract:

```python
MASK_HEAD_PREFIXES = (
    "x_mask.", "x_query.", "rel_encoder.", "swa_layers.",
    "swa_ffn_layers.", "out_norm.", "out_score.",
)


def bare_parameter_name(name):
    return name[7:] if name.startswith("module.") else name


def parameter_group_name(name):
    name = bare_parameter_name(name)
    if name.startswith("source_choice_selector."):
        return "selector"
    if name.startswith("backbone_net."):
        return "backbone"
    if any(name.startswith(prefix) for prefix in MASK_HEAD_PREFIXES):
        return "mask_head"
    if name.startswith("text_encoder."):
        return "frozen_text"
    return "decoder"


def build_mcln_optimizer_param_groups(
        model, decoder_lr, backbone_lr, selector_lr,
        mask_head_lr_multiplier, require_selector=True):
    numeric = {
        "decoder_lr": decoder_lr,
        "backbone_lr": backbone_lr,
        "selector_lr": selector_lr,
        "mask_head_lr_multiplier": mask_head_lr_multiplier,
    }
    for label, value in numeric.items():
        if (not isinstance(value, (int, float)) or isinstance(value, bool)
                or not math.isfinite(float(value)) or float(value) <= 0.0):
            raise ValueError("{} must be finite and positive".format(label))

    buckets = {name: [] for name in (
        "decoder", "backbone", "mask_head", "selector",
    )}
    names = {name: [] for name in buckets}
    expected = set()
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        identity = id(parameter)
        if identity in expected:
            raise ValueError("trainable parameter identity is duplicated")
        expected.add(identity)
        group_name = parameter_group_name(name)
        if group_name == "frozen_text":
            raise ValueError("text_encoder parameter is unexpectedly trainable")
        buckets[group_name].append(parameter)
        names[group_name].append(bare_parameter_name(name))

    if require_selector and not buckets["selector"]:
        raise ValueError("complete innovation training requires selector parameters")
    lr_by_group = {
        "decoder": float(decoder_lr),
        "backbone": float(backbone_lr),
        "mask_head": float(decoder_lr) * float(mask_head_lr_multiplier),
        "selector": float(selector_lr),
    }
    groups = []
    for group_name in ("decoder", "backbone", "mask_head", "selector"):
        if not buckets[group_name]:
            continue
        groups.append({
            "name": group_name,
            "params": buckets[group_name],
            "parameter_names": tuple(sorted(names[group_name])),
            "lr": lr_by_group[group_name],
        })
    actual = [id(parameter) for group in groups for parameter in group["params"]]
    if len(actual) != len(set(actual)) or set(actual) != expected:
        raise ValueError("optimizer parameter coverage is not exact")
    return groups
```

Store both `name` and sorted `parameter_names` in each optimizer group for receipts. Never classify by substring; use the exact prefixes above.

- [ ] **Step 4: Run focused group tests**

Run: `/root/miniconda3/envs/bdetr/bin/python -m pytest tests/test_mcln_training_groups.py -q`

Expected: all tests pass.

- [ ] **Step 5: Add failing CLI and loss-forwarding tests**

Extend the test file to monkeypatch the criterion and require:

```python
args = SimpleNamespace(
    num_decoder_layers=6,
    query_points_obj_topk=4,
    use_source_choice_selector=True,
    source_choice_selector_loss_weight=0.5,
    source_choice_selector_default_source="default",
    source_choice_selector_choice_target=(
        "precision_gain_default_sourcewise_focal_bce"
    ),
    source_choice_selector_min_iou_gap=0.03,
    mask_loss_scale=2.0,
    consistency_loss_scale=0.25,
)
BaseTrainTester._compute_loss({}, fake_criterion, None, args)
assert received["mask_loss_scale"] == 2.0
assert received["consistency_loss_scale"] == 0.25
```

Also call `parse_option()` under a patched argv and require defaults of `1.0` for all three new numeric behaviors: mask-head LR multiplier, mask loss scale, and consistency scale.

- [ ] **Step 6: Wire CLI, optimizer, and loss scales**

Add:

```python
parser.add_argument('--mask_head_lr_multiplier', type=float, default=1.0)
parser.add_argument('--mask_loss_scale', type=float, default=1.0)
parser.add_argument('--consistency_loss_scale', type=float, default=1.0)
```

In the ordinary full-training branch with `use_source_choice_selector=True`, call `build_mcln_optimizer_param_groups` and pass `args.source_choice_selector_lr` explicitly. Preserve selector-only, frozen, and small-LR legacy branches. Forward both loss scales in `_compute_loss`.

- [ ] **Step 7: Verify focused and legacy tests**

Run:

```bash
/root/miniconda3/envs/bdetr/bin/python -m pytest \
  tests/test_mcln_training_groups.py \
  tests/test_main_utils_source_choice_checkpoint.py \
  tests/test_source_choice_selector.py -q
```

Expected: all tests pass.

### Task 2: Exact Five-Metric Evaluator Receipt

**Files:**
- Create: `tests/test_mcln_retrain_metrics.py`
- Modify: `src/grounding_evaluator.py:162-310`
- Modify: `train_dist_mod.py:2035-2102`

- [ ] **Step 1: Write failing export tests**

Instantiate `GroundingEvaluator`, populate exact counter keys, and assert a strict schema:

```python
metrics = evaluator.export_retrain_metrics(expected_sample_count=4)
assert metrics == {
    "schema": "mcln-retrain-metrics-v1",
    "sample_count": 4,
    "position": {
        "fixed_default": {"hits025": 2, "hits050": 1},
        "learned_selector": {"hits025": 3, "hits050": 2},
    },
    "mask": {
        "hits025": 3,
        "hits050": 2,
        "iou_sum": 2.25,
        "miou": 0.5625,
    },
}
```

Add failures for absent source counters, mismatched denominators, `hits050 > hits025`, non-finite IoU sum, and the evaluator's `1e-14` sentinel being mistaken for a sample.

- [ ] **Step 2: Run the tests and see the missing method fail**

Run: `/root/miniconda3/envs/bdetr/bin/python -m pytest tests/test_mcln_retrain_metrics.py -q`

Expected: failure because `export_retrain_metrics` is absent.

- [ ] **Step 3: Implement the strict exporter**

Add helpers that round integer-like counters only within `1e-9`, require one common denominator, and compute accuracy only in consumers. Use these exact source keys:

```python
fixed_key = ('source_choice', 'fixed_default', threshold, 1)
learned_key = ('source_choice', 'learned_selector', threshold, 1)
```

Use `dets['overall_mask']`, `dets['overall50_mask']`, `dets['mask_sem']`, and rounded `gts['mask_sem']` for the mask receipt.

- [ ] **Step 4: Return metrics from `TrainTester.evaluate_one_epoch`**

After synchronization and existing logging, set on rank zero:

```python
metrics = evaluator.export_retrain_metrics(
    expected_sample_count=getattr(args, "expected_eval_sample_count", None)
)
return metrics
```

Return `None` on non-root ranks. Existing call sites may ignore the return value.

- [ ] **Step 5: Run evaluator regression tests**

Run:

```bash
/root/miniconda3/envs/bdetr/bin/python -m pytest \
  tests/test_mcln_retrain_metrics.py \
  tests/test_grounding_evaluator_source_choice.py \
  tests/test_grounding_evaluator.py -q
```

Expected: all available tests pass; omit `tests/test_grounding_evaluator.py` only if that file does not exist.

### Task 3: Train-Only Scene Views

**Files:**
- Create: `scripts/tuning/scanrefer_train_only.py`
- Create: `tests/test_scanrefer_train_only.py`

- [ ] **Step 1: Write failing pure partition tests**

Build fake ScanRefer and repeated Scannet annotations and require calibration scenes never appear in fit, including Scannet rows:

```python
views = partition_train_annotations(annos, seed=0, calibration_fraction=0.10)
fit_scenes = {row["scan_id"] for row in views["fit_annos"]}
cal_scenes = {row["scan_id"] for row in views["calibration_annos"]}
assert fit_scenes.isdisjoint(cal_scenes)
assert {row["dataset"] for row in views["calibration_annos"]} == {"scanrefer"}
assert all(
    row["scan_id"] not in cal_scenes
    for row in views["fit_annos"] if row["dataset"] == "scannet"
)
```

Add a fake extra Scannet-only scene and require it remains in fit. Reject duplicate/missing dataset fields and a ScanRefer population not covering at least two scenes.

- [ ] **Step 2: Run and verify the missing module failure**

Run: `/root/miniconda3/envs/bdetr/bin/python -m pytest tests/test_scanrefer_train_only.py -q`

Expected: collection fails because the module is absent.

- [ ] **Step 3: Implement partition and dataset-view helpers**

Reuse `models.rec_finetune.build_rec_finetune_scene_split` for ScanRefer rows. Return `fit_annos`, `calibration_annos`, and exact metadata. Implement shallow dataset views so heavy scan/superpoint state is shared safely but annotations and augmentation flags are independent:

```python
def make_dataset_views(base_dataset, split):
    fit_dataset = copy.copy(base_dataset)
    fit_dataset.annos = list(split["fit_annos"])

    calibration_dataset = copy.copy(base_dataset)
    calibration_dataset.annos = list(split["calibration_annos"])
    calibration_dataset.augment = False
    calibration_dataset.augment_det = False
    calibration_dataset.joint_det = False
    calibration_dataset.random_utt = False
    return fit_dataset, calibration_dataset
```

Require authoritative real-data metadata to equal 562/506/56 scenes and 36,665/33,040/3,625 ScanRefer rows before a formal run.

- [ ] **Step 4: Test validation isolation**

Monkeypatch `Joint3DDataset` and require the builder constructs only `split='train'`. Assert neither `val_v3scans.pkl` nor a `_val.json` path is supplied or opened.

- [ ] **Step 5: Run focused tests**

Run: `/root/miniconda3/envs/bdetr/bin/python -m pytest tests/test_scanrefer_train_only.py -q`

Expected: all tests pass.

### Task 4: Pure Optuna Contract

**Files:**
- Create: `scripts/tuning/mcln_optuna_contract.py`
- Create: `tests/test_mcln_optuna_contract.py`

- [ ] **Step 1: Write failing metric and selection tests**

Define baseline/trial fixtures with exact counts. Test all five non-decrease constraints, same-trial selector constraints, objective, and tie-break:

```python
assessment = assess_trial_metrics(baseline, trial)
assert assessment["feasible"] is True
expected = 100.0 * (
    min(assessment["deltas"].values())
    + 0.25 * sum(assessment["deltas"].values()) / 5.0
)
assert assessment["objective"] == pytest.approx(expected, abs=1e-12)
assert select_best_trial([worse, better])["trial_number"] == 4
```

Require exact rejection of 3,624 samples, NaN, mismatched mask IoU sum, learned Position hits below same-trial fixed-default, and any negative baseline delta.

- [ ] **Step 2: Write failing search-space tests**

Use `optuna.trial.FixedTrial` for each preset and assert all seven parameter names and bounds. Require `seed_presets()` returns exactly the three approved mappings.

- [ ] **Step 3: Write failing disk and retention tests**

Use a temporary directory with fake checkpoint files. Require cleanup uses explicit `exists()` checks, keeps only the global best, never calls `Path.unlink(missing_ok=...)`, and refuses a reported free-space value below 8 GiB.

- [ ] **Step 4: Implement the pure contract**

Expose:

```python
METRICS_SCHEMA = "mcln-retrain-metrics-v1"
EXPECTED_CALIBRATION_COUNT = 3625
MIN_FREE_BYTES = 8 * 1024 ** 3

def suggest_trial_params(trial):
    return {
        "decoder_lr": trial.suggest_float("decoder_lr", 5e-6, 4e-5, log=True),
        "mask_head_lr_multiplier": trial.suggest_categorical(
            "mask_head_lr_multiplier", [1.0, 2.0, 4.0]
        ),
        "selector_lr": trial.suggest_float("selector_lr", 2e-4, 2e-3, log=True),
        "mask_loss_scale": trial.suggest_float("mask_loss_scale", 0.5, 4.0, log=True),
        "consistency_loss_scale": trial.suggest_float(
            "consistency_loss_scale", 0.1, 2.0, log=True
        ),
        "selector_loss_weight": trial.suggest_float(
            "selector_loss_weight", 0.1, 1.0, log=True
        ),
        "selector_min_iou_gap": trial.suggest_categorical(
            "selector_min_iou_gap", [0.02, 0.03, 0.05, 0.08]
        ),
    }
```

Implement strict JSON validation, feasibility, objective, deterministic tie-break, completed-trial counting, and Python-3.7-safe checkpoint cleanup.

- [ ] **Step 5: Run contract tests**

Run: `/root/miniconda3/envs/bdetr/bin/python -m pytest tests/test_mcln_optuna_contract.py -q`

Expected: all tests pass.

### Task 5: Baseline And Two-Epoch Trial Runner

**Files:**
- Create: `scripts/tuning/train_mcln_optuna_trial.py`
- Create: `tests/test_train_mcln_optuna_trial.py`

- [ ] **Step 1: Write failing loop-contract tests with fakes**

Inject fake loaders, trainer, model, evaluator, scheduler and checkpoint writer. Require baseline mode performs zero optimizer steps and one calibration evaluation. Require trial mode trains exactly global epochs 55 and 56, evaluates after each, and publishes only the epoch-56 checkpoint:

```python
result = run_trial_core(context, mode="trial")
assert context.trained_epochs == [55, 56]
assert context.evaluated_epochs == [55, 56]
assert result["selection_epoch"] == 56
assert result["metrics"]["epoch_55"]["sample_count"] == 3625
assert result["metrics"]["epoch_56"]["sample_count"] == 3625
```

Require a 46-epoch cosine LambdaLR horizon and verify its factors at step 0, two epochs, and the final horizon.

- [ ] **Step 2: Run and verify the missing runner failure**

Run: `/root/miniconda3/envs/bdetr/bin/python -m pytest tests/test_train_mcln_optuna_trial.py -q`

Expected: collection fails because the runner is absent.

- [ ] **Step 3: Implement reusable initialization and atomic receipts**

The runner must initialize NCCL, seed Python/NumPy/Torch, construct only train data, create the `TrainTester`, build DDP, load epoch-54 with `reduce_lr=True`, and construct the strict optimizer. Add a pure scheduler helper:

```python
def cosine_factor(step, total_steps):
    if total_steps <= 0 or not 0 <= step <= total_steps:
        raise ValueError("cosine step is outside the fixed horizon")
    return 0.5 * (1.0 + math.cos(math.pi * step / float(total_steps)))
```

Use a `LambdaLR` stepped by the existing batch loop. Write receipts to a temporary sibling and `os.replace` them atomically.

- [ ] **Step 4: Implement baseline and trial CLI modes**

Required custom arguments:

```text
--mode baseline|trial
--receipt-path PATH
--checkpoint-output PATH
--base-checkpoint PATH
--expected-base-sha256 HEX
--split-seed 0
--calibration-fraction 0.10
--continuation-horizon 46
```

Baseline exports fixed-default and mask metrics. Trial mode requires all seven resolved Optuna parameters, trains exactly two epochs, and exports both epoch receipts plus optimizer-group names/LRs.

- [ ] **Step 5: Run unit tests and a dry command test**

Run: `/root/miniconda3/envs/bdetr/bin/python -m pytest tests/test_train_mcln_optuna_trial.py -q`

Expected: all tests pass, including no construction of a validation dataset.

### Task 6: Resumable Study Orchestrator

**Files:**
- Create: `scripts/tuning/optuna_mcln_complete_retrain.py`
- Create: `tests/test_optuna_mcln_complete_retrain.py`

- [ ] **Step 1: Write failing subprocess-command tests**

Assert the command fixes batch 18, source pair, fresh optimizer, epoch count, and every selected parameter:

```python
command = command_for_trial(args, params, trial_number=7)
assert command_value(command, "--batch_size") == "18"
assert command_value(command, "--source_choice_selector_sources") == (
    "default,default_rank_blend_contrastive010"
)
assert command_value(command, "--mask_loss_scale") == str(params["mask_loss_scale"])
assert command_value(command, "--consistency_loss_scale") == str(
    params["consistency_loss_scale"]
)
assert "--reduce_lr" in command
```

- [ ] **Step 2: Write failing resume tests**

Build an in-memory Optuna study containing complete, failed, and running trials. Require `remaining_successful_trials(study, 20)` ignores failed/running and returns the exact missing count. Require three presets are enqueued only for a new study.

- [ ] **Step 3: Write failing publication and long-dispatch tests**

Mock subprocess results and filesystem calls. Require no feasible trial writes `selection_status=no_feasible_trial` and never dispatches long training. Require a feasible best writes `best.json`, hardlinks its checkpoint, and dispatches one `train_mcln_complete_long.py` command with `start_new_session=True`.

- [ ] **Step 4: Implement the orchestrator**

Use `TPESampler(seed=0, n_startup_trials=5)` and SQLite storage. Run the baseline subprocess once, then loop one Optuna attempt at a time until 20 structurally valid `COMPLETE` trial receipts exist. Cap process attempts at 60 so persistent infrastructure failure exits instead of looping forever.

Persist `trial.set_user_attr("receipt", relative_path)` and `trial.set_user_attr("feasible", bool)`. Return the approved balanced objective for feasible trials and a deterministic negative constraint penalty for infeasible trials, while final selection still uses `select_best_trial` over feasible receipts only.

- [ ] **Step 5: Implement report and cleanup publication**

After every trial, atomically regenerate `trials.csv`, `study_summary.json`, and current `best.json`. Keep one global best short checkpoint. Never delete logs/config/receipts or protected inputs.

- [ ] **Step 6: Run orchestrator tests**

Run: `/root/miniconda3/envs/bdetr/bin/python -m pytest tests/test_optuna_mcln_complete_retrain.py -q`

Expected: all tests pass.

### Task 7: Full-Train Long Runner And Pareto Retention

**Files:**
- Create: `scripts/tuning/train_mcln_complete_long.py`
- Create: `tests/test_train_mcln_complete_long.py`

- [ ] **Step 1: Write failing schedule tests**

Require global epochs 55 through 100 inclusive and official validation only at the approved nine epochs:

```python
assert long_train_epochs(base_epoch=54, final_epoch=100) == tuple(range(55, 101))
assert validation_epochs() == (60, 65, 70, 75, 80, 85, 90, 95, 100)
```

- [ ] **Step 2: Write failing atomic-latest and Pareto tests**

Use small fake checkpoint files and five-metric receipts. Require dominated candidates are removed, at most three stable candidates remain, and selection roles are `target_distance`, `position025`, and `mask_balance`. Require protected inputs are rejected as deletion targets.

- [ ] **Step 3: Implement long-loop pure helpers**

Expose `long_train_epochs`, `validation_epochs`, `dominates`, `target_distance`, `mask_balance`, and `select_pareto_checkpoints`. Use exact final targets `(0.59, 4621/9508, 5582/9508, 4821/9508, 0.4472)` for distance only; do not claim a checkpoint passes until strict release gates are evaluated.

- [ ] **Step 4: Implement the GPU long runner**

Load `best.json`, revalidate its study contract and base checksum, build the standard full train and official val datasets, then train epochs 55-100 with the same 46-epoch cosine factor. Atomically overwrite `latest.pth` after each epoch. At approved validation epochs, save a candidate, export exact metrics, apply Pareto retention, and update `long_summary.json`.

On clean completion write `sidecar_handoff.json` containing each retained backbone path, SHA-256, epoch, five metrics, source snapshot digest, and the existing parent/geometry/joint rebuild entry points.

- [ ] **Step 5: Run long-run unit tests**

Run: `/root/miniconda3/envs/bdetr/bin/python -m pytest tests/test_train_mcln_complete_long.py -q`

Expected: all tests pass without loading CUDA or real ScanRefer data.

### Task 8: Provenance, Protection, And Stable Launcher

**Files:**
- Create: `scripts/tuning/mcln_retrain_provenance.py`
- Create: `tests/test_mcln_retrain_provenance.py`
- Create: `scripts/tuning/run_optuna_mcln_complete_retrain20.sh`

- [ ] **Step 1: Write failing provenance tests**

Require SHA-256 streaming, exact mode/size verification, atomic JSON rejecting NaN, source manifest path sorting, and snapshot exclusions for `.pth`, `.pt`, `__pycache__`, `.pytest_cache`, and output directories.

- [ ] **Step 2: Implement provenance helpers**

Public functions:

```python
def sha256_file(path, chunk_size=1024 * 1024):
    digest = hashlib.sha256()
    with open(str(path), "rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_file_contract(path, sha256, size, mode):
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(str(path))
    stat_result = path.stat()
    actual = {
        "path": str(path),
        "size": int(stat_result.st_size),
        "mode": int(stat.S_IMODE(stat_result.st_mode)),
        "sha256": sha256_file(path),
    }
    expected = {
        "size": int(size),
        "mode": int(mode),
        "sha256": str(sha256),
    }
    for key, value in expected.items():
        if actual[key] != value:
            raise ValueError("file contract mismatch for {}: {}".format(path, key))
    return actual


def atomic_json_dump(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp.{}".format(os.getpid()))
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(
                payload, handle, sort_keys=True, indent=2,
                ensure_ascii=True, allow_nan=False,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(str(temporary), str(path))
    finally:
        if temporary.exists():
            temporary.unlink()
```

Implement environment capture and a gzip tar source snapshot with a canonical manifest.

- [ ] **Step 3: Implement the shell launcher**

The launcher must resolve defaults without repurposing system environment variables:

```bash
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/bdetr/bin/python}"
DATA_ROOT="${DATA_ROOT:-/root/autodl-tmp/DATA_ROOT/}"
BASE_CHECKPOINT="${BASE_CHECKPOINT:-${REPO_ROOT}/pretained model/ckpt_epoch_54.pth}"
N_TRIALS="${N_TRIALS:-20}"
GPU="${GPU:-0}"
```

Create a UTC run ID, output under `/root/autodl-tmp/DATA_ROOT/output/tuning/`, write PID/log paths, and `exec` the orchestrator. The outer formal launch will use `nohup` so the study survives the current shell.

- [ ] **Step 4: Run provenance and shell syntax tests**

Run:

```bash
/root/miniconda3/envs/bdetr/bin/python -m pytest tests/test_mcln_retrain_provenance.py -q
bash -n scripts/tuning/run_optuna_mcln_complete_retrain20.sh
```

Expected: tests pass and `bash -n` exits zero.

### Task 9: Full Verification And Formal Launch

**Files:**
- Modify: `docs/REC_3DRES_OPTIMIZATION_LOG.md`

- [ ] **Step 1: Run every new focused test**

Run:

```bash
/root/miniconda3/envs/bdetr/bin/python -m pytest \
  tests/test_mcln_training_groups.py \
  tests/test_mcln_retrain_metrics.py \
  tests/test_scanrefer_train_only.py \
  tests/test_mcln_optuna_contract.py \
  tests/test_train_mcln_optuna_trial.py \
  tests/test_optuna_mcln_complete_retrain.py \
  tests/test_train_mcln_complete_long.py \
  tests/test_mcln_retrain_provenance.py -q
```

Expected: all tests pass.

- [ ] **Step 2: Run relevant legacy regression tests**

Run:

```bash
/root/miniconda3/envs/bdetr/bin/python -m pytest \
  tests/test_source_choice_selector.py \
  tests/test_source_choice_adapter.py \
  tests/test_grounding_evaluator_source_choice.py \
  tests/test_main_utils_source_choice_checkpoint.py \
  tests/test_monitor_mcln_source_choice_best.py -q
```

Expected: all tests pass.

- [ ] **Step 3: Run the full CPU suite**

Run: `/root/miniconda3/envs/bdetr/bin/python -m pytest tests -q`

Expected: zero failures.

- [ ] **Step 4: Protect and re-audit inputs**

Run:

```bash
chmod 0444 'pretained model/ckpt_epoch_54.pth'
sha256sum 'pretained model/ckpt_epoch_54.pth'
stat -c '%n|%s|%a' 'pretained model/ckpt_epoch_54.pth'
df -B1 /root/autodl-tmp
nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu --format=csv,noheader
```

Expected: digest `a9930065996fce1d0dd5ee9fe00a120bdb3a2c88d158b7a3666717d842ac113d`, size `793041121`, mode `444`, at least 8 GiB free, and GPU 0 available.

- [ ] **Step 5: Create the immutable source snapshot**

Run the provenance CLI to write `source_snapshot.tar.gz`, `source_manifest.json`, `environment.json`, and `inputs.json` into the new study root. Re-run its verifier and require identical manifest SHA-256.

- [ ] **Step 6: Run one real GPU smoke batch**

Launch the trial runner in smoke mode with one training batch and one calibration batch. Require finite loss, finite gradients in decoder/backbone/mask/selector groups, expected group LRs, no official-val file access, and no retained 0.79 GB checkpoint.

- [ ] **Step 7: Launch the 20-trial study**

Run the stable launcher detached with explicit paths and capture its PID:

```bash
nohup env \
  PYTHON_BIN=/root/miniconda3/envs/bdetr/bin/python \
  DATA_ROOT=/root/autodl-tmp/DATA_ROOT/ \
  N_TRIALS=20 GPU=0 \
  bash scripts/tuning/run_optuna_mcln_complete_retrain20.sh \
  > /root/autodl-tmp/DATA_ROOT/output/tuning/mcln_complete_retrain_launcher.log 2>&1 \
  &
```

Expected within the first baseline phase: a live orchestrator PID, nonzero GPU memory during evaluation, `study_contract.json`, `baseline_metrics.json` after 3,625 rows, and no access to official validation inputs.

- [ ] **Step 8: Update the optimization handoff log**

Append the design/plan paths, fixed code issues, test counts, epoch-54 digest/mode, study root, PID, launcher log, Optuna DB, baseline receipt status, fixed search contract, estimated 60-80 GPU hours, and automatic long-run behavior to `docs/REC_3DRES_OPTIMIZATION_LOG.md`.

- [ ] **Step 9: Record the implementation snapshot checkpoint**

Generate a final post-launch source manifest and SHA-256 receipt. Because the workspace has no Git repository, do not run `git init`, `git add`, or `git commit`.
