import json
import math
from types import SimpleNamespace

import pytest

from scripts.tuning.mcln_optuna_contract import (
    EXPECTED_CALIBRATION_COUNT,
    METRICS_SCHEMA,
)
from scripts.tuning import train_mcln_optuna_trial as runner


def _metrics(offset=0, sample_count=EXPECTED_CALIBRATION_COUNT):
    iou_sum = 1600.0 + offset
    return {
        "schema": METRICS_SCHEMA,
        "sample_count": sample_count,
        "position": {
            "fixed_default": {
                "hits025": 2200 + offset,
                "hits050": 1700 + offset,
            },
            "learned_selector": {
                "hits025": 2210 + offset,
                "hits050": 1710 + offset,
            },
        },
        "mask": {
            "hits025": 2250 + offset,
            "hits050": 1850 + offset,
            "iou_sum": iou_sum,
            "miou": iou_sum / float(sample_count),
        },
    }


def _smoke_metrics(sample_count=18):
    iou_sum = sample_count * 0.5
    return {
        "schema": METRICS_SCHEMA,
        "sample_count": sample_count,
        "position": {
            "fixed_default": {"hits025": 12, "hits050": 9},
            "learned_selector": {"hits025": 13, "hits050": 10},
        },
        "mask": {
            "hits025": 12,
            "hits050": 9,
            "iou_sum": iou_sum,
            "miou": iou_sum / float(sample_count),
        },
    }


def _loss_receipt(epoch):
    return {
        "schema": "mcln-train-loss-epoch-v1",
        "batch_count": 3,
        "loss_means": {
            "mask_loss": float(epoch) / 100.0,
            "total_loss": float(epoch) / 10.0,
        },
    }


class _FakeContext:
    def __init__(self):
        self.trained_epochs = []
        self.evaluated_epochs = []
        self.published_epochs = []
        self.optimizer_steps = 0
        self.steps_per_epoch = 3

    def train_epoch(self, epoch):
        self.trained_epochs.append(epoch)
        self.optimizer_steps += self.steps_per_epoch
        return _loss_receipt(epoch)

    def evaluate_epoch(self, epoch):
        self.evaluated_epochs.append(epoch)
        return _metrics(offset=epoch - 54)

    def publish_checkpoint(self, epoch):
        self.published_epochs.append(epoch)
        return "/tmp/epoch_{}.pth".format(epoch)

    def optimizer_group_receipt(self):
        return (
            {
                "name": "decoder",
                "initial_lr": 2e-5,
                "parameter_names": ("decoder.weight",),
            },
            {
                "name": "backbone",
                "initial_lr": 2e-4,
                "parameter_names": ("backbone_net.weight",),
            },
            {
                "name": "mask_head",
                "initial_lr": 4e-5,
                "parameter_names": ("x_mask.weight",),
            },
            {
                "name": "selector",
                "initial_lr": 5e-4,
                "parameter_names": ("source_choice_selector.weight",),
            },
        )

    def optimizer_gradient_receipt(self):
        return tuple({
            "name": name,
            "gradient_tensor_count": 1,
            "all_finite": True,
            "total_norm": float(index + 1),
        } for index, name in enumerate(
            ("decoder", "backbone", "mask_head", "selector")
        ))


def test_baseline_mode_never_trains_and_evaluates_calibration_once():
    context = _FakeContext()

    result = runner.run_trial_core(context, mode="baseline")

    assert context.trained_epochs == []
    assert context.optimizer_steps == 0
    assert context.evaluated_epochs == [54]
    assert context.published_epochs == []
    assert result["mode"] == "baseline"
    assert result["selection_epoch"] == 54
    assert result["metrics"] == {"epoch_54": _metrics(offset=0)}
    assert result["checkpoint"] is None
    assert "losses" not in result


def test_trial_mode_trains_exactly_epochs_55_and_56_and_publishes_only_56():
    context = _FakeContext()

    result = runner.run_trial_core(context, mode="trial")

    assert context.trained_epochs == [55, 56]
    assert context.optimizer_steps == 6
    assert context.evaluated_epochs == [55, 56]
    assert context.published_epochs == [56]
    assert result["selection_epoch"] == 56
    assert result["metrics"]["epoch_55"]["sample_count"] == 3625
    assert result["metrics"]["epoch_56"]["sample_count"] == 3625
    assert result["losses"] == {
        "epoch_55": _loss_receipt(55),
        "epoch_56": _loss_receipt(56),
    }
    assert result["checkpoint"] == "/tmp/epoch_56.pth"
    assert [group["name"] for group in result["optimizer_groups"]] == [
        "decoder", "backbone", "mask_head", "selector"
    ]
    assert result["optimizer_groups"][0]["parameter_names"] == (
        "decoder.weight",
    )


def test_smoke_mode_runs_one_epoch_without_publishing_checkpoint():
    context = _FakeContext()
    def evaluate_smoke(epoch):
        context.evaluated_epochs.append(epoch)
        return _smoke_metrics()
    context.evaluate_epoch = evaluate_smoke

    result = runner.run_smoke_core(context, expected_sample_count=18)

    assert context.trained_epochs == [55]
    assert context.optimizer_steps == 3
    assert context.evaluated_epochs == [55]
    assert context.published_epochs == []
    assert result["mode"] == "smoke"
    assert result["checkpoint"] is None
    assert result["losses"] == {"epoch_55": _loss_receipt(55)}
    assert [group["name"] for group in result["gradients"]] == [
        "decoder", "backbone", "mask_head", "selector"
    ]
    assert all(group["all_finite"] for group in result["gradients"])


def test_smoke_core_rejects_wrong_calibration_batch_size():
    context = _FakeContext()
    context.evaluate_epoch = lambda _epoch: _smoke_metrics(sample_count=17)

    with pytest.raises(ValueError):
        runner.run_smoke_core(context, expected_sample_count=18)


def test_trial_core_rejects_unknown_mode_and_invalid_metrics():
    context = _FakeContext()
    with pytest.raises(ValueError):
        runner.run_trial_core(context, mode="validation")

    context.evaluate_epoch = lambda _epoch: _metrics(offset=0)
    invalid = context.evaluate_epoch(54)
    invalid["sample_count"] = 3624
    context.evaluate_epoch = lambda _epoch: invalid
    with pytest.raises(ValueError):
        runner.run_trial_core(context, mode="baseline")


@pytest.mark.parametrize(
    "step,total_steps,expected",
    [
        (0, 46 * 7, 1.0),
        (
            2 * 7,
            46 * 7,
            0.5 * (1.0 + math.cos(math.pi * 2.0 / 46.0)),
        ),
        (46 * 7, 46 * 7, 0.0),
    ],
)
def test_cosine_factor_uses_the_fixed_46_epoch_horizon(
        step, total_steps, expected):
    assert runner.cosine_factor(step, total_steps) == pytest.approx(
        expected, abs=1e-15
    )


@pytest.mark.parametrize(
    "step,total_steps",
    [(-1, 46), (47, 46), (0, 0), (0, -1), (1.5, 46), (1, 46.0)],
)
def test_cosine_factor_rejects_invalid_steps(step, total_steps):
    with pytest.raises(ValueError):
        runner.cosine_factor(step, total_steps)


def test_cosine_horizon_steps_requires_positive_integer_contract():
    assert runner.cosine_horizon_steps(7, continuation_horizon=46) == 322
    for steps, horizon in ((0, 46), (7, 0), (1.5, 46), (7, True)):
        with pytest.raises(ValueError):
            runner.cosine_horizon_steps(
                steps, continuation_horizon=horizon
            )


def test_atomic_json_receipt_uses_sibling_replace_and_leaves_no_temp(
        tmp_path):
    path = tmp_path / "trial_receipt.json"
    payload = {"schema": "mcln-optuna-trial-v1", "complete": True}

    runner.atomic_write_json(path, payload)

    assert json.loads(path.read_text()) == payload
    assert list(tmp_path.glob("*.tmp")) == []


def _study_binding(base_sha256="a" * 64):
    split_metadata = dict(runner.AUTHORITATIVE_SCANREFER_SPLIT_METADATA)
    return {
        "study_contract_digest": "d" * 64,
        "source_snapshot_digest": "b" * 64,
        "base_checkpoint_sha256": base_sha256,
        "pointnet_checkpoint_sha256": "c" * 64,
        "repo_root": "/workspace/repo",
        "data_root": "/data/root",
        "python_bin": "/opt/python",
        "run_manifest_sha256": "e" * 64,
        "environment_sha256": "f" * 64,
        "inputs_sha256": "1" * 64,
        "study_name": "mcln-test",
        "storage_identity": "/tmp/optuna.db",
        "split_metadata": split_metadata,
        "split_metadata_sha256": runner.canonical_json_sha256(split_metadata),
        "data_digests": {
            "fit_scene_sha256": split_metadata["fit_scene_sha256"],
            "calibration_scene_sha256": split_metadata[
                "calibration_scene_sha256"
            ],
            "mapping_sha256": split_metadata["mapping_sha256"],
        },
        "continuation_horizon": 46,
    }


TRIAL_ARGS = [
    "--mode", "trial",
    "--receipt-path", "/tmp/receipt.json",
    "--checkpoint-output", "/tmp/trial.pth",
    "--base-checkpoint", "/tmp/epoch54.pth",
    "--expected-base-sha256", "a" * 64,
    "--split-seed", "0",
    "--calibration-fraction", "0.10",
    "--continuation-horizon", "46",
    "--study-binding-json", json.dumps(_study_binding(), sort_keys=True),
    "--decoder-lr", "0.00002",
    "--mask-head-lr-multiplier", "2",
    "--selector-lr", "0.0005",
    "--mask-loss-scale", "2",
    "--consistency-loss-scale", "0.5",
    "--selector-loss-weight", "0.5",
    "--selector-min-iou-gap", "0.03",
]


def test_trial_argument_parser_requires_all_seven_resolved_parameters():
    args = runner.parse_runner_args(TRIAL_ARGS)

    assert args.mode == "trial"
    assert runner.resolved_trial_params(args) == {
        "decoder_lr": 2e-5,
        "mask_head_lr_multiplier": 2.0,
        "selector_lr": 5e-4,
        "mask_loss_scale": 2.0,
        "consistency_loss_scale": 0.5,
        "selector_loss_weight": 0.5,
        "selector_min_iou_gap": 0.03,
    }

    with pytest.raises(ValueError):
        runner.parse_runner_args(TRIAL_ARGS[:-2])


def test_baseline_argument_parser_does_not_require_trial_parameters():
    args = runner.parse_runner_args([
        "--mode", "baseline",
        "--receipt-path", "/tmp/baseline.json",
        "--base-checkpoint", "/tmp/epoch54.pth",
        "--expected-base-sha256", "b" * 64,
        "--study-binding-json", json.dumps(
            _study_binding(base_sha256="b" * 64), sort_keys=True
        ),
    ])

    assert args.mode == "baseline"
    assert args.checkpoint_output is None
    assert runner.resolved_trial_params(args) is None


def test_smoke_argument_parser_requires_params_but_no_checkpoint_output():
    smoke_args = list(TRIAL_ARGS)
    smoke_args[1] = "smoke"
    checkpoint_index = smoke_args.index("--checkpoint-output")
    del smoke_args[checkpoint_index:checkpoint_index + 2]

    args = runner.parse_runner_args(smoke_args)

    assert args.mode == "smoke"
    assert args.checkpoint_output is None
    assert runner.resolved_trial_params(args)["decoder_lr"] == 2e-5


def test_runner_parser_does_not_abbreviate_shared_exp_as_expected_hash():
    smoke_args = list(TRIAL_ARGS)
    smoke_args[1] = "smoke"
    checkpoint_index = smoke_args.index("--checkpoint-output")
    del smoke_args[checkpoint_index:checkpoint_index + 2]
    smoke_args.extend(["--exp", "smoke"])

    args = runner.parse_runner_args(smoke_args)

    assert args.expected_base_sha256 == "a" * 64


def test_limited_loader_exposes_only_requested_batches():
    pulled = []
    class Loader:
        dataset = object()
        sampler = object()

        def __iter__(self):
            for value in ("first", "second", "third"):
                pulled.append(value)
                yield value

        def __len__(self):
            return 3

    limited = runner.LimitedLoader(Loader(), max_batches=1)

    assert list(limited) == ["first"]
    assert len(limited) == 1
    assert limited.dataset is Loader.dataset
    assert limited.sampler is Loader.sampler
    assert pulled == ["first"]


def test_apply_runner_contract_fixes_fresh_optimizer_and_model_settings():
    shared = SimpleNamespace()
    custom = runner.parse_runner_args(TRIAL_ARGS)

    runner.apply_runner_contract(shared, custom)

    assert shared.checkpoint_path == "/tmp/epoch54.pth"
    assert shared.reduce_lr is True
    assert shared.start_epoch == 55
    assert shared.max_epoch == 56
    assert shared.batch_size == 18
    assert shared.num_workers == 4
    assert shared.weight_decay == 5e-4
    assert shared.clip_norm == 0.1
    assert shared.model == "MCLN"
    assert shared.dataset == ["scanrefer"]
    assert shared.test_dataset == "scanrefer"
    assert shared.use_source_choice_selector is True
    assert shared.source_choice_selector_sources == (
        "default,default_rank_blend_contrastive010"
    )
    assert shared.source_choice_selector_default_source == "default"
    assert shared.lr == 2e-5
    assert shared.lr_backbone == 2e-4
    assert shared.mask_head_lr_multiplier == 2.0
    assert shared.expected_eval_sample_count == 3625


def test_train_only_tester_uses_fit_and_calibration_views_without_val(
        monkeypatch):
    fit = object()
    calibration = object()
    calls = []

    def fake_build_train_only_data(**kwargs):
        calls.append(kwargs)
        return {
            "fit_dataset": fit,
            "calibration_dataset": calibration,
        }

    monkeypatch.setattr(runner, "build_train_only_data", fake_build_train_only_data)
    args = SimpleNamespace(
        train_only_seed=0,
        train_only_calibration_fraction=0.10,
        data_root="/data/train",
        use_color=True,
        use_height=False,
        use_multiview=False,
        detect_intermediate=True,
        butd=True,
        butd_gt=False,
        butd_cls=False,
        augment_det=True,
        wo_obj_name="None",
        skip_missing_superpoints=True,
        test_dataset="scanrefer",
    )

    assert runner.TrainOnlyTrainTester.get_datasets(args) == (
        fit, calibration
    )
    assert len(calls) == 1
    assert calls[0]["formal_run"] is True
    assert calls[0]["split"] == "train"
    assert not any("val" in str(value).lower() for value in calls[0].values())
