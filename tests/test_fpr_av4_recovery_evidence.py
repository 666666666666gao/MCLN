from __future__ import print_function

import importlib.util
import json
import pathlib

import pytest


ROOT = pathlib.Path(__file__).resolve().parents[1]
VERIFIER_PATH = ROOT / "scripts" / (
    "verify_nr3d_fpr_tv_av4_failed_attempt.py"
)
EVIDENCE_PATH = ROOT / "scripts" / (
    "nr3d_fpr_tv_av4_failed_attempt_evidence_v1.json"
)


def _load_verifier():
    spec = importlib.util.spec_from_file_location(
        "verify_nr3d_fpr_tv_av4_failed_attempt", str(VERIFIER_PATH)
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _failed_main_source(optimizer_before_error=False):
    optimizer_line = "            optimizer.step()\n" if optimizer_before_error else ""
    trailing_optimizer = "" if optimizer_before_error else (
        "            optimizer.step()\n"
    )
    return (
        "class BaseTrainTester(object):\n"
        "    def train_one_epoch(self):\n"
        "            (loss / float(group_size)).backward()\n"
        + optimizer_line
        + "            raise ValueError(\n"
        "                \"actual Parent score-gradient audit is missing\"\n"
        "            )\n"
        + trailing_optimizer
        + "    # BRIEF eval\n"
    ).encode("utf-8")


def _failed_launch_log(progress="0/2806"):
    return (
        "  0%|          | {} [00:10<?, ?it/s]\n"
        "Traceback (most recent call last):\n"
        "  File \"main_utils.py\", line 7534, in train_one_epoch\n"
        "    \"actual Parent score-gradient audit is missing\"\n"
        "ValueError: actual Parent score-gradient audit is missing\n"
    ).format(progress).encode("utf-8")


def test_zero_step_proof_requires_failure_before_optimizer_step():
    verifier = _load_verifier()
    verifier._verify_zero_step_control_flow(
        _failed_main_source(), _failed_launch_log()
    )

    with pytest.raises(ValueError, match="pre-optimizer"):
        verifier._verify_zero_step_control_flow(
            _failed_main_source(optimizer_before_error=True),
            _failed_launch_log(),
        )


def test_zero_step_proof_requires_first_batch_progress_counter():
    verifier = _load_verifier()
    with pytest.raises(ValueError, match="lacks marker"):
        verifier._verify_zero_step_control_flow(
            _failed_main_source(), _failed_launch_log(progress="1/2806")
        )


def test_frozen_failure_evidence_declares_no_training_artifacts():
    evidence = json.loads(EVIDENCE_PATH.read_text(encoding="utf-8"))
    assert evidence["schema"] == (
        "mcln-fpr-tv-av4-failed-attempt-evidence-v1"
    )
    assert evidence["failure_stage"] == (
        "first_batch_post_backward_pre_optimizer_step_gradient_audit"
    )
    assert evidence["optimizer_steps"] == 0
    assert evidence["receipts"] == 0
    assert evidence["decisions"] == 0
    assert evidence["weights"] == 0
    assert evidence["formal_validation_accessed"] is False
