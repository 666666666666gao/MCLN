#!/usr/bin/env python3
import argparse
import json

import torch


def model_state(checkpoint):
    for key in ("model", "model_state_dict", "state_dict"):
        value = checkpoint.get(key)
        if isinstance(value, dict):
            return value
    raise ValueError("checkpoint has no model state")


def optimizer_state(checkpoint):
    for key in ("optimizer", "optimizer_state_dict"):
        value = checkpoint.get(key)
        if isinstance(value, dict) and isinstance(value.get("state"), dict):
            return value["state"]
    raise ValueError("checkpoint has no optimizer state")


def scalar_step(value):
    if isinstance(value, torch.Tensor):
        return int(value.item())
    return int(value)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--expected-step", required=True, type=int)
    parser.add_argument(
        "--expected-optimizer-numel", type=int, default=285236,
    )
    args = parser.parse_args()

    baseline = torch.load(args.baseline, map_location="cpu")
    trained = torch.load(args.checkpoint, map_location="cpu")
    baseline_state = model_state(baseline)
    trained_state = model_state(trained)

    common = sorted(set(baseline_state) & set(trained_state))
    changed = [
        name for name in common
        if not torch.equal(baseline_state[name], trained_state[name])
    ]
    new_names = sorted(set(trained_state) - set(baseline_state))
    joint_new_names = [
        name for name in new_names
        if name.startswith("joint_query_quality_reranker.")
        or name.startswith("module.joint_query_quality_reranker.")
    ]
    finite_new = all(
        bool(torch.isfinite(trained_state[name]).all().item())
        for name in joint_new_names
    )

    optimizer = optimizer_state(trained)
    steps = []
    optimizer_numel = 0
    moment_finite = True
    moment_nonzero = True
    for state in optimizer.values():
        if "step" in state:
            steps.append(scalar_step(state["step"]))
        exp_avg = state.get("exp_avg")
        exp_avg_sq = state.get("exp_avg_sq")
        if not isinstance(exp_avg, torch.Tensor):
            moment_finite = False
            moment_nonzero = False
            continue
        optimizer_numel += exp_avg.numel()
        for moment in (exp_avg, exp_avg_sq):
            if not isinstance(moment, torch.Tensor):
                moment_finite = False
                moment_nonzero = False
                continue
            moment_finite &= bool(torch.isfinite(moment).all().item())
        moment_nonzero &= bool(exp_avg.abs().sum().item() > 0.0)

    result = {
        "schema": "mcln-v70-trained-checkpoint-audit-v1",
        "baseline": args.baseline,
        "checkpoint": args.checkpoint,
        "common_tensor_count": len(common),
        "changed_common_tensor_count": len(changed),
        "changed_common_tensors": changed,
        "new_tensor_count": len(new_names),
        "joint_new_tensor_count": len(joint_new_names),
        "unexpected_new_tensors": sorted(set(new_names) - set(joint_new_names)),
        "new_tensors_finite": finite_new,
        "optimizer_state_count": len(optimizer),
        "optimizer_parameter_numel": optimizer_numel,
        "expected_optimizer_parameter_numel": (
            args.expected_optimizer_numel
        ),
        "optimizer_steps": sorted(set(steps)),
        "optimizer_moments_finite": moment_finite,
        "optimizer_exp_avg_nonzero_per_state": moment_nonzero,
    }
    result["pass"] = bool(
        len(common) == 1228
        and not changed
        and len(joint_new_names) == 30
        and not result["unexpected_new_tensors"]
        and finite_new
        and optimizer_numel == args.expected_optimizer_numel
        and result["optimizer_steps"] == [args.expected_step]
        and moment_finite
        and moment_nonzero
    )
    with open(args.output, "w", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
