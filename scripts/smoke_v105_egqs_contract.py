#!/usr/bin/env python3
"""Fast public-contract smoke for the V105 EGQS mask refiner."""

import argparse
import json
import math
from pathlib import Path
import sys

import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from models.mask_fusion import EvidenceGeometryQuerySuperpointMaskRefiner


SCHEMA = "mcln-v105-egqs-contract-smoke-v1"


def _write_json(path, payload):
    if path is None:
        return
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(target)


def _inputs(query_count=5, superpoint_count=9, d_model=288):
    query = torch.randn(1, query_count, d_model, requires_grad=True)
    superpoints = [
        torch.randn(d_model, superpoint_count, requires_grad=True)
    ]
    xyz = [torch.randn(1, superpoint_count, 3, requires_grad=True)]
    boxes = torch.randn(1, query_count, 6, requires_grad=True)
    boxes = torch.cat((boxes[..., :3], boxes[..., 3:].abs() + 0.25), dim=-1)
    text = [torch.randn(1, query_count, superpoint_count, requires_grad=True)]
    source = [torch.randn(query_count, superpoint_count, requires_grad=True)]
    alpha = [torch.tensor(0.37, requires_grad=True)]
    return query, superpoints, xyz, boxes, text, source, alpha


def _activate_heads(model):
    with torch.no_grad():
        for parameter in (
                model.query_projection[-1].weight,
                model.query_projection[-1].bias,
                model.evidence_coefficients.weight,
                model.evidence_coefficients.bias,
                model.geometry_coefficients.weight,
                model.geometry_coefficients.bias):
            values = torch.linspace(
                -0.02, 0.02, parameter.numel(), dtype=parameter.dtype
            ).reshape_as(parameter)
            parameter.copy_(values)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output")
    args = parser.parse_args()
    torch.manual_seed(105)

    parameter_numel = None
    tensor_count = None
    for components in EvidenceGeometryQuerySuperpointMaskRefiner.COMPONENTS:
        model = EvidenceGeometryQuerySuperpointMaskRefiner(
            components=components
        )
        inputs = _inputs()
        result = model(*inputs)
        if any(not torch.equal(row, torch.zeros_like(row))
               for row in result["residuals"]):
            raise RuntimeError("zero initialization changed parent masks")
        observed_numel = sum(value.numel() for value in model.parameters())
        observed_tensors = len(model.state_dict())
        if parameter_numel is None:
            parameter_numel = observed_numel
            tensor_count = observed_tensors
        if (observed_numel, observed_tensors) != (parameter_numel, tensor_count):
            raise RuntimeError("component ablations changed architecture shape")

    if (parameter_numel, tensor_count) != (26095, 16):
        raise RuntimeError(
            "unexpected EGQS size: {}/{}".format(parameter_numel, tensor_count)
        )

    model = EvidenceGeometryQuerySuperpointMaskRefiner(components="all")
    _activate_heads(model)
    inputs = _inputs()
    result = model(*inputs)
    residual = result["residuals"][0]
    if (residual.shape != (5, 9)
            or not bool(torch.isfinite(residual).all().item())
            or float(residual.abs().mean().item()) <= 0.0
            or float(result["superpoint_std_mean"].item()) <= 0.0
            or float(result["query_std_mean"].item()) <= 0.0):
        raise RuntimeError("activated EGQS residual contract failed")

    query_permutation = torch.tensor([3, 0, 4, 1, 2])
    superpoint_permutation = torch.tensor([8, 2, 5, 0, 7, 1, 4, 6, 3])
    query, features, xyz, boxes, text, source, alpha = inputs
    permuted = model(
        query[:, query_permutation],
        [features[0][:, superpoint_permutation]],
        [xyz[0][:, superpoint_permutation]],
        boxes[:, query_permutation],
        [text[0][:, query_permutation][:, :, superpoint_permutation]],
        [source[0][query_permutation][:, superpoint_permutation]],
        alpha,
    )["residuals"][0]
    expected = residual[query_permutation][:, superpoint_permutation]
    max_permutation_error = float((permuted - expected).abs().max().item())
    if max_permutation_error > 2e-6:
        raise RuntimeError("EGQS is not query/superpoint permutation equivariant")

    swapped = model(
        query, features, xyz, boxes, source, text,
        [1.0 - alpha[0]],
    )["residuals"][0]
    if swapped.shape != residual.shape or not bool(torch.isfinite(swapped).all()):
        raise RuntimeError("source-swap contract failed")

    residual.sum().backward()
    detached_inputs = (
        query.grad is None
        and features[0].grad is None
        and xyz[0].grad is None
        and text[0].grad is None
        and source[0].grad is None
        and alpha[0].grad is None
    )
    if not detached_inputs:
        raise RuntimeError("EGQS leaked gradients into frozen parent inputs")

    payload = {
        "schema": SCHEMA,
        "pass": True,
        "parameter_numel": parameter_numel,
        "state_tensor_count": tensor_count,
        "zero_initialization_exact": True,
        "activated_residual_abs_mean": float(residual.abs().mean().item()),
        "activated_superpoint_std_mean": float(
            result["superpoint_std_mean"].item()
        ),
        "activated_query_std_mean": float(result["query_std_mean"].item()),
        "permutation_max_abs_error": max_permutation_error,
        "source_swap_finite": True,
        "parent_inputs_detached": True,
    }
    _write_json(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
