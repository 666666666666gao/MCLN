#!/usr/bin/env python3
"""Fast public-contract smoke for the V106 graph mask refiner."""

import argparse
import json
from pathlib import Path
import sys

import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from models.mask_fusion import BoundaryAwareSuperpointGraphMaskRefiner


SCHEMA = "mcln-v106-graph-mask-contract-smoke-v1"


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


def _inputs(query_count=5, superpoint_count=13, d_model=288):
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


def _activate_head(model):
    with torch.no_grad():
        for parameter in (
                model.graph_coefficients.weight,
                model.graph_coefficients.bias):
            values = torch.linspace(
                -0.02, 0.02, parameter.numel(), dtype=parameter.dtype
            ).reshape_as(parameter)
            parameter.copy_(values)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output")
    args = parser.parse_args()
    torch.manual_seed(106)

    parameter_numel = None
    tensor_count = None
    for graph_mode in BoundaryAwareSuperpointGraphMaskRefiner.GRAPH_MODES:
        model = BoundaryAwareSuperpointGraphMaskRefiner(graph_mode=graph_mode)
        inputs = _inputs()
        result = model(*inputs)
        if any(not torch.equal(row, torch.zeros_like(row))
               for row in result["residuals"]):
            raise RuntimeError("zero initialization changed parent masks")
        observed = (
            sum(value.numel() for value in model.parameters()),
            len(model.state_dict()),
        )
        if parameter_numel is None:
            parameter_numel, tensor_count = observed
        if observed != (parameter_numel, tensor_count):
            raise RuntimeError("graph modes changed architecture shape")

    if (parameter_numel, tensor_count) != (2888, 4):
        raise RuntimeError(
            "unexpected graph size: {}/{}".format(parameter_numel, tensor_count)
        )

    model = BoundaryAwareSuperpointGraphMaskRefiner(graph_mode="bilateral")
    _activate_head(model)
    inputs = _inputs()
    result = model(*inputs)
    residual = result["residuals"][0]
    if (residual.shape != (5, 13)
            or not bool(torch.isfinite(residual).all().item())
            or float(residual.abs().mean().item()) <= 0.0
            or float(result["superpoint_std_mean"].item()) <= 0.0
            or float(result["query_std_mean"].item()) <= 0.0):
        raise RuntimeError("activated graph residual contract failed")

    feature_row = inputs[1][0].detach().transpose(0, 1)
    xyz_row = inputs[2][0].detach().squeeze(0)
    graph = model._build_graph(xyz_row, feature_row)
    indices = graph["indices"]
    row_indices = torch.arange(indices.shape[0]).unsqueeze(1)
    no_self = bool((indices != row_indices).all().item())
    no_duplicates = all(
        torch.unique(row).numel() == row.numel() for row in indices
    )
    in_bounds = bool(((indices >= 0) & (indices < indices.shape[0])).all().item())
    if not (no_self and no_duplicates and in_bounds):
        raise RuntimeError("KNN index contract failed")

    query_permutation = torch.tensor([3, 0, 4, 1, 2])
    superpoint_permutation = torch.tensor([8, 2, 12, 5, 0, 7, 1, 10, 4, 6, 3, 11, 9])
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
        raise RuntimeError("graph refiner is not permutation equivariant")

    swapped = model(
        query, features, xyz, boxes, source, text, [1.0 - alpha[0]],
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
        raise RuntimeError("graph refiner leaked gradients into parent inputs")

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
        "knn_no_self": no_self,
        "knn_no_duplicates": no_duplicates,
        "knn_in_bounds": in_bounds,
        "source_swap_finite": True,
        "parent_inputs_detached": True,
    }
    _write_json(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
