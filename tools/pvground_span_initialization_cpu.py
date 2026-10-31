"""Check the deployed mixer on synthetic CPU inputs; no checkpoint is edited.

Run from a deployed PV-Ground code directory:
PYTHONPATH=. python /path/to/pvground_span_initialization_cpu.py
"""
import hashlib
import inspect
import json
from pathlib import Path

import numpy as np
import torch

from extremal_span_mixer import ExtremalSpanMixer


def check():
    torch.set_num_threads(1)
    torch.manual_seed(111)
    module = ExtremalSpanMixer("extremal_support").eval()
    query = torch.zeros(256, 288)
    support = torch.zeros(288, 4)
    native_center = torch.zeros(256, 3)
    native_size = torch.full((256, 3), 2.0)
    mask_center = torch.ones(256, 3)
    mask_size = torch.full((256, 3), 4.0)
    text = torch.ones(256, 4)
    own = torch.ones(256, 4)
    alpha = torch.tensor(0.5)
    geometry = dict(
        native_ids=np.arange(4), origin=np.zeros(3), span=np.ones(3),
        count=np.ones(4), lower=np.zeros((4, 3)),
        upper=np.ones((4, 3)), mean=np.full((4, 3), 0.5))
    valid = torch.ones(256, dtype=torch.bool)
    rows = []
    assert torch.count_nonzero(module.output.weight) == 0
    assert torch.count_nonzero(module.output.bias) == 0
    for bias in (0.0, 0.5, 1.0):
        with torch.no_grad():
            module.output.bias.fill_(bias)
        module.zero_grad(set_to_none=True)
        center, size, evidence = module.mix_one(
            query, support, native_center, native_size, mask_center,
            mask_size, text, own, alpha, geometry, valid)
        assert torch.equal(center, (1-bias)*mask_center + bias*native_center)
        assert torch.equal(size, (1-bias)*mask_size + bias*native_size)
        assert torch.equal(evidence["axis_gate"], torch.full((256, 3), bias))
        (center.sum()+size.sum()).backward()
        gradient = module.output.bias.grad.item()
        assert gradient != 0
        rows.append(dict(output_bias=bias, gate=float(evidence["axis_gate"][0, 0]),
                         matches_expected=True, bias_gradient=gradient,
                         center_example=center[0].detach().tolist(),
                         size_example=size[0].detach().tolist()))
    source = Path(inspect.getfile(ExtremalSpanMixer))
    return dict(
        diagnostic="actual_extremal_span_mixer_cpu_endpoints",
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        torch_version=torch.__version__, seed=111, synthetic_inputs=True,
        gpu_used=False, training_modified=False, rows=rows)


if __name__ == "__main__":
    print(json.dumps(check(), indent=2))
