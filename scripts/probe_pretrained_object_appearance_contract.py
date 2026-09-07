"""CPU contract checks for object-memory fusion, not model training or REC."""
import argparse
import importlib.util
import json
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--module', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    specification = importlib.util.spec_from_file_location('appearance', str(args.module))
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    torch.manual_seed(23)
    fusion = module.PretrainedObjectAppearance()
    original = torch.randn(2, 5, 288)
    features = torch.randn(2, 5, 1280)
    available = torch.tensor([[1, 0, 1, 1, 0], [0, 1, 1, 0, 1]], dtype=torch.bool)
    features[~available] = 0
    result = fusion(original, features, available)
    assert torch.equal(result, original)
    objective = (result[..., 128:] * torch.randn_like(result[..., 128:])).sum()
    objective.backward()
    assert torch.isfinite(fusion.projection.weight.grad).all()
    assert fusion.projection.weight.grad.abs().sum() > 0
    with torch.no_grad():
        fusion.projection.weight.add_(fusion.projection.weight.grad, alpha=-1e-4)
    changed = fusion(original, features, available)
    assert torch.equal(changed[..., :128], original[..., :128])
    assert torch.equal(changed[~available], original[~available])
    assert not torch.equal(changed[available], original[available])
    permutation = torch.tensor([3, 0, 4, 2, 1])
    permuted = fusion(original[:, permutation], features[:, permutation], available[:, permutation])
    assert torch.allclose(permuted, changed[:, permutation], atol=1e-6, rtol=1e-6)
    result = dict(status='pass', parameters=sum(v.numel() for v in fusion.parameters()),
                  initial_identity=True, position_channels_preserved=True,
                  unavailable_slots_preserved=True, gradient_nonzero=True,
                  slot_permutation_equivariant=True, synthetic_parameter_updates=1,
                  mcln_forwards=0, dataset_rows=0, formal_rows=0, new_rec_metrics=False)
    args.output.write_bytes(json.dumps(result, indent=2).encode() + b'\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
