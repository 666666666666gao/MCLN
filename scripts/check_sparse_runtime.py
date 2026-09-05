"""Validate pinned sparse CUDA operators against dense math and input indices."""

import argparse
import json
from pathlib import Path
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    import numpy as np
    import torch
    from torch import nn
    import spconv
    import spconv.pytorch as sparse
    import cumm

    assert torch.__version__ == '1.10.2+cu111' and np.__version__ == '1.21.5'
    assert spconv.__version__ == '2.3.6' and cumm.__version__ == '0.4.11'
    assert torch.cuda.is_available()
    torch.manual_seed(41)
    torch.cuda.manual_seed_all(41)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    spconv.constants.SPCONV_ALLOW_TF32 = False
    started = time.time()
    grid = torch.cartesian_prod(torch.arange(2), torch.arange(5), torch.arange(4), torch.arange(3))
    indices = grid[(grid[:, 1:].sum(dim=1) % 3) != 0].int().cuda()
    shape = [5, 4, 3]
    features = torch.randn(len(indices), 8, device='cuda', requires_grad=True)
    dense_features = features.detach().clone().requires_grad_(True)
    layer = sparse.SubMConv3d(8, 16, 3, padding=1, bias=True, indice_key='subm').cuda()
    dense_layer = nn.Conv3d(8, 16, 3, padding=1, bias=True).cuda()
    assert tuple(layer.weight.shape) == (16, 3, 3, 3, 8)
    with torch.no_grad():
        dense_layer.weight.copy_(layer.weight.permute(0, 4, 1, 2, 3))
        dense_layer.bias.copy_(layer.bias)
    source = sparse.SparseConvTensor(features, indices, shape, batch_size=2)
    result = layer(source)
    assert torch.equal(result.indices, indices)
    dense = features.new_zeros((2, 8) + tuple(shape))
    index = indices.long()
    dense[index[:, 0], :, index[:, 1], index[:, 2], index[:, 3]] = dense_features
    reference = dense_layer(dense)[index[:, 0], :, index[:, 1], index[:, 2], index[:, 3]]
    assert torch.allclose(result.features, reference, atol=1e-5, rtol=1e-4)
    target = torch.randn_like(reference)
    (result.features * target).sum().backward()
    (reference * target).sum().backward()
    comparisons = {
        'forward': (result.features, reference),
        'input_gradient': (features.grad, dense_features.grad),
        'weight_gradient': (layer.weight.grad.permute(0, 4, 1, 2, 3), dense_layer.weight.grad),
        'bias_gradient': (layer.bias.grad, dense_layer.bias.grad),
    }
    errors = {}
    for name, (actual, expected) in comparisons.items():
        assert torch.isfinite(actual).all() and torch.isfinite(expected).all()
        assert torch.allclose(actual, expected, atol=1e-5, rtol=1e-4), name
        errors[name] = float((actual - expected).abs().max())
    assert features.grad.norm() > 0 and layer.weight.grad.norm() > 0

    inverse_features = features.detach().clone().requires_grad_(True)
    down = sparse.SparseConv3d(8, 16, 3, stride=2, padding=1, bias=False, indice_key='scale').cuda()
    up = sparse.SparseInverseConv3d(16, 8, 3, bias=False, indice_key='scale').cuda()
    inverse_source = sparse.SparseConvTensor(inverse_features, indices, shape, batch_size=2)
    coarse = down(inverse_source)
    restored = up(coarse)
    assert torch.equal(restored.indices, indices)
    assert list(restored.spatial_shape) == shape
    assert restored.features.shape == inverse_features.shape and torch.isfinite(restored.features).all()
    restored.features.square().mean().backward()
    gradients = [inverse_features.grad, down.weight.grad, up.weight.grad]
    assert all(torch.isfinite(value).all() and value.norm() > 0 for value in gradients)
    torch.cuda.synchronize()
    receipt = {
        'schema': 'mcln-sparse-runtime-kernel-check-v1', 'status': 'pass',
        'torch': torch.__version__, 'spconv': spconv.__version__, 'cumm': cumm.__version__,
        'gpu': torch.cuda.get_device_name(0), 'active_voxels': len(indices), 'synthetic_batches': 2,
        'submanifold_indices_unchanged': True, 'inverse_indices_and_shape_restored': True,
        'max_abs_dense_reference_errors': errors, 'reference_atol': 1e-5, 'reference_rtol': 1e-4,
        'inverse_path_gradient_norms': [float(value.norm()) for value in gradients],
        'sparse_operator_forwards': 3, 'dense_reference_forwards': 1, 'backwards': 3,
        'native_model_forwards': 0, 'dataset_rows': 0, 'optimizer_steps': 0,
        'tf32_enabled': False, 'elapsed_seconds': time.time() - started,
        'max_gpu_allocated_bytes': torch.cuda.max_memory_allocated(),
    }
    with args.output.open('x') as stream:
        json.dump(receipt, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()
