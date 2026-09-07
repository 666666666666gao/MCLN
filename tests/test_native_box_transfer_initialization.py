import copy

import pytest
import torch

from scripts.export_native_box_transfer_initialization import build_initialization


def fixture():
    suffixes = ['net.0.weight', 'net.1.weight', 'net.1.bias', 'net.4.weight',
                'net.5.weight', 'net.5.bias', 'net.8.weight', 'net.8.bias']
    names = ['prediction_heads.5.' + head + '.' + suffix
             for head in ['center_residual_head', 'size_pred_head'] for suffix in suffixes]
    base = {'model': {'module.' + name: torch.zeros(2) for name in names},
            'config': {'dataset': ['scanrefer']}, 'optimizer': {'old': True}, 'epoch': 71}
    base['model']['module.backbone_net.frozen'] = torch.ones(3)
    for head in ['center_residual_head', 'size_pred_head']:
        for layer in [1, 5]:
            prefix = 'module.prediction_heads.5.' + head + '.net.' + str(layer) + '.'
            base['model'][prefix + 'running_mean'] = torch.zeros(2)
            base['model'][prefix + 'running_var'] = torch.ones(2)
            base['model'][prefix + 'num_batches_tracked'] = torch.tensor(7)
    endpoint = {'schema': 'mcln-scanrefer-native-box-head-state-v1', 'arm': 'gt_teacher_box',
                'steps': 2482, 'pretrained_artifacts': {'backbone': {'sha256': 'base'}},
                'head_parameters': {name: torch.ones(2) for name in names},
                'core_trainable_tensors': names, 'manifest_sha256': 'training', 'optimizer': {'old': True}}
    return base, endpoint


def test_native_prefix_preserves_frozen_state_and_discards_both_optimizers():
    base, endpoint = fixture()
    original = copy.deepcopy(base)
    output = build_initialization(base, endpoint, 'base')
    assert set(output['model']) == set(base['model'])
    assert torch.equal(output['model']['module.backbone_net.frozen'], original['model']['module.backbone_net.frozen'])
    for name, value in original['model'].items():
        if name[7:] not in endpoint['head_parameters']:
            assert torch.equal(output['model'][name], value)
    for name, value in endpoint['head_parameters'].items():
        assert torch.equal(output['model']['module.' + name], value)
        assert torch.equal(base['model']['module.' + name], original['model']['module.' + name])
    assert 'optimizer' not in output and 'scheduler' not in output and output['epoch'] == 0


@pytest.mark.parametrize('failure', ['wrong_base', 'missing_head', 'extra_head', 'bn_buffer', 'wrong_shape', 'wrong_dtype', 'control_arm'])
def test_incompatible_delta_is_not_exported(failure):
    base, endpoint = fixture()
    name = next(iter(endpoint['head_parameters']))
    if failure == 'wrong_base':
        endpoint['pretrained_artifacts']['backbone']['sha256'] = 'other'
    elif failure == 'missing_head':
        del endpoint['head_parameters'][name]
    elif failure == 'extra_head':
        endpoint['head_parameters']['backbone_net.frozen'] = torch.ones(3)
    elif failure == 'bn_buffer':
        endpoint['head_parameters']['prediction_heads.5.center_residual_head.net.1.running_mean'] = torch.ones(2)
    elif failure == 'wrong_shape':
        endpoint['head_parameters'][name] = torch.ones(3)
    elif failure == 'wrong_dtype':
        endpoint['head_parameters'][name] = torch.ones(2, dtype=torch.float64)
    elif failure == 'control_arm':
        endpoint['arm'] = 'gt_only'
    with pytest.raises(AssertionError):
        build_initialization(base, endpoint, 'base')
