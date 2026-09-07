"""Export a promoted ScanRefer box-head endpoint for native model-only training."""
import argparse
import datetime
import json
from pathlib import Path

import torch


PREFIXES = ('prediction_heads.5.center_residual_head.', 'prediction_heads.5.size_pred_head.')


def build_initialization(backbone, endpoint, backbone_sha256):
    assert endpoint['schema'] == 'mcln-scanrefer-native-box-head-state-v1'
    assert endpoint['arm'] == 'gt_teacher_box' and endpoint['steps'] == 2482
    assert endpoint['pretrained_artifacts']['backbone']['sha256'] == backbone_sha256
    original = backbone['model']
    assert all(name.startswith('module.') for name in original)
    # These existing MLPs also contain BatchNorm running buffers. The endpoint
    # updates only affine/Conv parameters; frozen buffers remain from E71.
    expected = {name[7:] for name in original
                if name[7:].startswith(PREFIXES) and name.endswith(('.weight', '.bias'))}
    assert len(expected) == 16
    assert set(endpoint['head_parameters']) == set(endpoint['core_trainable_tensors']) == expected
    state = dict(original)
    for name, value in endpoint['head_parameters'].items():
        previous = original['module.' + name]
        assert value.device.type == 'cpu' and previous.device.type == 'cpu'
        assert value.shape == previous.shape and value.dtype == previous.dtype, name
        assert torch.isfinite(value).all(), name
        state['module.' + name] = value
    return {'model': state, 'epoch': 0, 'config': backbone['config'],
            'initialization_provenance': {
                'schema': 'mcln-native-box-transfer-initialization-v1',
                'backbone_sha256': backbone_sha256,
                'training_manifest_sha256': endpoint['manifest_sha256'],
                'source_training_steps': endpoint['steps'],
                'replaced_tensors': sorted(expected),
                'optimizer_and_scheduler_transferred': False,
                'requires_model_only_initialization': True,
                'scanrefer_native_performance_not_implied_by_system_promotion': True}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--formal-directory', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    option = parser.parse_args()
    from scripts.audit_scanrefer_native_box_transfer_official import audit_inputs, audit_rows
    from scripts.evaluate_scanrefer_native_box_transfer_official import file_sha

    # Recompute the existing formal records; never create a new evaluation here.
    result = audit_rows(option.formal_directory)
    assert result['integrity_pass'] and result['promotion']['advance_to_nr3d_sr3d_rec']
    inputs = audit_inputs(option.formal_directory)
    independent = json.loads((option.formal_directory / 'result/independent_audit.json').read_text())
    assert independent['integrity_pass'] and independent['promotion'] == result['promotion']
    assert independent['receipt_sha256'] == result['receipt_sha256']
    assert independent['inputs'] == inputs
    manifest = json.loads((option.formal_directory / 'input_manifest.json').read_text())
    trained = manifest['trained_checkpoints']['gt_teacher_box']
    backbone_item = inputs['artifacts_verified_after_evaluation']['backbone']
    backbone = torch.load(backbone_item['path'], map_location='cpu')
    endpoint = torch.load(trained['path'], map_location='cpu')
    initialization = build_initialization(backbone, endpoint, backbone_item['sha256'])
    initialization['initialization_provenance'].update({
        'endpoint_sha256': trained['sha256'], 'formal_receipt_sha256': result['receipt_sha256'],
        'formal_audit_sha256': file_sha(option.formal_directory / 'result/independent_audit.json')})
    with option.output.open('xb') as stream:
        torch.save(initialization, stream)
    restored = torch.load(option.output, map_location='cpu')
    assert set(restored['model']) == set(initialization['model'])
    assert all(torch.equal(value, restored['model'][name]) for name, value in initialization['model'].items())
    receipt = {'schema': 'mcln-native-box-transfer-export-receipt-v1',
        'time_cst': datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
        'output': str(option.output), 'bytes': option.output.stat().st_size, 'sha256': file_sha(option.output),
        'restored_model_tensors': len(restored['model']),
        'provenance': initialization['initialization_provenance'],
        'native_dataset_model_load_still_required': True,
        'new_gpu_forwards': 0, 'new_optimizer_steps': 0, 'new_formal_rows': 0}
    with option.output.with_suffix('.receipt.json').open('x') as stream:
        json.dump(receipt, stream, indent=2, sort_keys=True)
        stream.write('\n')
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()
