"""Check the two observed zero-extent Sr3D objects using actual CPU inputs."""

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time


def file_sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', type=Path, required=True)
    options = parser.parse_args()
    root = options.manifest.parent
    manifest = json.loads(options.manifest.read_text())
    source = Path(manifest['model_source'])

    def verify_inputs():
        assert file_sha(source / 'g0_source_manifest.json') == manifest['source_manifest_sha256']
        for name, digest in json.loads((source / 'g0_source_manifest.json').read_text())['files'].items():
            assert file_sha(source / name) == digest, name
        for name, digest in manifest['files'].items():
            assert file_sha(root / name) == digest, name
        for name, metadata in manifest['data_files'].items():
            assert file_sha(Path(name)) == metadata['sha256'], name
        for key in ['prior_receipt', 'cpu_receipt', 'frozen_nr_module']:
            assert file_sha(Path(manifest[key])) == manifest[key + '_sha256'], key

    verify_inputs()
    module_path = 'scripts/nr3d_object_point_appearance.py'
    def crop_ast(path):
        tree = ast.parse(path.read_text())
        return ast.dump(next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                             and node.name == 'box_crop_mask'))
    assert crop_ast(root / module_path) == crop_ast(Path(manifest['frozen_nr_module']))
    prior = json.loads(Path(manifest['prior_receipt']).read_text())
    assert not prior['input_contract_pass'] and prior['empty_crop_slots'] == 0
    assert prior['slots_with_nonpositive_axes'] == 2
    cases = prior['violations']
    assert [case['row_id'] for case in cases] == [18112, 21602]
    prior_rows = {row['scan_id']: row for row in prior['rows']}
    cpu = json.loads(Path(manifest['cpu_receipt']).read_text())
    assert cpu['status'] == 'pass' and cpu['tests_passed'] == 7
    assert cpu['files'][module_path] == manifest['files'][module_path]
    os.chdir(str(source))
    sys.path.insert(0, str(source))
    import numpy as np
    import torch
    import scripts
    scripts.__path__ = [str(root / 'scripts')] + list(scripts.__path__)
    from src.joint_det_dataset import Joint3DDataset
    from scripts.nr3d_object_point_appearance import ObjectPointAppearanceResidual, box_crop_mask
    assert not torch.cuda.is_available()
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    class CaseInputs(Joint3DDataset):
        def _scene_graph_parse(self, annos):
            annos[:] = [annos[case['row_id']] for case in cases]
            super()._scene_graph_parse(annos)

    dataset = CaseInputs(dataset_dict={'sr3d': 1}, test_dataset='sr3d', split='train',
                         data_path='/root/autodl-tmp/DATA_ROOT/', use_color=True,
                         detect_intermediate=True, butd_cls=True, skip_missing_superpoints=True)
    dataset.augment = False
    addon = ObjectPointAppearanceResidual().eval()
    initial = {name: value.clone() for name, value in addon.state_dict().items()}
    records = []
    started = time.time()
    for index, case in enumerate(cases):
        item = dataset[index]
        assert item['scan_ids'] == case['scan_id']
        old = prior_rows[case['scan_id']]
        points = torch.from_numpy(item['point_clouds']).unsqueeze(0)
        boxes = torch.from_numpy(item['all_detected_boxes']).unsqueeze(0)
        valid = torch.from_numpy(item['all_detected_bbox_label_mask'].astype(bool)).unsqueeze(0)
        assert hashlib.sha256(points.numpy().tobytes()).hexdigest() == old['point_cloud_sha256']
        assert hashlib.sha256(boxes.numpy().tobytes()).hexdigest() == old['input_boxes_sha256']
        object_ids = valid[0].nonzero().flatten().tolist()
        assert object_ids == old['object_ids']
        captured = []
        handle = addon.point_encoder[0].register_forward_pre_hook(
            lambda module, args: captured.append(args[0].detach().clone()))
        with torch.no_grad():
            zero = addon(points, boxes, valid)
        handle.remove()
        assert torch.count_nonzero(zero) == 0 and len(captured) == len(object_ids)
        positive_slots = 0
        for position, object_id in enumerate(object_ids):
            box = boxes[0, object_id]
            crop = points[0, box_crop_mask(points[0, :, :3], box)]
            assert len(crop) == old['crop_point_counts'][position] > 0
            assert torch.equal(captured[position][:, 3:], crop[:, 3:])
            if object_id == case['object_id']:
                assert torch.equal(box, torch.tensor(case['box']))
                assert len(crop) == 1 and torch.count_nonzero(box[3:]) == 0
                assert torch.count_nonzero(captured[position][:, :3]) == 0
            else:
                assert (box[3:] > 0).all()
                assert torch.equal(captured[position][:, :3], (crop[:, :3] - box[:3]) / (box[3:] * .5))
                positive_slots += 1
        with torch.no_grad():
            addon.output.weight.copy_(torch.eye(288, 128) * .001)
        output = addon(points, boxes, valid)
        assert torch.isfinite(output).all() and torch.count_nonzero(output[~valid]) == 0
        exceptional = output[0, case['object_id']]
        assert exceptional.norm() > 0
        (exceptional * torch.randn_like(exceptional)).sum().backward()
        gradients = {name: float(parameter.grad.norm()) for name, parameter in addon.named_parameters()}
        assert all(torch.isfinite(parameter.grad).all() and parameter.grad.norm() > 0
                   for parameter in addon.parameters())
        addon.zero_grad(set_to_none=True)
        with torch.no_grad():
            addon.output.weight.zero_()
        records.append({'scan_id': case['scan_id'], 'row_id': case['row_id'],
                        'object_id': case['object_id'], 'valid_object_slots': len(object_ids),
                        'positive_slots_exact_original_normalization': positive_slots,
                        'exceptional_crop_points': 1, 'zero_axis_coordinates_are_zero': True,
                        'actual_crop_rgb_unchanged': True, 'point_cloud_sha256': old['point_cloud_sha256'],
                        'input_boxes_sha256': old['input_boxes_sha256'],
                        'exceptional_output_norm_fixed_perturbation': float(exceptional.detach().norm()),
                        'exceptional_only_loss_gradient_norms': gradients})
    assert all(torch.equal(value, initial[name]) for name, value in addon.state_dict().items())
    assert all(parameter.grad is None for parameter in addon.parameters())
    verify_inputs()
    result = {'schema': 'mcln-sr3d-zero-extent-input-probe-v1', 'status': 'pass',
              'training_rows': 2, 'scans': 2, 'observed_zero_extent_objects': 2,
              'valid_object_slots': sum(row['valid_object_slots'] for row in records),
              'appearance_cpu_forwards': 4, 'cpu_backwards': 2, 'optimizer_steps': 0,
              'native_model_forwards': 0, 'gpu_forwards': 0, 'formal_validation_rows': 0,
              'crop_function_ast_identical': True, 'zero_initial_outputs': True,
              'addon_state_restored': True, 'source_and_data_unchanged': True,
              'frozen_nr_pair_module_unchanged': True, 'prior_failed_receipt_unchanged': True,
              'instance_memberships_used_for_features': False, 'rows': records,
              'manifest_sha256': file_sha(options.manifest),
              'elapsed_seconds_excluding_dataset_init': time.time() - started}
    with (root / 'receipt.json').open('x') as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
    print('SR ZERO EXTENT PROBE COMPLETE', json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
