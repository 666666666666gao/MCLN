"""CPU-only object-appearance probe on the existing sixteen audited inputs."""

import argparse
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
        for name, value in manifest['data_files'].items():
            assert file_sha(Path(name)) == value['sha256'], name
        assert file_sha(Path(manifest['prior_receipt'])) == manifest['prior_receipt_sha256']

    verify_inputs()
    os.chdir(str(source))
    sys.path.insert(0, str(source))
    import numpy as np
    import torch
    import scripts
    scripts.__path__ = [str(root / 'scripts')] + list(scripts.__path__)
    from src.joint_det_dataset import Joint3DDataset
    from scripts.nr3d_object_point_appearance import ObjectPointAppearanceResidual
    from scripts.run_nr3d_view_pair_role import read_train_rows

    assert not torch.cuda.is_available()
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    rows = read_train_rows(Path('/root/autodl-tmp/DATA_ROOT'))
    row_ids = manifest['fit_row_ids']
    prior = json.loads(Path(manifest['prior_receipt']).read_text())
    assert row_ids == [row['fit_row_id'] for row in prior['rows']]
    prior_rows = {row['fit_row_id']: row for row in prior['rows']}

    class FixedFit(Joint3DDataset):
        def _scene_graph_parse(self, annos):
            annos[:] = [annos[i] for i in row_ids]
            super()._scene_graph_parse(annos)

    dataset = FixedFit(dataset_dict={'nr3d': 1}, test_dataset='nr3d', split='train',
                       data_path='/root/autodl-tmp/DATA_ROOT/', use_color=True, detect_intermediate=True,
                       butd_cls=True, skip_missing_superpoints=True)
    dataset.augment = False
    addon = ObjectPointAppearanceResidual().eval()
    initial = {name: value.clone() for name, value in addon.state_dict().items()}
    results = []
    started = time.time()
    with torch.no_grad():
        for index, row_id in enumerate(row_ids):
            item = dataset[index]
            assert item['scan_ids'] == rows[row_id]['scan_id']
            points = torch.from_numpy(item['point_clouds']).unsqueeze(0)
            boxes = torch.from_numpy(item['all_detected_boxes']).unsqueeze(0)
            valid = torch.from_numpy(item['all_detected_bbox_label_mask'].astype(bool)).unsqueeze(0)
            digest = hashlib.sha256(points.numpy().tobytes()).hexdigest()
            assert digest == prior_rows[row_id]['input_point_cloud_sha256']
            objects = prior_rows[row_id]['objects']
            assert valid[0].nonzero().flatten().tolist() == [obj['object_id'] for obj in objects]
            crop_sizes = []
            for obj in objects:
                box = boxes[0, obj['object_id']]
                assert np.array_equal(box.numpy(), np.asarray(obj['box'], dtype=np.float32))
                inside = ((points[0, :, :3] - box[:3]).abs() <= box[3:] * .5).all(dim=-1)
                count = int(inside.sum())
                assert count == obj['crop_points'] and count > 0
                crop_sizes.append(count)
            zero = addon(points, boxes, valid)
            assert torch.count_nonzero(zero) == 0
            addon.output.weight.copy_(torch.eye(288, 128) * .001)
            changed = addon(points, boxes, valid)
            assert torch.isfinite(changed).all() and torch.count_nonzero(changed[~valid]) == 0
            norms = changed[valid].norm(dim=-1)
            assert (norms > 0).all()
            addon.output.weight.zero_()
            record = {'fit_row_id': row_id, 'scan_id': item['scan_ids'], 'input_point_sha256': digest,
                      'valid_object_slots': int(valid.sum()), 'crop_point_counts': crop_sizes,
                      'appearance_norm_min_after_fixed_perturbation': float(norms.min()),
                      'appearance_norm_max_after_fixed_perturbation': float(norms.max())}
            results.append(record)
            print('OBJECT APPEARANCE INPUT', json.dumps(record), flush=True)
    assert all(torch.equal(value, initial[name]) for name, value in addon.state_dict().items())
    assert all(parameter.grad is None for parameter in addon.parameters())
    assert sum(row['valid_object_slots'] for row in results) == 683
    verify_inputs()
    result = {'schema': 'mcln-object-appearance-real-input-v1', 'status': 'complete',
              'fit_rows': 16, 'fit_scenes': 16, 'valid_object_slots': 683, 'appearance_cpu_forwards': 32,
              'native_model_forwards': 0, 'gpu_forwards': 0, 'optimizer_steps': 0,
              'holdout_rows': 0, 'formal_rows': 0, 'quality_screen_executed': False,
              'same_audited_input_points_boxes_and_crops': True, 'zero_initial_output': True,
              'padded_slots_unchanged': True, 'addon_state_restored': True,
              'source_and_data_unchanged': True, 'instance_labels_used_for_features': False,
              'rows': results, 'manifest_sha256': file_sha(options.manifest),
              'elapsed_seconds_excluding_dataset_init': time.time() - started}
    with (root / 'receipt.json').open('x') as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
    print('OBJECT APPEARANCE INPUT COMPLETE', json.dumps({key: value for key, value in result.items() if key != 'rows'}), flush=True)


if __name__ == '__main__':
    main()
