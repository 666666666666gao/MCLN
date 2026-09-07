"""Export four deterministic training-scene inputs without model inference or GT inputs."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--training-labels', action='store_true')
    parser.add_argument('--reference-fixtures', type=Path)
    option = parser.parse_args()
    started = time.time()
    manifest = json.loads(option.manifest.read_bytes())
    source = Path(manifest['model_source'])
    assert sha(source / 'appearance_source_manifest.json') == manifest['source_manifest_sha256']
    for name, digest in json.loads((source / 'appearance_source_manifest.json').read_bytes())['files'].items():
        assert sha(source / name) == digest, name
    assert manifest['data_root'] == '/root/autodl-tmp/DATA_ROOT_mcln_meshsp/'
    assert sha(manifest['split_protocol']) == manifest['split_protocol_sha256']
    split = json.loads(Path(manifest['split_protocol']).read_bytes())
    partitions = split['row_ids']
    assert len(partitions['fit']) == 29778 and len(partitions['holdout']) == 6887
    os.chdir(str(source))
    sys.path.insert(0, str(source))
    import numpy as np
    import torch
    from src.joint_det_dataset import Joint3DDataset
    from scripts.scanrefer_data_contract import verify_scanrefer_superpoints
    random.seed(2027)
    np.random.seed(2027)
    torch.manual_seed(2027)
    verify_scanrefer_superpoints(manifest['data_root'], 'train', manifest['superpoint_files']['train'])

    class Dataset(Joint3DDataset):
        def _scene_graph_parse(self, annos):
            assert len(annos) == 36665
            actual = {'fit': [], 'holdout': []}
            for index, row in enumerate(annos):
                row['_local_training_id'] = index
                code = (manifest['split_salt'] + '\0' + row['scan_id'].split('_')[0]).encode()
                fold = int(hashlib.sha256(code).hexdigest()[:8], 16) % 5
                actual['holdout' if fold == 0 else 'fit'].append(index)
            assert actual == partitions
            selected, scenes = [], set()
            for row_id in partitions['fit']:
                physical = annos[row_id]['scan_id'].split('_')[0]
                if physical not in scenes:
                    selected.append(row_id)
                    scenes.add(physical)
                if len(selected) == 4:
                    break
            self.fixture_row_ids = selected
            # Full annotations remain for native distractor/unique counts; only four rows are fetched.
            super()._scene_graph_parse([annos[i] for i in selected])

    print('PVG_FIXTURE_DATASET_LOADING', flush=True)
    dataset = Dataset(dataset_dict={'scanrefer': 1}, test_dataset='scanrefer', split='train',
        data_path=manifest['data_root'], use_color=True, use_height=False, use_multiview=False,
        detect_intermediate=True, butd=True, butd_gt=False, butd_cls=False,
        augment_det=False, skip_missing_superpoints=True)
    dataset.augment = False
    assert [row['_local_training_id'] for row in dataset.annos] == list(range(36665))
    selected = dataset.fixture_row_ids
    assert len(selected) == 4 and not set(selected).intersection(partitions['holdout'])
    option.output.mkdir()
    records = []
    if option.training_labels:
        assert option.reference_fixtures is not None
        reference = json.loads((option.reference_fixtures / 'receipt.json').read_bytes())
        assert [r['training_row_id'] for r in reference['rows']] == selected
    for row_id in selected:
        sample = dataset[row_id]
        inputs = {
            'point_clouds': torch.from_numpy(sample['point_clouds']).float().unsqueeze(0),
            'text': [sample['utterances']],
            'det_boxes': torch.from_numpy(sample['all_detected_boxes']).unsqueeze(0),
            'det_bbox_label_mask': torch.from_numpy(sample['all_detected_bbox_label_mask']).unsqueeze(0),
            'det_class_ids': torch.from_numpy(sample['all_detected_class_ids']).unsqueeze(0),
            'superpoint': sample['superpoint'].unsqueeze(0),
        }
        assert inputs['point_clouds'].shape == (1, 50000, 6)
        assert inputs['superpoint'].shape == (1, 50000)
        assert torch.isfinite(inputs['point_clouds']).all()
        path = option.output / ('row_%05d.pt' % row_id)
        if option.training_labels:
            original = reference['rows'][len(records)]
            assert inputs['text'][0] == original['text']
            for key, value in inputs.items():
                if torch.is_tensor(value):
                    assert hashlib.sha256(value.numpy().tobytes()).hexdigest() == original['tensor_sha256'][key], key
            label_keys = ['box_label_mask', 'center_label', 'sem_cls_label', 'size_gts', 'gt_masks',
                'positive_map', 'modify_positive_map', 'pron_positive_map', 'other_entity_map',
                'rel_positive_map', 'auxi_entity_positive_map', 'auxi_box', 'point_instance_label',
                'language_dataset', 'is_view_dep', 'is_hard', 'is_unique']
            labels = {key: sample[key] for key in label_keys}
            # The source mask is exactly binary; boolean storage preserves every label and saves disk.
            assert np.isin(labels['gt_masks'], [0, 1]).all()
            labels['gt_masks'] = labels['gt_masks'].astype(np.bool_)
            torch.save({'inputs': inputs, 'labels': labels}, str(path))
        else:
            torch.save(inputs, str(path))
        row = dataset.annos[row_id]
        records.append({'training_row_id': row_id, 'scan_id': row['scan_id'], 'target_id': int(row['target_id']),
            'text': inputs['text'][0], 'file': path.name, 'file_sha256': sha(path),
            'input_keys': sorted(inputs),
            'tensor_sha256': {k: hashlib.sha256(v.numpy().tobytes()).hexdigest() for k,v in inputs.items() if torch.is_tensor(v)},
            'detected_boxes': int(inputs['det_bbox_label_mask'].sum()),
            'superpoint_count': int(inputs['superpoint'].max())+1})
        print('PVG_FIXTURE_ROW '+json.dumps(records[-1]), flush=True)
    receipt = {'status':'pass','time_cst':datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
        'selection':'first four distinct physical scenes in existing ordered fit row IDs', 'rows':records,
        'source_manifest_sha256':manifest['source_manifest_sha256'], 'data_root':manifest['data_root'],
        'split_protocol_sha256':manifest['split_protocol_sha256'], 'manifest_sha256':sha(option.manifest),
        'model_forwards':0,'optimizer_steps':0,'formal_rows':0,'training_fixture_rows':4,
        'gt_model_inputs':False,'separate_training_labels':option.training_labels,
        'augmentation':False,'annotations_retained_for_counts':36665,'annotations_parsed':4,
        'script_sha256':sha(__file__),'elapsed_seconds':time.time()-started}
    (option.output/'receipt.json').write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n')
    print('PVG_FIXTURES_COMPLETE '+json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()
