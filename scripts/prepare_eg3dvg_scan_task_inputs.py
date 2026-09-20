"""Prepare author ScanRefer training annotations and a shared corrected input path."""
import ast
import hashlib
import json
import os
from pathlib import Path
import pickle
import random
import shutil
import sys
import time


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    root = Path(__file__).resolve().parent
    acceptance = Path('/root/autodl-tmp/mcln_eg3dvg_acceptance_20260920_v1')
    prototype = Path('/root/autodl-tmp/mcln_eg3dvg_nr3d_task_read_20260920_v1')
    data = Path('/root/autodl-tmp/DATA_ROOT_mcln_meshsp')
    source = root / 'source'
    shutil.copytree(prototype / 'source', source, ignore=shutil.ignore_patterns('__pycache__'))
    file = source / 'src/joint_det_dataset.py'
    text = file.read_text()
    tree = ast.parse(text)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'Joint3DDataset')
    init = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '__init__')
    init_end = max(node.lineno for node in ast.walk(init) if hasattr(node, 'lineno'))
    lines = text.splitlines(keepends=True)
    # The accepted input adaptation caches the author's parsed annotations.
    start = next(i for i in range(init.lineno, init_end) if '# Cache produced' in lines[i])
    block = '''        assert test_dataset == 'scanrefer'
        if self.split == 'val':
            with open('/root/autodl-tmp/mcln_eg3dvg_acceptance_20260920_v1/annotations.pkl', 'rb') as file:
                self.annos = pickle.load(file)
        else:
            assert self.split == 'train' and dataset_dict == {'scanrefer': 1, 'scannet': 10}
            with open('SCAN_CACHE', 'rb') as file:
                self.annos = pickle.load(file)
            with open('/root/autodl-tmp/mcln_eg3dvg_nr3d_transfer_20260920_v1/train_input_cache/scannet_annotations.pkl', 'rb') as file:
                self.annos += pickle.load(file) * 10
'''.replace('SCAN_CACHE', str(root / 'scanrefer_annotations.pkl'))
    lines[start:init_end] = [block]
    text = ''.join(lines)
    rotations = "            all_det_pts = rot_z(all_det_pts, augmentations['theta_z'])\n            all_det_pts = rot_x(all_det_pts, augmentations['theta_x'])\n            all_det_pts = rot_y(all_det_pts, augmentations['theta_y'])\n"
    assert text.count(rotations) == 1
    text = text.replace(rotations, '')
    marker = "            all_det_pts += augmentations['shift']"
    assert text.count(marker) == 1
    text = text.replace(marker, rotations + marker)
    compile(text, str(file), 'exec')
    file.write_text(text)
    os.chdir(source)
    sys.path.insert(0, str(source))
    sys.path.insert(1, str(source / 'pointnet2'))
    import numpy as np
    import torch
    from src.joint_det_dataset import Joint3DDataset, unpickle_data, read_label_mapping

    random.seed(2027)
    np.random.seed(2027)
    torch.manual_seed(2027)
    loader = Joint3DDataset.__new__(Joint3DDataset)
    loader.split = 'train'
    loader.overfit = False
    loader.wo_obj_name = 'None'
    loader.data_path = str(data) + '/'
    loader.scans = list(unpickle_data(str(data / 'train_v3scans.pkl')))[0]
    loader.label_map = read_label_mapping('data/meta_data/scannetv2-labels.combined.tsv', label_from='raw_category', label_to='id')
    loader.label_mapclass = read_label_mapping('data/meta_data/scannetv2-labels.combined.tsv', label_from='raw_category', label_to='nyu40class')
    began = time.time()
    print('SCAN_TRAIN_PARSE_BEGIN', flush=True)
    annotations = loader.load_annos('scanrefer')
    with (root / 'scanrefer_annotations.pkl').open('xb') as f:
        pickle.dump(annotations, f, protocol=4)
    with (acceptance / 'annotations.pkl').open('rb') as f:
        validation = pickle.load(f)
    assert not {a['scan_id'] for a in annotations} & {a['scan_id'] for a in validation}
    del loader
    dataset = Joint3DDataset(dataset_dict={'scanrefer': 1, 'scannet': 10}, test_dataset='scanrefer',
                            split='train', data_path=str(data) + '/', use_color=True,
                            detect_intermediate=True, butd=True, augment_det=True)
    assert dataset.butd and not dataset.butd_cls and not dataset.butd_gt and dataset.augment
    assert len(dataset) == len(annotations) + 11990
    indices = [0, len(annotations)//7, len(annotations)//3, len(annotations)//2,
               len(annotations)-2, len(annotations)-1, len(annotations), len(dataset)-1]
    samples = []
    for index in indices:
        item = dataset[index]
        assert item['point_clouds'].shape == (50000, 6)
        assert np.isfinite(item['all_detected_boxes']).all()
        samples.append({'index': index, 'scan_id': item['scan_ids'],
                        'dataset': dataset.annos[index]['dataset'],
                        'point_sha256': hashlib.sha256(item['point_clouds'].tobytes()).hexdigest()})
    checkpoint = torch.load(acceptance / 'official_scanrefer.pth', map_location='cpu')
    lrs = [group['lr'] for group in checkpoint['optimizer']['param_groups']]
    report = {'status': 'complete', 'scanrefer_rows': len(annotations), 'fit_rows': len(dataset),
              'steps_per_epoch': (len(dataset)+7)//8, 'scannet_rows': 1199, 'scannet_repeat': 10,
              'train_validation_scenes_disjoint': True, 'samples': samples,
              'annotations_sha256': sha(root / 'scanrefer_annotations.pkl'),
              'dataset_sha256': sha(file), 'author_checkpoint_epoch': checkpoint['epoch'],
              'author_saved_optimizer_lrs': lrs, 'seconds': time.time()-began,
              'detection_transform_order': 'flip then Z/X/Y rotate then shift then scale',
              'model_forwards': 0, 'optimizer_steps': 0}
    (root / 'input_receipt.json').write_text(json.dumps(report, indent=2))
    print('SCAN_INPUT_READY ' + json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
