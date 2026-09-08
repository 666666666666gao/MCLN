"""Compare detected-box augmentation with the affine transform of real input points."""
import argparse
import ast
import copy
import datetime
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import textwrap
import time
import types


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--spec', type=Path, required=True)
    args = parser.parse_args()
    spec = json.loads(args.spec.read_bytes())
    root = args.spec.parent
    started = time.time()
    manifest = json.loads(Path(spec['input_manifest']).read_bytes())
    native = Path(manifest['model_source'])
    source = native / 'src/joint_det_dataset.py'
    assert sha(source) == spec['native_dataset_sha256']
    assert sha(root / 'fixed_dataset.py') == spec['fixed_dataset_sha256']
    assert sha(manifest['split_protocol']) == manifest['split_protocol_sha256']
    partitions = json.loads(Path(manifest['split_protocol']).read_bytes())['row_ids']
    import numpy as np
    import torch
    assert not torch.cuda.is_initialized() and os.environ['CUDA_VISIBLE_DEVICES'] == ''
    torch.set_num_threads(1)
    os.chdir(str(native))
    sys.path.insert(0, str(native))
    import src.joint_det_dataset as module
    assert Path(module.__file__).resolve() == source
    namespace = dict(vars(module))
    exec(compile((root / 'fixed_detected_objects.py').read_bytes(), 'fixed_detected_objects', 'exec'), namespace)
    fixed = namespace['_get_detected_objects']
    selected = []

    class Probe(module.Joint3DDataset):
        def _scene_graph_parse(self, annos):
            assert len(annos) == 36665
            ordered = sorted(partitions['fit'], key=lambda i: hashlib.sha256(('scanrefer_det_affine_v1\0' + str(i)).encode()).hexdigest())
            seen = set()
            for i in ordered:
                physical = annos[i]['scan_id'].split('_')[0]
                if physical in seen:
                    continue
                seen.add(physical)
                selected.append(dict(training_row_id=i, scan_id=annos[i]['scan_id'], target_id=annos[i]['target_id']))
                if len(selected) == spec['rows']:
                    break
            annos[:] = [annos[r['training_row_id']] for r in selected]
            super()._scene_graph_parse(annos)

        def _augment(self, pc, color, rotate):
            self.before = pc[:, :3].copy()
            result = super()._augment(pc, color, rotate)
            self.after = result[0][:, :3].copy()
            self.transform = copy.deepcopy(result[2])
            return result

    dataset = Probe(dataset_dict={'scanrefer': 1}, test_dataset='scanrefer', split='train',
        data_path=manifest['data_root'], use_color=True, use_height=False, use_multiview=False,
        detect_intermediate=True, butd=True, butd_cls=False, butd_gt=False,
        augment_det=False, skip_missing_superpoints=True)
    original = copy.deepcopy(dataset.annos)
    print('SELECTED ' + json.dumps(selected), flush=True)

    def seed(value):
        random.seed(value)
        np.random.seed(value)
        torch.manual_seed(value)

    rows = []
    for index, selection in enumerate(selected):
        dataset.augment = True
        dataset.annos = copy.deepcopy(original)
        seed(spec['seed'] + index)
        old = dataset[index]
        t = dataset.transform
        rng_old = np.random.get_state()
        # Recover the actual global affine map from observed point pairs after removing
        # the independent per-point noise. This does not repeat the box transform code.
        x = np.concatenate([dataset.before.astype(np.float64), np.ones((len(dataset.before), 1))], axis=1)
        y = dataset.after.astype(np.float64) - np.asarray(t['noise']) * t['scale']
        affine = np.linalg.lstsq(x[::13], y[::13], rcond=None)[0]
        affine_error = float(np.abs(x @ affine - y).max())
        assert affine_error < 3e-5, affine_error
        det_path = Path(manifest['data_root']) / 'group_free_pred_bboxes/group_free_pred_bboxes_train' / (selection['scan_id'] + '.npy')
        raw = np.load(str(det_path), allow_pickle=True).item()['box']
        raw = np.asarray(raw)
        bits = np.array([[a, b, c] for a in [0, 1] for b in [0, 1] for c in [0, 1]])
        corners = raw[:, None, :3] * (1-bits) + raw[:, None, 3:] * bits
        transformed = np.concatenate([corners, np.ones((*corners.shape[:2], 1))], axis=-1) @ affine
        low, high = transformed.min(1), transformed.max(1)
        reference = np.concatenate([(low+high)/2, high-low], axis=1)
        n = len(raw)
        assert int(old['all_detected_bbox_label_mask'].sum()) == n
        old_error = np.abs(old['all_detected_boxes'][:n] - reference)
        dataset._get_detected_objects = types.MethodType(fixed, dataset)
        dataset.annos = copy.deepcopy(original)
        seed(spec['seed'] + index)
        new = dataset[index]
        rng_new = np.random.get_state()
        assert rng_old[0] == rng_new[0] and np.array_equal(rng_old[1], rng_new[1]) and rng_old[2:] == rng_new[2:]
        for key in old:
            if key == 'all_detected_boxes':
                continue
            left, right = old[key], new[key]
            same = torch.equal(left, right) if torch.is_tensor(left) else np.array_equal(left, right) if isinstance(left, np.ndarray) else left == right
            assert same, key
        new_error = np.abs(new['all_detected_boxes'][:n] - reference)
        assert float(new_error.max()) < 3e-5, float(new_error.max())
        centers = np.linalg.norm(old['all_detected_boxes'][:n, :3] - reference[:, :3], axis=1)
        row = dict(**selection, seed=spec['seed']+index, boxes=n,
            yz_flip=bool(t.get('yz_flip',False)), xz_flip=bool(t.get('xz_flip',False)),
            angles=[float(t[k]) for k in ['theta_z','theta_x','theta_y']],
            affine_point_max_error=affine_error, old_box_max_error=float(old_error.max()),
            fixed_box_max_error=float(new_error.max()), old_center_max_m=float(centers.max()),
            old_centers_above_1cm=int((centers > .01).sum()), det_file_sha256=sha(det_path),
            unchanged_point_gt_mask_text_class_and_rng=True)
        rows.append(row)
        del dataset._get_detected_objects
        # Evaluation with augmentation disabled must be byte-identical.
        dataset.augment = False
        seed(spec['seed']+index)
        plain_old = dataset[index]['all_detected_boxes']
        dataset._get_detected_objects = types.MethodType(fixed, dataset)
        seed(spec['seed']+index)
        plain_new = dataset[index]['all_detected_boxes']
        assert np.array_equal(plain_old, plain_new)
        del dataset._get_detected_objects
        print('ROW ' + json.dumps(row), flush=True)
    assert not torch.cuda.is_initialized()
    result = dict(status='pass', time_cst=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
        elapsed_seconds=time.time()-started, rows=len(rows), boxes=sum(r['boxes'] for r in rows),
        rows_with_center_error_above_1cm=sum(r['old_centers_above_1cm'] > 0 for r in rows),
        old_centers_above_1cm=sum(r['old_centers_above_1cm'] for r in rows),
        old_max_center_error_m=max(r['old_center_max_m'] for r in rows),
        fixed_max_box_error=max(r['fixed_box_max_error'] for r in rows),
        point_gt_mask_text_class_and_rng_unchanged=True, augmentation_off_identical=True,
        formal_rows=0, optimizer_steps=0, weights_loaded=0, model_forwards=0,
        native_dataset_sha256=sha(source), fixed_dataset_sha256=sha(root/'fixed_dataset.py'),
        script_sha256=sha(__file__), spec_sha256=sha(args.spec))
    (root/'rows.json').write_text(json.dumps(rows,indent=2)+'\n')
    result['rows_sha256']=sha(root/'rows.json')
    (root/'receipt.json').write_text(json.dumps(result,indent=2)+'\n')
    print('AUDIT_COMPLETE '+json.dumps(result),flush=True)


if __name__ == '__main__':
    main()
