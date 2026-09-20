"""Freeze ScanRefer campaign from completed input checks and launch one controller."""
import hashlib
import json
import os
from pathlib import Path
import pickle
import subprocess


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda: f.read(8388608), b''):
            h.update(part)
    return h.hexdigest()


def main():
    root = Path(__file__).resolve().parent
    assert (root / 'input_receipt.json').is_file()
    import numpy as np
    inputs = json.loads((root / 'input_receipt.json').read_text())
    assert inputs['status'] == 'complete' and inputs['train_validation_scenes_disjoint']
    subprocess.check_call([os.sys.executable, '-u', str(root / 'check_alignment.py')])
    reference = Path('/root/autodl-tmp/mcln_eg3dvg_acceptance_20260920_v1')
    old = json.loads((reference / 'spec.json').read_text())
    source = root / 'source'
    source_files = {name: sha(source / name) for name in old['source_files']}
    source_files['models/eg3dvg_task_read.py'] = sha(source / 'models/eg3dvg_task_read.py')
    # Use the published checkpoint's late-training LR scale, fresh AdamW for both arms.
    lrs = inputs['author_saved_optimizer_lrs']
    assert len(lrs) == 3 and all(value > 0 for value in lrs)
    with (root / 'scanrefer_annotations.pkl').open('rb') as f:
        annotations = pickle.load(f)
    detection = Path('/root/autodl-tmp/mcln_eg3dvg_nr3d_transfer_20260920_v1/train_input_cache/scannet_annotations.pkl')
    with detection.open('rb') as f:
        all_annotations = annotations + pickle.load(f) * 10
    assert len(all_annotations) == inputs['fit_rows']
    validation = json.loads((reference / 'annotation_manifest.json').read_text())
    assert not {a['scan_id'] for a in all_annotations} & {a['scan_id'] for a in validation}
    data = Path(old['data_root'])
    input_paths = [root / 'scanrefer_annotations.pkl', detection, data / 'train_v3scans.pkl',
                   data / 'ScanRefer/ScanRefer_filtered_train.json', data / 'ScanRefer/ScanRefer_filtered_train.txt']
    for scene in sorted({a['scan_id'] for a in all_annotations}):
        input_paths += [data / 'superpoints/train' / (scene + '_superpoint.pth'),
                        data / 'group_free_pred_bboxes/group_free_pred_bboxes_train' / (scene + '.npy')]
    order_paths = []
    rng = np.random.RandomState(2027)
    for epoch in range(1, 4):
        path = root / ('training_order_%02d.npy' % epoch)
        np.save(str(path), rng.permutation(inputs['fit_rows']).astype(np.int64))
        order_paths.append(str(path))
        input_paths.append(path)
    hashes = {str(path): sha(path) for path in input_paths}
    first = np.load(order_paths[0]).tolist()
    for arm in ['task', 'native']:
        (root / arm).mkdir()
        states = Path('/root/mcln_eg3dvg_scan_%s_states_20260921_v1' % arm)
        states.mkdir()
        spec = dict(source=str(source), source_files=source_files, checkpoint=old['checkpoint'],
                    checkpoint_sha256=old['checkpoint_sha256'], data_root=old['data_root'],
                    input_hashes=hashes, seed=2027, batch_size=8, fit_rows=inputs['fit_rows'],
                    fit_updates=inputs['steps_per_epoch'], max_epochs=3, order_paths=order_paths,
                    preflight_indices=first[:16], lr=lrs[0], lr_backbone=lrs[1], text_encoder_lr=lrs[2],
                    schedule='constant published epoch69 LR for this bounded continuation',
                    state_root=str(states), task_read=arm == 'task',
                    trainer_sha256=sha(root / 'train.py'), environment_sha256=old['base_environment_sha256'],
                    clip_norm=.1, weight_decay=.0005, validation_used_for_gradient=False)
        (root / arm / 'spec.json').write_text(json.dumps(spec, indent=2))
    campaign = dict(reference_root=str(reference), max_epochs=3, seed=2027,
                    baseline_hits=[5542,4952], promotion_min_hits=[5543,4953], formal_rows=9508,
                    primary_mode='bbs', mask_metric_gate=False, controller_sha256=sha(root/'controller.py'),
                    wait_for='/root/autodl-tmp/mcln_eg3dvg_sr3d_transfer_20260920_v1',
                    nr3d_sr3d_training_authorized_before_scan_gate=False,
                    stop_rule='first complete paired epoch whose task model beats both native pretrained REC thresholds',
                    scripts={name:sha(root/name) for name in ['train.py','initial.py','evaluate.py','audit.py']})
    (root / 'campaign.json').write_text(json.dumps(campaign, indent=2))
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='0')
    with (root/'controller.log').open('xb') as log:
        child = subprocess.Popen([old['runtime'], '-u', str(root/'controller.py')],
                                  stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True)
    report = dict(status='launched', pid=child.pid, campaign_sha256=sha(root/'campaign.json'),
                  task_spec_sha256=sha(root/'task/spec.json'), native_spec_sha256=sha(root/'native/spec.json'),
                  fit_rows=inputs['fit_rows'], steps_per_epoch=inputs['steps_per_epoch'], lrs=lrs,
                  optimizer_steps_observed=0)
    (root/'launch.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
