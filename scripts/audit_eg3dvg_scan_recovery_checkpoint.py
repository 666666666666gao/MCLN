"""Inspect the first saved ScanRefer state on CPU, without selecting or resuming it."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import time

import torch


def main():
    torch.set_num_threads(1)
    parser = argparse.ArgumentParser()
    parser.add_argument('--spec', type=Path, required=True)
    parser.add_argument('--controller-pid', type=int, required=True)
    args = parser.parse_args()
    spec = json.loads(args.spec.read_text())
    assert spec['task_read']
    root = args.spec.parent.parent
    path = Path(spec['state_root']) / 'latest.pth'
    while not path.exists():
        assert not (root / 'controller.exit').exists(), 'Controller ended before saving a state'
        command = Path('/proc/%d/cmdline' % args.controller_pid).read_bytes().split(b'\0')
        assert str(root / 'controller.py').encode() in command
        print('WAIT_RECOVERY_CHECKPOINT ' + str(path), flush=True)
        time.sleep(300)

    started = time.time()
    # One open file identifies the saved artifact even if latest.pth is later replaced.
    with path.open('rb') as stream:
        checkpoint = torch.load(stream, map_location='cpu')
        stream.seek(0)
        digest = hashlib.sha256()
        size = 0
        for block in iter(lambda: stream.read(8388608), b''):
            digest.update(block)
            size += len(block)
    parent = torch.load(spec['checkpoint'], map_location='cpu')
    key = 'module.decoder.5.task_queries'
    assert set(checkpoint['model']) == set(parent['model']) | {key}
    assert checkpoint['model'][key].shape == (2, 288, 288)
    assert all(torch.count_nonzero(v) > 0 for v in checkpoint['model'][key])
    assert all(checkpoint['model'][k].shape == v.shape and
               checkpoint['model'][k].dtype == v.dtype for k, v in parent['model'].items())
    assert all(torch.isfinite(v).all() for v in checkpoint['model'].values())
    assert checkpoint['spec_sha256'] == hashlib.sha256(args.spec.read_bytes()).hexdigest()
    assert checkpoint['parent_checkpoint_sha256'] == spec['checkpoint_sha256']
    assert checkpoint['epoch'] == 1 and checkpoint['step'] >= 512
    assert checkpoint['rows'] == min(checkpoint['step'] * 8, spec['fit_rows'])
    assert len(checkpoint['optimizer']['state']) > 0
    assert len(checkpoint['optimizer']['param_groups']) == 3
    for state in checkpoint['optimizer']['state'].values():
        for value in state.values():
            if torch.is_tensor(value):
                assert torch.isfinite(value).all()
    for field in ['random_state', 'numpy_state', 'torch_rng_state', 'cuda_rng_states']:
        assert field in checkpoint
    report = {
        'status': 'load_verified', 'path': str(path), 'bytes': size,
        'sha256': digest.hexdigest(), 'epoch': checkpoint['epoch'],
        'step': checkpoint['step'], 'rows': checkpoint['rows'],
        'model_states': len(checkpoint['model']),
        'optimizer_states': len(checkpoint['optimizer']['state']),
        'task_matrix_norms': [float(v.norm()) for v in checkpoint['model'][key]],
        'model_and_optimizer_finite': True, 'rng_saved': True,
        'seconds': time.time() - started,
        'time_cst': datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
        'model_forwards': 0, 'optimizer_steps_executed': 0, 'checkpoint_selection': False,
        'claim_boundary': 'CPU deserialization and saved payload checks only; no resumed training step or accuracy evaluation.'}
    target = args.spec.parent / ('recovery_checkpoint_%d.json' % checkpoint['step'])
    with target.open('x') as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
