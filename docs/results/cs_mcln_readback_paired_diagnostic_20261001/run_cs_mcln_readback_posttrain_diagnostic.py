"""Run the fixed-prediction R diagnosis after its formal 21-epoch job exits."""

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--training-pid', type=int, required=True)
    parser.add_argument('--not-before', type=datetime.fromisoformat, required=True)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--best-dir', type=Path, required=True)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--diagnostic', type=Path, required=True)
    parser.add_argument('--parent', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, required=True)
    opt = parser.parse_args()
    assert opt.not_before.tzinfo is not None
    directory = opt.run_dir / 'readback_diagnostic'
    assert not directory.exists()
    directory.mkdir()
    print('WAIT_UNTIL', opt.not_before.isoformat(), flush=True)
    time.sleep(max(0, opt.not_before.timestamp() - time.time()))
    while True:
        process = subprocess.run(
            ['ps', '-p', str(opt.training_pid), '-o', 'args='],
            capture_output=True, text=True,
        )
        if process.returncode == 1:
            break
        process.check_returncode()
        assert 'scripts/train_cs_mcln_scanrefer.py --arm cs_readback --mode train' in process.stdout
        print('WAIT_TRAINING_EXIT', datetime.now().astimezone().isoformat(), flush=True)
        time.sleep(300)
    assert int(Path(str(opt.run_dir) + '.exit.txt').read_text().strip()) == 0
    log = Path(str(opt.run_dir) + '.log').read_text()
    epochs = []
    for number in range(1, 22):
        assert ('EPOCH cs_readback {"epoch": %d,' % number) in log
        epoch = json.loads((opt.run_dir / ('epoch_%d.json' % number)).read_text())
        assert epoch['epoch'] == number and epoch['steps'] == 4055
        assert epoch['samples'] == 48655 and epoch['validation']['samples'] == 9508
        epochs.append(epoch)
    expected_best = max(epochs, key=lambda epoch: (
        min(epoch['validation']['hits025'] / 5572.0,
            epoch['validation']['hits050'] / 4797.0),
        epoch['validation']['hits025'] + epoch['validation']['hits050'],
    ))
    best = json.loads((opt.best_dir / 'best.json').read_text())
    assert best['arm'] == 'cs_readback' and best['epoch'] == expected_best['epoch']
    assert best['metrics'] == expected_best['validation']
    gpu = subprocess.run([
        'nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits',
    ], check=True, capture_output=True, text=True)
    assert not gpu.stdout.strip(), gpu.stdout
    environment = os.environ.copy()
    environment['PYTHONPATH'] = str(opt.source) + ':' + str(opt.base)
    started = datetime.now().astimezone().isoformat()
    (directory / 'started.json').write_text(json.dumps({
        'started_at': started, 'best': best,
        'phase_order': ['first_12_validation_expressions', 'full_9508_validation'],
        'optimizer_steps': 0,
    }, indent=2) + '\n')
    phases = []
    for name, limit in (('sanity_first12', 12), ('full_validation', None)):
        output = directory / (name + '.json')
        command = [
            sys.executable, '-u', str(opt.diagnostic),
            '--parent', str(opt.parent), '--checkpoint', str(opt.best_dir / 'best.pth'),
            '--data-root', str(opt.data_root), '--output', str(output), '--batch-size', '12',
        ]
        if limit is not None:
            command.extend(['--limit-expressions', str(limit)])
        phase_started = datetime.now().astimezone().isoformat()
        print('START_PHASE', name, phase_started, flush=True)
        with (directory / (name + '.log')).open('w') as stream:
            completed = subprocess.run(
                command, cwd=str(opt.base), env=environment,
                stdout=stream, stderr=subprocess.STDOUT,
            )
        phase = {
            'name': name, 'started_at': phase_started,
            'finished_at': datetime.now().astimezone().isoformat(),
            'returncode': completed.returncode, 'output': str(output),
        }
        phases.append(phase)
        (directory / 'execution.json').write_text(json.dumps({
            'started_at': started, 'best': best, 'phases': phases,
        }, indent=2) + '\n')
        assert completed.returncode == 0, phase
        result = json.loads(output.read_text())
        assert result['checkpoint_epoch'] == best['epoch']
        assert result['checkpoint_formal_metrics'] == best['metrics']
        assert result['sample_count'] == (12 if limit is not None else 9508)
        print('COMPLETE_PHASE', name, json.dumps(result['summary']), flush=True)


if __name__ == '__main__':
    main()
