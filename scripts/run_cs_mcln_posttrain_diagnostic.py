"""Run the fixed CS support panel after the formal 21-epoch job has exited."""

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
    parser.add_argument('--diagnostic', type=Path, required=True)
    parser.add_argument('--parent', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, required=True)
    opt = parser.parse_args()
    assert opt.not_before.tzinfo is not None
    output_dir = opt.run_dir / 'diagnostics'
    output = output_dir / 'm3_fixed_train_panel.json'
    receipt = output_dir / 'posttrain_diagnostic_execution.json'
    assert not output.exists() and not receipt.exists()
    print('WAIT_UNTIL', opt.not_before.isoformat(), flush=True)
    time.sleep(max(0, opt.not_before.timestamp() - time.time()))
    while True:
        process = subprocess.run(['ps', '-p', str(opt.training_pid), '-o', 'args='],
                                 capture_output=True, text=True)
        if process.returncode == 1:
            break
        process.check_returncode()
        assert 'scripts/train_cs_mcln_scanrefer.py' in process.stdout
        assert '--arm cs' in process.stdout
        print('WAIT_TRAINING_EXIT', datetime.now().astimezone().isoformat(), flush=True)
        time.sleep(300)

    log = Path(str(opt.run_dir) + '.log').read_text()
    epochs = []
    for number in range(1, 22):
        assert ('EPOCH cs {"epoch": %d,' % number) in log
        result = json.loads((opt.run_dir / ('epoch_%d.json' % number)).read_text())
        assert result['epoch'] == number and result['steps'] == 4055
        assert result['samples'] == 48655 and result['validation']['samples'] == 9508
        epochs.append(result)
    expected_best = max(epochs, key=lambda item: (
        min(item['validation']['hits025'] / 5572.0,
            item['validation']['hits050'] / 4797.0),
        item['validation']['hits025'] + item['validation']['hits050'],
    ))
    best = json.loads((opt.best_dir / 'best.json').read_text())
    assert best['arm'] == 'cs' and best['epoch'] == expected_best['epoch']
    assert best['metrics'] == expected_best['validation']

    import torch
    saved = torch.load(str(opt.best_dir / 'best.pth'), map_location='cpu')
    assert saved['epoch'] == best['epoch'] and saved['metrics'] == best['metrics']
    assert saved['arm'] == 'cs' and saved['seed'] == 2027 and saved['batch_size'] == 12
    del saved
    latest = torch.load(str(opt.run_dir / 'latest.pth'), map_location='cpu')
    assert latest['epoch'] == 21 and latest['arm'] == 'cs'
    del latest
    gpu = subprocess.run([
        'nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits',
    ], check=True, capture_output=True, text=True)
    assert not gpu.stdout.strip(), gpu.stdout

    started = datetime.now().astimezone().isoformat()
    print('START_DIAGNOSTIC', started, 'BEST_EPOCH', best['epoch'], flush=True)
    environment = os.environ.copy()
    environment['PYTHONPATH'] = str(opt.source)
    with (output_dir / 'm3_fixed_train_panel.log').open('w') as stream:
        completed = subprocess.run([
            sys.executable, '-u', str(opt.diagnostic),
            '--parent', str(opt.parent), '--checkpoint', str(opt.best_dir / 'best.pth'),
            '--data-root', str(opt.data_root), '--output', str(output),
            '--panel-scenes', '128', '--batch-size', '4',
        ], cwd=str(opt.source), env=environment, stdout=stream, stderr=subprocess.STDOUT)
    record = {'started': started, 'finished': datetime.now().astimezone().isoformat(),
              'returncode': completed.returncode, 'best': best, 'output': str(output)}
    receipt.write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(record), flush=True)
    assert completed.returncode == 0


if __name__ == '__main__':
    main()
