"""One normalized continuation followed by the unchanged formal evaluator."""
import argparse
import datetime
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    root = parser.parse_args().root
    preflight = json.loads((root / 'preflight/preflight.json').read_bytes())
    assert preflight['status'] == 'pass' and preflight['optimizer_steps'] == 2
    assert preflight['batch_size'] == 8 and preflight['g_strict_restore']
    assert preflight['semantic_consistency'] and preflight['new_model_states'] == 0
    assert preflight['capacity']['expanded_count_normalization']
    spec = json.loads((root / 'normalized/spec.json').read_bytes())
    environment = json.loads((Path(spec['runtime']) / 'env_spec.json').read_bytes())
    env = dict(os.environ, **environment['env'])
    env['PYTHONPATH'] = str(root) + ':' + env['PYTHONPATH']
    lock = environment['resource_limits']['gpu_lock']
    assert not (root / 'status.json').exists()
    assert shutil.disk_usage(root).free >= 2 * preflight['serialization_bytes'] + 256 * 1024**2
    record = dict(status='running', completed=[], started_cst=datetime.datetime.now().astimezone().isoformat())

    def save():
        temporary = root / 'status.json.tmp'
        temporary.write_text(json.dumps(record, indent=2) + '\n')
        os.replace(str(temporary), str(root / 'status.json'))

    directory = root / 'normalized'
    for mode in ('train', 'formal'):
        if mode == 'train':
            assert shutil.disk_usage(root).free >= 2 * preflight['serialization_bytes'] + 256 * 1024**2
        begin = time.time()
        command = ['flock', '-n', lock, sys.executable, '-u', str(root / 'run.py'),
                   '--spec', str(directory / 'spec.json'), '--mode', mode]
        with (directory / (mode + '.log')).open('x') as log:
            child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env)
            record.update(stage='normalized/' + mode, process_pid=child.pid)
            save()
            code = child.wait()
        (directory / (mode + '.exit')).write_text(str(code) + '\n')
        if code:
            record.update(status='failed', exit_code=code)
            save()
            raise SystemExit(code)
        receipt = json.loads((directory / ('receipt.json' if mode == 'train' else 'formal/receipt.json')).read_bytes())
        if mode == 'train':
            assert receipt['training_steps'] == 3723 and receipt['fit_seen_exactly_once']
        else:
            assert receipt['status'] == 'pass' and receipt['rows'] == 9508
        record['completed'].append(dict(mode=mode, seconds=time.time() - begin))
        save()
    record.update(status='complete', finished_cst=datetime.datetime.now().astimezone().isoformat())
    save()


if __name__ == '__main__':
    main()
