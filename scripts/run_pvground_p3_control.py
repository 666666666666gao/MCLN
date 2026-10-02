"""Run P3 preflight; optional fixed-budget training and formal validation."""
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
    stage = parser.add_mutually_exclusive_group()
    stage.add_argument('--preflight-only', action='store_true')
    stage.add_argument('--train-after-preflight', action='store_true')
    args = parser.parse_args()
    root = args.root
    spec = json.loads((root/'p3/spec.json').read_bytes())
    assert spec['p3'] and not spec['p2']
    environment = json.loads((Path(spec['runtime'])/'env_spec.json').read_bytes())
    env = dict(os.environ, **environment['env'])
    env['PYTHONPATH'] = str(root)+':'+env['PYTHONPATH']
    record = {'status':'running', 'completed':[],
              'started_cst':datetime.datetime.now().astimezone().isoformat()}

    def save():
        temporary = root/'status.json.tmp'
        temporary.write_text(json.dumps(record, indent=2)+'\n')
        os.replace(str(temporary), str(root/'status.json'))

    phases = [('preflight','preflight')]
    if args.train_after_preflight:
        assert json.loads((root/'status.json').read_bytes())['status'] == 'preflight_complete'
        phases = []
    if not args.preflight_only:
        phases += [('p3','train'), ('p3','formal')]
    for phase, mode in phases:
        directory = root/phase
        if mode == 'train':
            preflight = json.loads((root/'preflight/preflight.json').read_bytes())
            assert preflight['status'] == 'pass' and preflight['optimizer_steps'] == 2
            assert preflight['direct_semantic_to_p3_gradients_zero']
            required = 2 * preflight['serialization_bytes'] + 128 * 1024**2
            assert shutil.disk_usage(root).free >= required
        command = ['flock','-n',environment['resource_limits']['gpu_lock'],sys.executable,
                   '-u',str(root/'run.py'),'--spec',str(directory/'spec.json'),'--mode',mode]
        begin = time.time()
        with (directory/(mode+'.log')).open('x') as log:
            child = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
            record.update(stage=phase+'/'+mode, process_pid=child.pid)
            save()
            code = child.wait()
        (directory/(mode+'.exit')).write_text(str(code)+'\n')
        if code:
            record.update(status='failed', exit_code=code)
            save()
            raise SystemExit(code)
        receipt_path = directory/('preflight.json' if mode == 'preflight' else
                                 'receipt.json' if mode == 'train' else 'formal/receipt.json')
        receipt = json.loads(receipt_path.read_bytes())
        if mode == 'train':
            assert receipt['training_steps'] == 3723 and receipt['fit_seen_exactly_once']
        else:
            assert receipt['status'] == 'pass'
        record['completed'].append({'stage':phase+'/'+mode, 'seconds':time.time()-begin})
        save()
    record.update(status='preflight_complete' if args.preflight_only else 'complete',
                  finished_cst=datetime.datetime.now().astimezone().isoformat())
    save()


if __name__ == '__main__':
    main()
