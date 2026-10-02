"""Run one paired support arm only after the current raw-point P3 completes."""
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
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--arm',choices=['tail_raw','tail_fused'],required=True)
    stage = parser.add_mutually_exclusive_group()
    stage.add_argument('--preflight-only',action='store_true')
    stage.add_argument('--train-after-preflight',action='store_true')
    args = parser.parse_args()
    root = args.root
    spec = json.loads((root/args.arm/'spec.json').read_bytes())
    assert spec['support_arm'] == args.arm
    assert json.loads((Path(spec['prior_p3_root'])/'status.json').read_bytes())['status'] == 'complete'
    environment = json.loads((Path(spec['runtime'])/'env_spec.json').read_bytes())
    env = dict(os.environ,**environment['env'])
    env['PYTHONPATH'] = str(root)+':'+env['PYTHONPATH']
    status_path = root/(args.arm+'_status.json')
    record = dict(status='running',completed=[],support_arm=args.arm,
                  started_cst=datetime.datetime.now().astimezone().isoformat())

    def save():
        temporary = status_path.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(record,indent=2)+'\n')
        os.replace(str(temporary),str(status_path))

    preflight_dir = 'preflight_'+args.arm
    phases = [(preflight_dir,'preflight')]
    if args.train_after_preflight:
        assert json.loads(status_path.read_bytes())['status'] == 'preflight_complete'
        phases = []
    if not args.preflight_only:
        phases += [(args.arm,'train'),(args.arm,'formal')]
    for phase,mode in phases:
        directory = root/phase
        if mode == 'train':
            proof = json.loads((root/preflight_dir/'preflight.json').read_bytes())
            assert proof['status'] == 'pass' and proof['optimizer_steps'] == 2
            assert proof['support_arm'] == args.arm and proof['member_mask_mapping_exact']
            assert proof['direct_semantic_to_p3_gradients_zero']
            assert proof['direct_native_mask_loss_to_refiner_gradients_zero']
            assert proof['native_call_order_verified'] and len(proof['native_call_witnesses']) == 4
            required = 2*proof['serialization_bytes']+64*1024**2
            assert shutil.disk_usage(root).free >= required
            if args.arm == 'tail_fused':
                control = json.loads((root/'tail_raw/formal/receipt.json').read_bytes())
                assert control['status'] == 'pass' and control['rows'] == 9508
        command = ['flock','-n',environment['resource_limits']['gpu_lock'],sys.executable,
                   '-u',str(root/'run.py'),'--spec',str(directory/'spec.json'),'--mode',mode]
        begin = time.time()
        with (directory/(mode+'.log')).open('x') as log:
            child = subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT)
            record.update(stage=phase+'/'+mode,process_pid=child.pid)
            save()
            code = child.wait()
        (directory/(mode+'.exit')).write_text(str(code)+'\n')
        if code:
            record.update(status='failed',exit_code=code)
            save()
            raise SystemExit(code)
        receipt_name = 'preflight.json' if mode == 'preflight' else 'receipt.json' if mode == 'train' else 'formal/receipt.json'
        receipt = json.loads((directory/receipt_name).read_bytes())
        if mode == 'train':
            assert receipt['training_steps'] == 3723 and receipt['fit_seen_exactly_once']
        else:
            assert receipt['status'] == 'pass'
        record['completed'].append(dict(stage=phase+'/'+mode,seconds=time.time()-begin))
        save()
    record.update(status='preflight_complete' if args.preflight_only else 'complete',
                  finished_cst=datetime.datetime.now().astimezone().isoformat())
    save()


if __name__ == '__main__':
    main()
