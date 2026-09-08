"""Run a bounded diagnostic only after the existing ScanRefer pipeline exits."""
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


def main():
    root=Path(__file__).parent
    spec=json.loads((root/'spec.json').read_bytes())
    runtime=Path(spec['runtime'])
    dependency=Path(spec['dependency_root'])
    (root/'queue.pid').write_text(str(os.getpid())+'\n')
    first=datetime.datetime.fromisoformat(spec['first_check_cst']).timestamp()
    if time.time()<first:time.sleep(first-time.time())
    while not (dependency/'controller.exit').exists():
        process=Path('/proc')/str(spec['dependency_pid'])/'cmdline'
        assert process.exists(),'Formal controller absent without terminal receipt; do not restart'
        assert str(dependency).encode() in process.read_bytes(),'Dependency PID identity changed'
        print('REPLAY_WAIT '+datetime.datetime.now().isoformat(),flush=True)
        time.sleep(300)
    assert (dependency/'controller.exit').read_text().strip()=='0'
    assert (root/'input_controller.exit').read_text().strip()=='0'
    for name,digest in spec['files'].items():
        assert hashlib.sha256((root/name).read_bytes()).hexdigest()==digest,name
    env=os.environ.copy();env.update(json.loads((runtime/'env_spec.json').read_bytes())['env'])
    env.update(CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    with open(spec['gpu_lock'],'a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for name in ['process_a','process_b']:
            with (root/(name+'.log')).open('xb') as log:
                result=subprocess.run([str(runtime/'venv/bin/python'),'-u',str(root/'replay.py'),
                    '--spec',str(root/'spec.json'),'--output',str(root/name)],env=env,stdout=log,stderr=subprocess.STDOUT)
            (root/(name+'.exit')).write_text(str(result.returncode)+'\n')
            assert result.returncode==0,name+' failed; no automatic retry'
    # GPU lock released before CPU-only trace comparison.
    os.environ['CUDA_VISIBLE_DEVICES']=''
    import torch
    torch.set_num_threads(1)
    assert not torch.cuda.is_initialized()
    receipts=[json.loads((root/n/'receipt.json').read_bytes()) for n in ['process_a','process_b']]
    traces=[]
    for name,receipt in zip(['process_a','process_b'],receipts):
        path=root/name/'reference_trace.pt'
        assert hashlib.sha256(path.read_bytes()).hexdigest()==receipt['reference_trace_sha256']
        traces.append(torch.load(str(path),map_location='cpu'))
    assert receipts[0]['input_sha256']==receipts[1]['input_sha256']
    left,right=traces
    assert list(left)==list(right)
    comparison={}
    for key in left:
        assert left[key].shape==right[key].shape and left[key].dtype==right[key].dtype
        delta=(left[key].double()-right[key].double()).abs()
        comparison[key]=dict(exact=torch.equal(left[key],right[key]),max_abs=float(delta.max()),mean_abs=float(delta.mean()))
    different=[key for key,value in comparison.items() if not value['exact']]
    result=dict(status='complete',time_cst=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
        processes=2,full_forwards=6,optimizer_steps=0,formal_rows=0,input_sha256=receipts[0]['input_sha256'],
        cross_process_first_forward=comparison,first_observed_difference=different[0] if different else None,
        within_process_seed_repeat_exact=[r['repeated_seed_exact'] for r in receipts],
        scope='one frozen batch; no whole6887 replay or original capacity backward; not a metric or causal proof')
    (root/'comparison.json').write_text(json.dumps(result,indent=2)+'\n')
    print('REPLAY_QUEUE_COMPLETE '+json.dumps(result),flush=True)


if __name__=='__main__':main()
