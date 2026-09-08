import datetime,hashlib,json,subprocess,time
from pathlib import Path
root=Path(__file__).parent
spec=json.loads((root/'spec.json').read_bytes())
for name,digest in spec['files'].items():assert hashlib.sha256((root/name).read_bytes()).hexdigest()==digest,name
audit=Path(spec['candidate_audit']);candidate=Path(spec['candidate_root'])
time.sleep(max(0,datetime.datetime.fromisoformat(spec['first_check_cst']).timestamp()-time.time()))
for stage,filename in [('initial','initial_audit.json'),('terminal','audit.json')]:
    while not (audit/filename).is_file():
        if (audit/'controller.exit').is_file():
            raise RuntimeError('Audit terminated without required '+filename+'; exit='+ (audit/'controller.exit').read_text().strip())
        process=Path('/proc')/str(spec['audit_controller_pid'])/'cmdline'
        assert process.is_file() and (str(audit)+'/controller.py').encode() in process.read_bytes(),'Original audit controller missing; do not restart training'
        time.sleep(spec['poll_seconds'])
    receipt=json.loads((audit/filename).read_bytes())
    assert receipt['integrity_pass'] and receipt['formal_rows']==0
    bound=candidate/('receipt.json' if stage=='terminal' else 'initial/receipt.json')
    assert hashlib.sha256(bound.read_bytes()).hexdigest()==receipt['receipt_sha256']
    command=['/root/miniconda3/envs/bdetr/bin/python','-u',str(root/'compare.py'),'--spec',str(root/'spec.json'),'--stage',stage,'--output',str(root/(stage+'.json'))]
    with (root/(stage+'.log')).open('xb') as stream:
        result=subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT)
    (root/(stage+'.exit')).write_text(str(result.returncode)+'\n')
    assert result.returncode==0,stage
    print('PVG_CROSS_CONTROL_COMPLETE '+stage,flush=True)
