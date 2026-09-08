import datetime,hashlib,json,subprocess,time
from pathlib import Path
root=Path(__file__).parent
spec=json.loads((root/'spec.json').read_bytes())
for name,digest in spec['files'].items():assert hashlib.sha256((root/name).read_bytes()).hexdigest()==digest,name
audit=Path(spec['candidate_audit']);candidate=Path(spec['candidate_root'])
for stage,filename in [('initial','initial_audit.json'),('terminal','audit.json')]:
    if stage=='terminal':time.sleep(max(0,datetime.datetime.fromisoformat(spec['terminal_first_check_cst']).timestamp()-time.time()))
    while not (audit/filename).is_file():
        if (audit/'controller.exit').is_file():raise RuntimeError('Original audit terminated without '+filename)
        process=Path('/proc')/str(spec['audit_controller_pid'])/'cmdline'
        assert process.is_file() and (str(audit)+'/controller.py').encode() in process.read_bytes()
        time.sleep(300)
    receipt=json.loads((audit/filename).read_bytes())
    assert receipt['integrity_pass'] and receipt['formal_rows']==0
    bound=candidate/('receipt.json' if stage=='terminal' else 'initial/receipt.json')
    assert hashlib.sha256(bound.read_bytes()).hexdigest()==receipt['receipt_sha256']
    for label in ['A','B']:
        prefix=stage+'_'+label+'_C'
        cmd=['/root/miniconda3/envs/bdetr/bin/python','-u',str(root/'compare.py'),'--spec',str(root/(label+'_C.json')),'--stage',stage,'--output',str(root/(prefix+'.json'))]
        with (root/(prefix+'.log')).open('xb') as stream:result=subprocess.run(cmd,stdout=stream,stderr=subprocess.STDOUT)
        (root/(prefix+'.exit')).write_text(str(result.returncode)+'\n')
        assert result.returncode==0,prefix
        print('PVG_OBSERVATION_COMPARISON_COMPLETE '+prefix,flush=True)
