import datetime,hashlib,importlib.util,json,os,subprocess,time
from pathlib import Path
root=Path('/root/autodl-tmp/mcln_pvground_scanrefer_endpoint_audit_20260917_fixed_memory_v1')
spec=json.loads((root/'spec.json').read_bytes())
training=Path(spec['training_root'])
for name,digest in spec['files'].items():
    assert hashlib.sha256((root/name).read_bytes()).hexdigest()==digest,name
assert hashlib.sha256((training/'spec.json').read_bytes()).hexdigest()==spec['training_spec_sha256']
deadline=datetime.datetime.fromisoformat(spec['first_check_cst']).timestamp()
time.sleep(max(0,deadline-time.time()))
module_spec=importlib.util.spec_from_file_location('endpoint_auditor',str(root/'audit.py'))
module=importlib.util.module_from_spec(module_spec);module_spec.loader.exec_module(module)
input_spec=json.loads((training/'spec.json').read_bytes())
manifest=json.loads(Path(input_spec['input_manifest']).read_bytes())
assert module.sha(manifest['split_protocol'])==manifest['split_protocol_sha256']
ids=json.loads(Path(manifest['split_protocol']).read_bytes())['row_ids']['holdout']
initial_done=False
while True:
    if not initial_done and (training/'initial/receipt.json').is_file():
        rows,metrics,checks=module.audit_stage(training,'initial',ids)
        result={'integrity_pass':True,'stage':'initial','formal_rows':0,'metrics':metrics,'checks':checks,
                'receipt_sha256':module.sha(training/'initial/receipt.json'),
                'time_cst':datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()}
        with (root/'initial_audit.json').open('x') as stream:json.dump(result,stream,indent=2);stream.write('\n')
        print('PVG_INITIAL_AUDIT_COMPLETE '+json.dumps(result),flush=True)
        initial_done=True
    if (training/'controller.exit').is_file():
        code=int((training/'controller.exit').read_text())
        if code!=0:
            result={'status':'dependency_failed','training_exit':code,'formal_rows':0}
            (root/'dependency_failure.json').write_text(json.dumps(result)+'\n')
            print('PVG_DEPENDENCY_FAILED '+json.dumps(result),flush=True)
            raise SystemExit(code)
        command=[str(Path(input_spec['runtime'])/'venv/bin/python'),'-u',str(root/'audit.py'),'--root',str(training),'--out',str(root/'audit.json')]
        environment=json.loads((Path(input_spec['runtime'])/'env_spec.json').read_bytes())['env']
        result=subprocess.run(command,env=dict(os.environ,**dict(environment,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1')))
        (root/'audit.exit').write_text(str(result.returncode)+'\n')
        if result.returncode==0:
            verified=json.loads((root/'audit.json').read_bytes())
            assert verified['integrity_pass'] and verified['fixed_visual_memory_verified']
            receipt=json.loads((training/'receipt.json').read_bytes())
            assert verified['receipt_sha256']==module.sha(training/'receipt.json')
            assert receipt['terminal_sha256']==module.sha(training/'terminal.pth')
            latest=training/'latest.pth'
            assert latest.resolve().parent==training.resolve() and latest.is_file()
            record={'path':str(latest),'sha256':module.sha(latest),'bytes':latest.stat().st_size,
                    'retained_terminal_sha256':receipt['terminal_sha256'],
                    'reason':'superseded snapshot after independent fixed endpoint audit'}
            (root/'cleanup_planned.json').write_text(json.dumps(record)+'\n')
            latest.unlink()
            record['deleted']=not latest.exists()
            (root/'cleanup_receipt.json').write_text(json.dumps(record)+'\n')
        raise SystemExit(result.returncode)
    process=Path('/proc')/str(spec['training_controller_pid'])/'cmdline'
    assert process.is_file() and (str(training)+'/controller.py').encode() in process.read_bytes(),'original training controller disappeared without an exit receipt'
    time.sleep(spec['poll_seconds'])
