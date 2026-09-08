import datetime,hashlib,importlib.util,json,sys,time
from pathlib import Path
root=Path('/root/autodl-tmp/mcln_pvground_scanrefer_formal_20260908_vsaorder_v1')
spec=json.loads((root/'spec.json').read_bytes())
for name,digest in spec['files'].items():
    assert hashlib.sha256((root/name).read_bytes()).hexdigest()==digest,name
begin=time.time()
modules={}
for name in ['audit','evaluate']:
    module_spec=importlib.util.spec_from_file_location('formal_'+name,str(root/(name+'.py')))
    module=importlib.util.module_from_spec(module_spec);module_spec.loader.exec_module(module)
    modules[name]=module
directory=Path('/root/autodl-tmp/mcln_pvground_scanrefer_finetune_20260908_v1/initial')
rows,metrics,checks=modules['audit'].recount_native_rows(directory)
assert len(rows)==6887
assert metrics=={mode:modules['evaluate'].row_metrics(rows,mode) for mode in ['bbs','bbf']}
assert 'torch' not in sys.modules
result={'status':'pass','time_cst':datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
        'scope':'new formal recount code checked on completed initial holdout outputs',
        'actual_recount_rows':6887,'formal_rows':0,'gpu_forwards':0,'optimizer_steps':0,
        'torch_imported':False,'elapsed_seconds':time.time()-begin,'checks':checks,
        'initial_receipt_sha256':hashlib.sha256((directory/'receipt.json').read_bytes()).hexdigest(),
        'evaluator_sha256':spec['files']['evaluate.py'],'auditor_sha256':spec['files']['audit.py']}
with (root/'preparation_receipt.json').open('x') as stream:json.dump(result,stream,indent=2);stream.write('\n')
print('PVG_FORMAL_CPU_PREPARATION_PASS '+json.dumps(result),flush=True)
