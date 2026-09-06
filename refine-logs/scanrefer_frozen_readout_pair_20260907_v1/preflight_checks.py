import ast,hashlib,json,subprocess,sys
from pathlib import Path
from scripts.evaluate_scanrefer_frozen_readout_official import promotion_check
from scripts.audit_scanrefer_frozen_readout_official import native_metrics,rec_compare
from scripts.audit_scanrefer_frozen_readout_pair import audit_rows,audit_checkpoints
m=json.loads(Path('input_manifest.json').read_text())
for name,digest in m['files'].items():
 raw=Path(name).read_bytes()
 assert hashlib.sha256(raw).hexdigest()==digest
 ast.parse(raw,filename=name)
base={'rows':9508,'rec_hits025':5572,'rec_hits050':4797,'mask_hits025':5689,'mask_hits050':4974,'mask_miou':45.9226}
assert promotion_check(base,base)['advance_to_nr3d_sr3d_rec']
for field,value in [('rec_hits025',5571),('rec_hits050',4796),('mask_hits025',5581),('mask_hits050',4820),('mask_miou',44.71)]:
 bad=dict(base);bad[field]=value
 assert not promotion_check(base,bad)['advance_to_nr3d_sr3d_rec'],field
better=dict(base,rec_hits025=5573)
assert not promotion_check(better,base)['advance_to_nr3d_sr3d_rec']
historical=Path('/root/autodl-tmp/mcln_scanrefer_range_official_20260907_v1/result/native_rows.json')
rows=json.loads(historical.read_text())
assert native_metrics(rows['protected_v99'])=={'rows':9508,'rec_hits025':5515,'rec_hits050':4411}
effect=rec_compare(rows['protected_v99'],rows['local_v99'])
assert effect['effects']['025']['net']==-13 and effect['effects']['050']['net']==8
for module in ['run_scanrefer_frozen_readout_pair','audit_scanrefer_frozen_readout_pair','evaluate_scanrefer_frozen_readout_official','audit_scanrefer_frozen_readout_official','queue_scanrefer_frozen_readout_posttraining']:
 p=subprocess.run([sys.executable,'-m','scripts.'+module,'--help'],stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
 assert p.returncode==0,p.stdout.decode()
print(json.dumps({'python':sys.version.split()[0],'original_environment_cli_and_imports':True,'all_source_ast':True,'promotion_nonregression_and_mask_checks':True,'historical_real_row_recomputation':True,'new_gpu_forwards':0,'no_trial_result_claim':True}))
