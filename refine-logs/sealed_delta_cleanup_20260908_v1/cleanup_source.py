import json,hashlib,datetime,subprocess,shutil
from pathlib import Path
roots=[Path('/root/autodl-tmp/mcln_scanrefer_mask_geometry_pair_20260907_v1'),Path('/root/autodl-tmp/mcln_scanrefer_native_box_transfer_pair_20260907_v1')]
current=Path('/root/autodl-tmp/mcln_scanrefer_object_appearance_pair_20260908_v1')
active=json.loads((current/'input_manifest.json').read_text());terminal=json.loads((current/'receipt.json').read_text())
protected={item['path']:item['sha256'] for item in active['artifacts'].values()}
protected.update({item['path']:item['sha256'] for item in terminal['checkpoints'].values()})
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
records=[];evidence=[]
ps=subprocess.check_output(['ps','-eo','pid,args']).decode().splitlines()
for root in roots:
 assert not any(str(root) in line and 'python -c' not in line for line in ps)
 receipt=json.loads((root/'receipt.json').read_text());audit=json.loads((root/'independent_audit.json').read_text())
 assert receipt['status']=='complete' and receipt['eligible_for_fixed_terminal_formal_evaluation'] is False
 assert (root/'controller.exit').read_text().strip()=='0'
 assert audit['status']=='pass' and audit['decision']=='seal_fixed_configuration'
 assert audit['receipt_sha256']==sha(root/'receipt.json')
 for name,key in [('baseline_rows.json','baseline_rows_sha256'),('terminal_rows.json','terminal_rows_sha256'),('fit_point_batches.json','fit_batches_sha256')]:assert sha(root/name)==receipt[key]
 evidence.append({'root':str(root),'receipt_sha256':sha(root/'receipt.json'),'independent_audit_sha256':sha(root/'independent_audit.json'),'terminal_rows_sha256':receipt['terminal_rows_sha256']})
 for item in receipt['checkpoints'].values():
  path=Path(item['path']);assert path.parent.resolve()==root.resolve() and not path.is_symlink()
  assert str(path) not in protected and path.is_file()
  assert path.stat().st_size==item['bytes'] and sha(path)==item['sha256']
  records.append(dict(item))
assert len(records)==4 and sum(x['bytes'] for x in records)==92441740
for p,digest in protected.items():assert sha(p)==digest,p
out=Path('/root/autodl-tmp/mcln_sealed_delta_cleanup_20260908_v1');out.mkdir()
proof={'time_cst':datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),'authorization':'User requested deletion of useless weights; these two sealed failed runs have no active process.','weights':records,'evidence':evidence,'bytes_removed':sum(x['bytes'] for x in records),'protected_artifacts':protected,'free_before':shutil.disk_usage(str(current)).free}
(out/'deletion_plan.json').write_text(json.dumps(proof,indent=2)+'\n')
for item in records:Path(item['path']).unlink()
assert all(not Path(x['path']).exists() for x in records)
for p,digest in protected.items():assert sha(p)==digest,p
proof['free_after']=shutil.disk_usage(str(current)).free;proof['status']='complete';proof['active_appearance_weights_unchanged']=True
proof['all_rows_logs_and_audits_retained']=all((Path(v['root'])/'terminal_rows.json').exists() and (Path(v['root'])/'independent_audit.json').exists() for v in evidence)
(out/'receipt.json').write_text(json.dumps(proof,indent=2)+'\n')
print(json.dumps(proof))
