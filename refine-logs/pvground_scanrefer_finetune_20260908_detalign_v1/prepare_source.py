import hashlib,json,shutil,subprocess
from pathlib import Path
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
root=Path('/root/autodl-tmp/mcln_scanrefer_detection_aligned_source_20260908_v1')
check=Path('/root/autodl-tmp/mcln_scanrefer_detection_augmentation_20260908_v1')
r=json.loads((check/'receipt.json').read_bytes())
assert r['status']=='pass' and r['rows']==32 and r['fixed_max_box_error']<3e-5
assert (check/'controller.exit').read_text().strip()=='0'
base=Path('/root/autodl-tmp/mcln_scanrefer_object_appearance_native_20260908_v1/model_source')
old=json.loads((base/'appearance_source_manifest.json').read_bytes())
assert len(old['files'])==625
dst=root/'model_source'
dst.mkdir(parents=True)
for name,digest in old['files'].items():
    assert sha(base/name)==digest,name
    target=dst/name
    target.parent.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(base/name,target)
shutil.copyfile(check/'fixed_dataset.py',dst/'src/joint_det_dataset.py')
new=dict(old)
new['parent_source']=str(base)
new['parent_manifest_sha256']=sha(base/'appearance_source_manifest.json')
new['files']={name:sha(dst/name) for name in old['files']}
changed=[name for name in old['files'] if new['files'][name]!=old['files'][name]]
assert changed==['src/joint_det_dataset.py']
assert new['files'][changed[0]]==r['fixed_dataset_sha256']
new['single_change']='detected corners flip before rotation to match points'
(dst/'appearance_source_manifest.json').write_text(json.dumps(new,indent=2)+'\n')
m=json.loads(Path('/root/autodl-tmp/mcln_scanrefer_object_appearance_pair_20260908_v1/input_manifest.json').read_bytes())
m['model_source']=str(dst)
m['source_manifest_sha256']=sha(dst/'appearance_source_manifest.json')
(root/'input_manifest.json').write_text(json.dumps(m,indent=2)+'\n')
env=json.loads(Path('/root/autodl-tmp/mcln_pvground_runtime_20260908_v1/env_spec.json').read_bytes())
assert hashlib.sha256(json.dumps(env,sort_keys=True,separators=(',',':')).encode()).hexdigest()=='966235b2ead7fa5a1de63e9537745b457374a4e5749ac4a7a26566b7b630c82c'
gpu=int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits']).decode().strip())
assert gpu<500
previous=Path('/root/autodl-tmp/mcln_pvground_scanrefer_finetune_20260908_v1')
prev=json.loads((previous/'receipt.json').read_bytes())
assert (previous/'controller.exit').read_text().strip()=='0'
assert sha(previous/'terminal.pth')==prev['terminal_sha256']
latest=previous/'latest.pth'
assert latest.resolve().parent==previous.resolve() and latest.is_file()
cleanup=dict(path=str(latest),bytes=latest.stat().st_size,sha256=sha(latest),reason='completed failed run; fixed terminal and parent retained')
latest.unlink()
assert sha(previous/'terminal.pth')==prev['terminal_sha256']
free=shutil.disk_usage('/root/autodl-tmp').free
assert free>3*1024**3
result=dict(status='pass',source_files=625,changed_files=changed,source_manifest_sha256=m['source_manifest_sha256'],dataset_sha256=r['fixed_dataset_sha256'],disk_free=free,gpu_mib=gpu,cleanup=cleanup)
(root/'preparation_receipt.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))
