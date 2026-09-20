import hashlib,json,shutil
from pathlib import Path
r=Path(__file__).resolve().parent;e=Path('/root/mcln_eg3dvg_torch112_20260920_v1');site=e/'venv/lib/python3.7/site-packages/pointnet2';build=e/'pointnet2_build/pointnet2'
assert (r/'pointnet112_build.exit').read_text().strip()=='0'
spec=json.loads((r/'env_torch112.json').read_text());files={}
for name in ['pointnet2_utils.py','pointnet2_modules.py','pytorch_utils.py']:
 p=r/'source/pointnet2'/name;raw=p.read_bytes();files[name]=hashlib.sha256(raw).hexdigest()
 for dst in [site/name,build/name]:assert not dst.exists();dst.write_bytes(raw)
spec['pointnet2_build']['python_wrappers']=files
h=hashlib.sha256(json.dumps(spec,sort_keys=True,separators=(',',':')).encode()).hexdigest()
for p in [r/'env_torch112.json',e/'env_spec.json']:p.write_text(json.dumps(spec,indent=2))
(e/'env_spec.sha256').write_text(h+'\n')
rec={'status':'copied_unchanged_author_wrappers','reason':'new compiled package shadows older package; model imports pointnet2.pointnet2_utils','files':files,'env_spec_sha256':h}
(r/'pointnet112_package_repair.json').write_text(json.dumps(rec,indent=2));print(json.dumps(rec))
