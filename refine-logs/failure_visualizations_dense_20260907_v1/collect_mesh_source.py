import hashlib,json,os,shlex
from pathlib import Path
import paramiko

root=Path('C:/Users/gb/Desktop/document/MCLN_3D_failure_visualizations_20260907')
out=root/'paper_style_dense';out.mkdir(exist_ok=True)
meshdir=out/'meshes';meshdir.mkdir(exist_ok=True)
cases=json.loads((root/'cases.json').read_bytes())['cases']
scenes=sorted({c['scene_id'] for c in cases})
base='/root/autodl-tmp/DATA_ROOT/scannet/scans/'
paths=[base+s+'/'+s+'_vh_clean_2.ply' for s in scenes]
align='/root/autodl-tmp/mcln_scanrefer_local_visual_preflight_20260906_v2/model_source/data/meta_data/scans_axis_alignment_matrices.json'
paths.append(align)
c=paramiko.SSHClient();c.load_system_host_keys();c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect('region-9.autodl.pro',port=33476,username='root',password=os.environ['MCLN_SSH_PASSWORD'],timeout=30)
code="""import hashlib,json,sys
from pathlib import Path
out=[]
for name in json.loads(sys.argv[1]):
 p=Path(name);h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8388608),b''):h.update(b)
 out.append({'remote_path':name,'bytes':p.stat().st_size,'sha256':h.hexdigest()})
print(json.dumps(out))
"""
_,stdout,stderr=c.exec_command('/root/miniconda3/envs/bdetr/bin/python -c '+shlex.quote(code)+' '+shlex.quote(json.dumps(paths)),timeout=60)
records=json.loads(stdout.read());assert stdout.channel.recv_exit_status()==0,stderr.read().decode()
sftp=c.open_sftp()
for item in records:
 dest=(meshdir if item['remote_path'].endswith('.ply') else out)/Path(item['remote_path']).name
 with sftp.open(item['remote_path'],'rb') as f:
  f.prefetch(file_size=item['bytes']);raw=f.read()
 assert len(raw)==item['bytes'] and hashlib.sha256(raw).hexdigest()==item['sha256']
 dest.write_bytes(raw);item['local_file']=str(dest.relative_to(out))
 print('MESH_SAVED',dest.name,item['bytes'],flush=True)
sftp.close();c.close()
(out/'mesh_sources.json').write_text(json.dumps({'files':records,'use':'Rendering only;original prediction boxes and GT are unchanged.'},indent=2)+'\n',encoding='utf-8')
