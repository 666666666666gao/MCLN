import hashlib,json,os,shlex
from pathlib import Path
import paramiko
repo=Path('C:/Users/gb/.codex_mcln_g0_20260905')
archive=repo/'refine-logs/scanrefer_object_appearance_native_20260908_v1';archive.mkdir()
root='/root/autodl-tmp/mcln_scanrefer_object_appearance_native_20260908_v1'
base=json.loads((repo/'refine-logs/native_score_fit_20260907_v2/input_manifest.json').read_bytes())
config={k:base[k] for k in ['artifacts','data_root','train_superpoint_files']}
config.update(selected_row_ids=base['selected_row_ids'][:16],cache_root='/root/autodl-tmp/mcln_scanrefer_openshape_cache_20260908_v1',
              parent_source=base['model_source'],parent_source_manifest_sha256=base['source_manifest_sha256'],
              formal_rows=0,disposable_optimizer_steps=2)
patches=[('                 pointnet_ckpt_sha256=""):', '                 pointnet_ckpt_sha256="",\n                 use_pretrained_object_appearance=False):'),
         ('        self.butd = butd\n','        self.butd = butd\n        self.object_appearance = None\n        if use_pretrained_object_appearance:\n            assert butd and d_model == 288\n            from .pretrained_object_appearance import PretrainedObjectAppearance\n            self.object_appearance = PretrainedObjectAppearance()\n'),
         ('                                        , 1).transpose(1, 2).contiguous()\n','                                        , 1).transpose(1, 2).contiguous()\n            if self.object_appearance is not None:\n                detected_feats = self.object_appearance(\n                    detected_feats, inputs[\'det_visual_features\'],\n                    inputs[\'det_visual_available\'] & ~detected_mask)\n')]
(archive/'patches.json').write_bytes(json.dumps(patches,indent=2).encode()+b'\n')
overlays=['models/pretrained_object_appearance.py','scripts/scanrefer_data_contract.py','scripts/scanrefer_joint_readout.py']
config['overlays']={n:hashlib.sha256((repo/n).read_bytes()).hexdigest() for n in overlays}
(archive/'manifest.json').write_bytes(json.dumps(config,indent=2).encode()+b'\n')
c=paramiko.SSHClient();c.load_system_host_keys();c.connect('region-9.autodl.pro',port=33476,username='root',password=os.environ['MCLN_SSH_PASSWORD'],timeout=30);s=c.open_sftp()
for sub in ['', '/overlays','/overlays/models','/overlays/scripts']:s.mkdir(root+sub)
for name in ['manifest.json','patches.json']:s.put(str(archive/name),root+'/'+name)
for name in overlays:s.put(str(repo/name),root+'/overlays/'+name)
s.put(str(repo/'scripts/probe_scanrefer_pretrained_object_memory.py'),root+'/probe.py')
prepare='''import ast,hashlib,json,sys
from pathlib import Path
root=Path(sys.argv[1]);config=json.loads((root/'manifest.json').read_text());parent=Path(config['parent_source'])
raw=(parent/'native_source_manifest.json').read_bytes()
assert hashlib.sha256(raw).hexdigest()==config['parent_source_manifest_sha256']
files=json.loads(raw)['files'];source=root/'model_source';source.mkdir()
for name,digest in files.items():
    content=(parent/name).read_bytes();assert hashlib.sha256(content).hexdigest()==digest,name
    target=source/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(content)
before=files['models/mcln.py'];path=source/'models/mcln.py';code=path.read_text()
for old,new in json.loads((root/'patches.json').read_text()):
    assert code.count(old)==1,(old,code.count(old));code=code.replace(old,new)
ast.parse(code);path.write_bytes(code.encode());files['models/mcln.py']=hashlib.sha256(path.read_bytes()).hexdigest()
for name,digest in config['overlays'].items():
    content=(root/'overlays'/name).read_bytes();assert hashlib.sha256(content).hexdigest()==digest
    if name in files:assert files[name]==digest,(name,'unexpected helper drift')
    (source/name).write_bytes(content);files[name]=digest
record=dict(parent_source=config['parent_source'],parent_manifest_sha256=config['parent_source_manifest_sha256'],base_mcln_sha256=before,files=files)
(source/'appearance_source_manifest.json').write_bytes(json.dumps(record,indent=2).encode()+b'\\n')
print(json.dumps(dict(files=len(files),source_sha256=hashlib.sha256((source/'appearance_source_manifest.json').read_bytes()).hexdigest())))
'''
(archive/'prepare_source.py').write_bytes(prepare.encode())
with s.open(root+'/prepare.py','wb') as f:f.write(prepare.encode())
_,o,e=c.exec_command('/root/miniconda3/envs/bdetr/bin/python '+shlex.quote(root+'/prepare.py')+' '+shlex.quote(root),timeout=60)
raw=o.read();err=e.read();code=o.channel.recv_exit_status()
(archive/'preparation_command.json').write_bytes(json.dumps(dict(stdout=raw.decode(),stderr=err.decode(),exit_code=code),indent=2).encode()+b'\n')
print(raw.decode(),err.decode());assert code==0
s.get(root+'/model_source/appearance_source_manifest.json',str(archive/'appearance_source_manifest.json'))
shell='cd '+shlex.quote(root)+'\nexport OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=0\nflock /root/autodl-tmp/mcln_v99_backbone_gpu0.lock /root/miniconda3/envs/bdetr/bin/python -u probe.py --manifest '+shlex.quote(root+'/manifest.json')+' > probe.log 2>&1\ncode=$?\nprintf "%s\\n" "$code" > probe.exit\nexit "$code"\n'
(archive/'run.sh').write_bytes(shell.encode())
with s.open(root+'/run.sh','wb') as f:f.write(shell.encode())
_,o,e=c.exec_command('screen -dmS mcln_os_native_probe_v1 bash '+shlex.quote(root+'/run.sh'))
assert o.channel.recv_exit_status()==0,e.read().decode()
_,o,e=c.exec_command('screen -ls');print(o.read().decode())
(archive/'prepare_from_local.py').write_bytes(Path(__file__).read_bytes())
s.close();c.close()
