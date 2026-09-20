import datetime,hashlib,json,os,shlex,subprocess,tarfile
from pathlib import Path
import paramiko

base=Path(__file__).resolve().parent
source=base/'upstream'
root='/root/autodl-tmp/mcln_eg3dvg_acceptance_20260920_v1'
commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=str(source)).decode().strip()
files={}
with tarfile.open(str(base/'source.tar.gz'),'w:gz') as tar:
 for p in sorted(source.rglob('*')):
  rel=p.relative_to(source)
  if not p.is_file() or any(x in rel.parts for x in ['.git','.idea','__pycache__','build','pointnet2.egg-info']):continue
  files[rel.as_posix()]=hashlib.sha256(p.read_bytes()).hexdigest()
  tar.add(str(p),arcname=rel.as_posix())
manifest={'repository':'https://github.com/Gwan9Wook/EG3DVG','commit':commit,'files':files,'time_cst':datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()}
(base/'upstream_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
c=paramiko.SSHClient();c.load_system_host_keys();c.connect('region-9.autodl.pro',port=33476,username='root',password=os.environ['MCLN_SSH_PASSWORD'],timeout=30)
s=c.open_sftp();s.mkdir(root);s.mkdir(root+'/source')
for name in ['source.tar.gz','upstream_manifest.json']:s.put(str(base/name),root+'/'+name)
remote='''import hashlib,json,os,sys,tarfile
from pathlib import Path
r=Path(ROOT)
with tarfile.open(str(r/'source.tar.gz')) as t:
 assert all(not Path(m.name).is_absolute() and '..' not in Path(m.name).parts for m in t.getmembers())
 t.extractall(str(r/'source'))
m=json.loads((r/'upstream_manifest.json').read_text())
assert all(hashlib.sha256((r/'source'/p).read_bytes()).hexdigest()==h for p,h in m['files'].items())
p=r/'source/src/joint_det_dataset.py'
raw=p.read_text()
assert raw.count(', weights_only=False')==1
raw=raw.replace(', weights_only=False','')
a=raw.index('        # step 4. load text dataset')
b=raw.index('    # BRIEF load text data',a)
raw=raw[:a]+"        # Evaluate original annotations using the upstream parser; no private cache.\\n        assert self.split == 'val'\\n        self.annos = self.load_annos(test_dataset)\\n\\n"+raw[b:]
p.write_text(raw)
(r/'source_patch.json').write_text(json.dumps({'file':'src/joint_det_dataset.py','before':m['files']['src/joint_det_dataset.py'],'after':hashlib.sha256(p.read_bytes()).hexdigest(),'changes':['Use upstream load_annos instead of author placeholder text cache','Omit weights_only=False, which is the legacy torch.load default']},indent=2))
os.chdir(str(r/'source'));sys.path.insert(0,str(r/'source'));sys.path.insert(1,str(r/'source/pointnet2'))
import torch,numpy,transformers,spacy
import pointnet2_utils
from models import EG
from src.joint_det_dataset import Joint3DDataset
torch.manual_seed(2027);torch.cuda.manual_seed_all(2027)
xyz=torch.rand(2,128,3,device='cuda')
idx=pointnet2_utils.furthest_point_sample(xyz,16)
assert idx.shape==(2,16) and idx.max()<128
rec={'torch':torch.__version__,'numpy':numpy.__version__,'transformers':transformers.__version__,'spacy':spacy.__version__,'gpu':torch.cuda.get_device_name(),'fps_shape':list(idx.shape),'extension':pointnet2_utils._ext.__file__,'training_steps':0,'formal_rows':0}
(r/'import_kernel_receipt.json').write_text(json.dumps(rec,indent=2));print(json.dumps(rec))
'''.replace('ROOT',repr(root))
(base/'stage_remote.py').write_text(remote,encoding='utf-8')
s.put(str(base/'stage_remote.py'),root+'/stage_remote.py')
cmd='OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0 /root/miniconda3/envs/bdetr/bin/python '+shlex.quote(root+'/stage_remote.py')
_,o,e=c.exec_command(cmd,timeout=120)
out=o.read().decode();err=e.read().decode();code=o.channel.recv_exit_status()
record={'command':cmd,'stdout':out,'stderr':err,'exit':code,'root':root}
(base/'stage_result.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
for name in ['source_patch.json','import_kernel_receipt.json']:
 if name in s.listdir(root):s.get(root+'/'+name,str(base/name))
s.close();c.close();print(json.dumps(record,indent=2));raise SystemExit(code)
