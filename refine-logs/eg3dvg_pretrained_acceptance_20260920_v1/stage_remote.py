import hashlib,json,os,sys,tarfile
from pathlib import Path
r=Path('/root/autodl-tmp/mcln_eg3dvg_acceptance_20260920_v1')
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
raw=raw[:a]+"        # Evaluate original annotations using the upstream parser; no private cache.\n        assert self.split == 'val'\n        self.annos = self.load_annos(test_dataset)\n\n"+raw[b:]
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
