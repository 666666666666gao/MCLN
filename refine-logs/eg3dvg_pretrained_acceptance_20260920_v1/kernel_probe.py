import hashlib,json,os,sys
from pathlib import Path
r=Path(__file__).resolve().parent
env=json.loads(Path('/root/autodl-tmp/mcln_pvground_runtime_20260908_v1/env_spec.json').read_text())
assert hashlib.sha256(json.dumps(env,sort_keys=True,separators=(',',':')).encode()).hexdigest()=='966235b2ead7fa5a1de63e9537745b457374a4e5749ac4a7a26566b7b630c82c'
os.chdir(str(r/'source'));sys.path.insert(0,str(r/'source'));sys.path.insert(1,str(r/'source/pointnet2'))
import torch,numpy,transformers,spacy
import pointnet2_utils
from models import EG
from src.joint_det_dataset import Joint3DDataset
torch.manual_seed(2027);torch.cuda.manual_seed_all(2027)
xyz=torch.rand(2,128,3,device='cuda');idx=pointnet2_utils.furthest_point_sample(xyz,16)
assert idx.shape==(2,16) and idx.max()<128
rec={'torch':torch.__version__,'numpy':numpy.__version__,'transformers':transformers.__version__,'spacy':spacy.__version__,'gpu':torch.cuda.get_device_name(),'fps_shape':list(idx.shape),'extension':pointnet2_utils._ext.__file__,'training_steps':0,'formal_rows':0,'base_environment_sha256':'966235b2ead7fa5a1de63e9537745b457374a4e5749ac4a7a26566b7b630c82c'}
(r/'import_kernel_receipt.json').write_text(json.dumps(rec,indent=2));print(json.dumps(rec))
