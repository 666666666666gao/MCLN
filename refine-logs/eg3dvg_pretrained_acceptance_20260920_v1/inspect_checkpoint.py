import hashlib,json,os,sys
from pathlib import Path
r=Path(__file__).resolve().parent
os.chdir(str(r/'source'));sys.path.insert(0,str(r/'source'));sys.path.insert(1,str(r/'source/pointnet2'))
import torch
from models import EG
h=hashlib.sha256()
with (r/'official_scanrefer.pth').open('rb') as f:
 for chunk in iter(lambda:f.read(8388608),b''):h.update(chunk)
assert h.hexdigest()==json.loads((r/'checkpoint_download.json').read_text())['sha256']
c=torch.load(str(r/'official_scanrefer.pth'),map_location='cpu')
model=EG(num_class=256,num_obj_class=485,input_feature_dim=3,num_queries=256,num_decoder_layers=6,self_position_embedding='loc_learned',contrastive_align_loss=True,butd=True,pointnet_ckpt=None,data_path='/root/autodl-tmp/DATA_ROOT_mcln_meshsp/',self_attend=True)
keys=list(c['model']);prefix='module.' if all(k.startswith('module.') for k in keys) else ''
state={k[len(prefix):]:v for k,v in c['model'].items()};expected=model.state_dict()
rec={'checkpoint_sha256':h.hexdigest(),'checkpoint_bytes':(r/'official_scanrefer.pth').stat().st_size,'top_keys':list(c),'epoch':c.get('epoch'),'key_prefix':prefix,'state_count':len(state),'model_state_count':len(expected),'missing':sorted(set(expected)-set(state)),'unexpected':sorted(set(state)-set(expected)),'shape_mismatches':[(k,list(state[k].shape),list(expected[k].shape)) for k in set(state)&set(expected) if state[k].shape!=expected[k].shape],'training_steps':0,'model_forwards':0}
(r/'checkpoint_inspection.json').write_text(json.dumps(rec,indent=2));print(json.dumps(rec,indent=2))
