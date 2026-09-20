import fcntl,hashlib,json,os,sys
from pathlib import Path
r=Path(__file__).resolve().parent;os.chdir(str(r/'source'));sys.path.insert(0,str(r/'source'));sys.path.insert(1,str(r/'source/pointnet2'))
lock=open('/root/autodl-tmp/mcln_v99_backbone_gpu0.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
import numpy,torch,transformers,spacy
import pointnet2._ext as ext
import pointnet2_utils
from models.encoder_decoder_layers import SWA
spec=json.loads((r/'env_torch112.json').read_text())
actual={'torch':torch.__version__,'numpy':numpy.__version__,'transformers':transformers.__version__,'spacy':spacy.__version__}
assert actual==spec['required_versions'],actual
assert ext.__file__.startswith(spec['runtime_root']+'/venv/'),ext.__file__
assert torch.randn(1).is_nested is False
torch.manual_seed(2027);torch.cuda.manual_seed_all(2027)
with torch.no_grad():
 xyz=torch.rand(2,64,3,device='cuda');features=torch.rand(2,8,64,device='cuda')
 fps=pointnet2_utils.furthest_point_sample(xyz,16)
 xyz_centers=pointnet2_utils.gather_operation(xyz.transpose(1,2).contiguous(),fps).transpose(1,2).contiguous()
 grouped,indices=pointnet2_utils.QueryAndGroup(radius=.2,nsample=8,use_xyz=False)(xyz,xyz_centers,features)
 assert grouped.shape==(2,8,16,8) and torch.isfinite(grouped).all()
 layer=SWA(288,8,dropout=.1).cuda().eval()
 source=torch.randn(32,1,288,device='cuda');query=torch.randn(256,1,288,device='cuda')
 mask=torch.zeros(1,256,32,dtype=torch.bool,device='cuda');mask[:,:,3]=True
 out,attention,source_weights=layer(source,query,attn_mask=mask)
 assert out.shape==(1,256,288) and torch.isfinite(out).all()
 assert torch.isfinite(attention).all() and torch.isfinite(source_weights).all()
 torch.cuda.synchronize()
rec={'status':'pass','versions':actual,'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(),'fps_shape':list(fps.shape),'group_shape':list(grouped.shape),'swa_output_shape':list(out.shape),'pointnet_extension':ext.__file__,'env_spec_sha256':hashlib.sha256(json.dumps(spec,sort_keys=True,separators=(',',':')).encode()).hexdigest(),'seed':2027,'training_steps':0,'formal_rows':0}
(r/'torch112_witness.json').write_text(json.dumps(rec,indent=2));print('EG_TORCH112_WITNESS '+json.dumps(rec))
