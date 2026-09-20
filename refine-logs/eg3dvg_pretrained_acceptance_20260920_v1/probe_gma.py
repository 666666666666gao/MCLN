import json,os,sys
from pathlib import Path
r=Path(__file__).resolve().parent;os.chdir(str(r/'source'));sys.path.insert(0,str(r/'source'));sys.path.insert(1,str(r/'source/pointnet2'))
import torch
from models.encoder_decoder_layers import GMA
torch.manual_seed(2027)
c=torch.load(str(r/'official_scanrefer.pth'),map_location='cpu')
prefix='module.decoder.0.cross_v.'
state={k[len(prefix):]:v for k,v in c['model'].items() if k.startswith(prefix)}
m=GMA(288,8).eval();m.load_state_dict(state,strict=True)
assert m.lang_cond_fc.out_features//m.n_head==m.pairwise_loc_fc[-1].out_features==9
print(json.dumps({'n_head':m.n_head,'language_width':m.lang_cond_fc.out_features,'geometry_width':m.pairwise_loc_fc[-1].out_features,'loaded_states':len(state)}),flush=True)
with torch.no_grad():
 output,attention=m(torch.randn(2,256,288),torch.randn(2,32,288),torch.randn(2,32,288),torch.randn(2,256,32,5),txt_embeds=torch.randn(2,288))
assert output.shape==(2,256,288) and attention.shape==(8,2,256,32)
assert torch.isfinite(output).all() and torch.isfinite(attention).all()
assert torch.allclose(attention.sum(-1),torch.ones(8,2,256),atol=1e-6)
assert all(torch.equal(v,state[k]) for k,v in m.state_dict().items())
rec={'status':'pass','weights':'official epoch69 decoder.0.cross_v','n_head':8,'language_width':72,'geometry_width':9,'output_shape':list(output.shape),'attention_shape':list(attention.shape),'state_unchanged':True,'cpu_only':True,'optimizer_steps':0,'formal_rows':0}
(r/'gma_probe.json').write_text(json.dumps(rec,indent=2));print(json.dumps(rec))
