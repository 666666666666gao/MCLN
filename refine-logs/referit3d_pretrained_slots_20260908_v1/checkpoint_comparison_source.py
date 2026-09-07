import hashlib,json,os
from pathlib import Path
import torch
torch.set_num_threads(1)
paths={
 'scanrefer_e71':'/root/autodl-tmp/DATA_ROOT/output/preserved_best/mcln_pair_sweep/mcln_pair_default_rankblend010_2ep_best_acc025_epoch71_0.57993.pth',
 'nr3d_e57':'/root/autodl-tmp/DATA_ROOT/output/network_v99_baseline_gt/nr3d/control/official_rec_monitor/official_best_rec025_epoch_57_0p56652741.pth'}
expected={'scanrefer_e71':'3e44f4bdad3bd66ad82102032e1cb0241de57d147c0aa1d3eff9736926ef2208','nr3d_e57':'76aa6cd49ca20a34e78509465f1185b1b9040e60807ad327d6b0876aeb6edba1'}
states={};metadata={}
for k,p in paths.items():
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 assert h.hexdigest()==expected[k]
 v=torch.load(p,map_location='cpu');states[k]={name[7:]:x for name,x in v['model'].items()}
 config=vars(v['config'])
 keys=['num_decoder_layers','num_target','use_color','use_height','use_multiview','use_soft_token_loss','use_source_choice_selector','source_choice_selector_sources','use_source_moe','butd','butd_gt','butd_cls','joint_det']
 metadata[k]=dict(path=p,sha256=h.hexdigest(),state_tensors=len(states[k]),config={n:config.get(n) for n in keys},evaluation_only=v.get('evaluation_only',False),optimizer_present='optimizer' in v)
a,b=states.values();common=sorted(set(a)&set(b))
result=dict(artifacts=metadata,scan_only_keys=sorted(set(a)-set(b)),nr_only_keys=sorted(set(b)-set(a)),
 shape_mismatches=[dict(name=n,scan_shape=list(a[n].shape),nr_shape=list(b[n].shape)) for n in common if a[n].shape!=b[n].shape],
 same_tensor_values=sum(torch.equal(a[n],b[n]) for n in common),common_keys=len(common),
 scope='CPU state-shape comparison, not native model load/forward or performance test.',gpu_forwards=0,optimizer_steps=0)
print(json.dumps(result))
