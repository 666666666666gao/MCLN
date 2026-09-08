from pathlib import Path
import json
base=Path('/root/autodl-tmp/mcln_pvground_runtime_20260908_v1/PV-Ground')
result={}
for name in ['main_utils.py','scripts/train_nr3d.sh','scripts/test_nr3d.sh']:
    lines=(base/name).read_text().splitlines()
    selected=set()
    for i,line in enumerate(lines):
        if 'butd' in line:
            selected.update(range(max(0,i-3),min(len(lines),i+4)))
    result[name]=[str(i+1)+': '+lines[i] for i in sorted(selected)]
root=Path('/root/autodl-tmp/mcln_pvground_nr_checkpoint_inspection_20260908_v1')
import torch
p=torch.load(str(root/'PV-Ground_NR3D.pth'),map_location='cpu')
scan=json.loads(Path('/root/autodl-tmp/mcln_pvground_scan_checkpoint_inspection_20260908_v1/state_inventory.json').read_bytes())
state=p['model'];result['nr_only_keys']=sorted(set(state)-set(scan));result['scan_only_keys']=sorted(set(scan)-set(state))
result['shape_differences']={n:{'nr':list(v.shape),'scan':scan[n]['shape']} for n,v in state.items() if n in scan and list(v.shape)!=scan[n]['shape']}
result['position_ids']={n:{'shape':list(v.shape),'dtype':str(v.dtype),'equals_arange':torch.equal(v,torch.arange(514).expand(1,-1))} for n,v in state.items() if 'position_ids' in n}
print(json.dumps(result))
