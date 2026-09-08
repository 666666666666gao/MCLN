import hashlib,json
from pathlib import Path
import numpy as np
old=Path('/root/autodl-tmp/mcln_pvground_scanrefer_finetune_20260908_v1/initial')
new=Path('/root/autodl-tmp/mcln_pvground_scanrefer_finetune_20260908_detalign_v1/initial')
a=[json.loads(x) for x in (old/'rows.jsonl').read_text().splitlines()]
b=[json.loads(x) for x in (new/'rows.jsonl').read_text().splitlines()]
assert len(a)==len(b)==6887
result={'rows':len(a),'identical_fields':{},'modes':{}}
for key in ['row_id','scan_id','target_id','point_sha256','root_box']:
    result['identical_fields'][key]=sum(x[key]==y[key] for x,y in zip(a,b))
boxes=[np.load(str(p/'boxes.npy'),mmap_mode='r') for p in [old,new]]
scores=[np.load(str(p/'scores.npy'),mmap_mode='r') for p in [old,new]]
for name,arrays in [('boxes',boxes),('scores',scores)]:
    diff=np.abs(arrays[0]-arrays[1])
    rowmax=diff.reshape(len(a),-1).max(1)
    result[name]=dict(shape=list(diff.shape),max=float(diff.max()),mean=float(diff.mean()),row_max_quantiles=np.quantile(rowmax,[0,.25,.5,.75,.9,.99,1]).tolist(),exact_rows=int((rowmax==0).sum()),rows_below_1e5=int((rowmax<1e-5).sum()),first10_max=rowmax[:10].tolist())
for mode in ['bbs','bbf']:
    changed=[i for i,(x,y) in enumerate(zip(a,b)) if x[mode]['query']!=y[mode]['query']]
    result['modes'][mode]={'query_changes':len(changed),'changed_first20':[a[i]['row_id'] for i in changed[:20]],'selected_box_equal':sum(x[mode]['box']==y[mode]['box'] for x,y in zip(a,b))}
print(json.dumps(result))
