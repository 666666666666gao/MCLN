import hashlib,json,torch
from pathlib import Path
p=Path('/root/autodl-tmp/mcln_sparse_warmup_regression_20260906_v1')
r=json.loads((p/'receipt.json').read_text())
raw=(p/'shared_gradients.pt').read_bytes()
assert hashlib.sha256(raw).hexdigest()==r['gradient_artifact_sha256']
a=torch.load(str(p/'shared_gradients.pt'),map_location='cpu')
assert a['manifest_sha256']==r['manifest_sha256']
reference=a['gradients']['plain_A_repeat'];result={}
for label in ['plain_A_first','sparse_zero_B']:
 current=a['gradients'][label]
 comparisons={name:{'max_abs':float((current[name].double()-v.double()).abs().max()),
   'exact':torch.equal(current[name],v),'allclose':torch.allclose(current[name],v,atol=1e-6,rtol=1e-5),
   'gpu_norm_exact':a['gpu_norms'][label][name]==a['gpu_norms']['plain_A_repeat'][name]}
  for name,v in reference.items()}
 result[label]=comparisons
out={'gradient_artifact_sha256':r['gradient_artifact_sha256'],'bytes':len(raw),
 'reference':'plain_A_repeat','comparisons':result,'gpu_forwards':0,'optimizer_steps':0}
with (p/'artifact_verification.json').open('x') as f:json.dump(out,f,indent=2,sort_keys=True)
print(json.dumps({label:{'max_abs':max(v['max_abs'] for v in items.values()),
 'allclose':all(v['allclose'] for v in items.values()),'all_gpu_norms_exact':all(v['gpu_norm_exact'] for v in items.values())}
 for label,items in result.items()}))
