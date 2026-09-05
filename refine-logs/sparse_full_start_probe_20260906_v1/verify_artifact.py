import hashlib,json,torch
from pathlib import Path
p=Path('/root/autodl-tmp/mcln_sparse_full_start_probe_20260906_v1')
r=json.loads((p/'receipt.json').read_text())
assert (p/'controller.exit').read_text().strip()=='0'
raw=(p/'shared_gradients.pt').read_bytes()
assert hashlib.sha256(raw).hexdigest()==r['gradient_artifact_sha256']
a=torch.load(str(p/'shared_gradients.pt'),map_location='cpu')
assert a['manifest_sha256']==r['manifest_sha256']
result={}
for key,record in r['comparisons'].items():
 first,second=key.split('__');parameters={}
 for name,v in a['gradients'][first].items():
  current=a['gradients'][second][name]
  delta=current.double()-v.double()
  old=torch.allclose(current,v,atol=1e-6,rtol=1e-5)
  exact=a['gpu_norms'][first][name]==a['gpu_norms'][second][name]
  maximum=float(delta.abs().max());relative=float(delta.norm()/v.double().norm())
  assert maximum==record['parameters'][name]['max_abs_difference']
  assert relative==record['parameters'][name]['relative_l2_difference']
  assert old==record['parameters'][name]['allclose_atol1e6_rtol1e5']
  assert exact==record['parameters'][name]['gpu_norm_exact']
  parameters[name]={'max_abs':maximum,'relative_l2':relative,
   'old_allclose':old,'gpu_norm_exact':exact,
   'allclose_atol1e5_rtol1e4':torch.allclose(current,v,atol=1e-5,rtol=1e-4)}
 result[key]=parameters
perturbed={name:torch.allclose(v*1.01,v,atol=1e-5,rtol=1e-4)
 for name,v in a['gradients']['native_check'].items()}
assert not any(perturbed.values())
out={'status':'pass','gradient_artifact_sha256':r['gradient_artifact_sha256'],'bytes':len(raw),
 'comparisons':result,'one_percent_perturbation_rejected_all16':True,
 'candidate_gradient_atol':1e-5,'candidate_gradient_rtol':1e-4,
 'all_recorded_pairs_pass_candidate_tolerance':all(v['allclose_atol1e5_rtol1e4'] for q in result.values() for v in q.values()),
 'gpu_forwards':0,'optimizer_steps':0,'underlying_cuda_kernel_identified':False}
with (p/'artifact_verification.json').open('x') as f:json.dump(out,f,indent=2,sort_keys=True)
print(json.dumps({'candidate_tolerance_pass':out['all_recorded_pairs_pass_candidate_tolerance'],
 'one_percent_perturbation_rejected_all16':True,'artifact_sha256':out['gradient_artifact_sha256'],
 'comparisons':{k:{'max_abs':max(v['max_abs'] for v in vals.values()),
 'max_relative_l2':max(v['relative_l2'] for v in vals.values()),
 'old_allclose':all(v['old_allclose'] for v in vals.values()),
 'candidate_allclose':all(v['allclose_atol1e5_rtol1e4'] for v in vals.values())} for k,vals in result.items()}}))
