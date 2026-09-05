import hashlib,importlib.util,json,torch
from pathlib import Path
p=Path('/root/autodl-tmp/mcln_sparse_full_start_probe_20260906_v1')
root=Path('/root/autodl-tmp/mcln_sparse_gradient_regression_20260906_v1')
manifest=json.loads((root/'input_manifest.json').read_text())
for name,digest in manifest['files'].items():assert hashlib.sha256((root/name).read_bytes()).hexdigest()==digest
receipt=json.loads((p/'receipt.json').read_text())
raw=(p/'shared_gradients.pt').read_bytes()
assert hashlib.sha256(raw).hexdigest()==manifest['gradient_artifact_sha256']==receipt['gradient_artifact_sha256']
a=torch.load(str(p/'shared_gradients.pt'),map_location='cpu')
spec=importlib.util.spec_from_file_location('comparison',str(root/'nr3d_shared_gradient_check.py'))
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
comparisons={}
for pair in receipt['comparisons']:
 first,second=pair.split('__')
 comparisons[pair]={name:module.shared_gradient_comparison(value,a['gradients'][second][name])
  for name,value in a['gradients'][first].items()}
assert all(value['passed'] for pair in comparisons.values() for value in pair.values())
assert not all(value['elementwise_atol1e6_rtol1e5'] for pair in comparisons.values() for value in pair.values())
rejected={name:not module.shared_gradient_comparison(value,value*1.01)['passed']
 for name,value in a['gradients']['native_check'].items()}
assert len(rejected)==16 and all(rejected.values())
result={'status':'pass','manifest_sha256':hashlib.sha256((root/'input_manifest.json').read_bytes()).hexdigest(),
 'gradient_artifact_sha256':receipt['gradient_artifact_sha256'],'comparisons':comparisons,
 'all80_native_and_cross_arm_parameter_comparisons_pass':True,
 'old_elementwise_gate_fails_recorded_native_repeat':True,'one_percent_perturbation_rejected_all16':True,
 'relative_l2_limit':1e-4,'gpu_forwards':0,'optimizer_steps':0,'quality_gate_unchanged':True}
with (root/'receipt.json').open('x') as f:json.dump(result,f,indent=2,sort_keys=True)
print(json.dumps({k:v for k,v in result.items() if k!='comparisons'}))
