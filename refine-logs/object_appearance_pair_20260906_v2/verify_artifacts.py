import hashlib,json,torch
from pathlib import Path
root=Path('/root/autodl-tmp/mcln_object_appearance_pair_20260906_v2')
assert (root/'controller.exit').read_text().strip()=='0'
receipt_raw=(root/'receipt.json').read_bytes();r=json.loads(receipt_raw)
manifest_raw=(root/'input_manifest.json').read_bytes();m=json.loads(manifest_raw)
assert r['status']=='complete' and r['optimizer_steps_per_arm']==1024
assert r['text_mask_and_alpha_exactly_equal_to_start']
assert r['early_queries_and_sampling_exactly_equal_to_start']
assert r['frozen_parameters_and_buffers_unchanged'] and r['source_data_and_parent_checkpoint_unchanged']
assert hashlib.sha256(manifest_raw).hexdigest()==r['manifest_sha256']
for name in ['baseline_rows','terminal_rows','fit_point_batches']:
 assert hashlib.sha256((root/(name+'.json')).read_bytes()).hexdigest()==r[name+'_sha256']
assert hashlib.sha256(json.dumps(r['fit_order_ids']).encode()).hexdigest()==r['fit_order_sha256']
assert len(r['fit_order_ids'])==4096
for epoch in range(2):assert sorted(r['fit_order_ids'][epoch*2048:(epoch+1)*2048])==m['row_ids']['fit']
points=json.loads((root/'fit_point_batches.json').read_text());assert len(points)==1024
for index,batch in enumerate(points):
 assert batch['step']==index+1 and batch['row_ids']==r['fit_order_ids'][index*4:(index+1)*4]
 assert len(batch['point_tensor_sha256'])==64
 int(batch['point_tensor_sha256'],16)
assert hashlib.sha256(Path(m['native_preflight_receipt']).read_bytes()).hexdigest()==m['native_preflight_receipt_sha256']
parent_path=Path(m['checkpoint'])
assert hashlib.sha256(parent_path.read_bytes()).hexdigest()==m['checkpoint_sha256']
parent=torch.load(str(parent_path),map_location='cpu')['model']
initial={name[7:]:value for name,value in parent.items() if name[7:].startswith(('decoder.5.cross_d.','decoder.5.norm_d.'))}
expected={name:tuple(value.shape) for name,value in initial.items()}
assert len(expected)==6
del parent
extra={'appearance.point_encoder.0.weight':(64,6),'appearance.point_encoder.0.bias':(64,),
 'appearance.point_encoder.2.weight':(64,64),'appearance.point_encoder.2.bias':(64,),
 'appearance.output.weight':(288,128)}
result={}
for arm in ['native','appearance']:
 meta=r['artifacts'][arm];path=Path(meta['path']);raw=path.read_bytes()
 assert len(raw)==meta['bytes'] and hashlib.sha256(raw).hexdigest()==meta['sha256']
 value=torch.load(str(path),map_location='cpu')
 assert value['arm']==arm and value['steps']==1024
 assert value['parent_checkpoint_sha256']==m['checkpoint_sha256']
 weights=value['object_attention_state'];shapes=dict(expected)
 if arm=='appearance':shapes.update(extra)
 assert set(weights)==set(shapes)
 assert set(r['changed_parameter_names'][arm])==set(weights)
 shared_delta={name:float((weights[name]-start).abs().max()) for name,start in initial.items()}
 assert all(value>0 for value in shared_delta.values())
 assert all(tuple(weight.shape)==shapes[name] and torch.isfinite(weight).all() for name,weight in weights.items())
 assert sum(weight.numel() for weight in weights.values())=={'native':333504,'appearance':374976}[arm]
 optimizer=value['optimizer'];assert len(optimizer['param_groups'])==1
 group=optimizer['param_groups'][0]
 assert group['lr']==1e-5 and group['weight_decay']==.0005
 assert len(optimizer['state'])==len(group['params'])==len(weights)
 for pid,weight in zip(group['params'],weights.values()):
  state=optimizer['state'][pid];assert float(state['step'])==1024
  for key in ['exp_avg','exp_avg_sq']:assert state[key].shape==weight.shape and torch.isfinite(state[key]).all()
 result[arm]={'bytes':len(raw),'sha256':meta['sha256'],'parameter_tensors':len(weights),
  'parameters':sum(weight.numel() for weight in weights.values()),'optimizer_steps':1024,
  'shared_parameter_max_abs_change_from_parent':shared_delta,
  'weight_norms':{name:float(weight.norm()) for name,weight in weights.items()}}
 if arm=='appearance':assert weights['appearance.output.weight'].norm()>0
print(json.dumps({'schema':'mcln-object-appearance-completed-artifact-verification-v1','status':'pass',
 'receipt_sha256':hashlib.sha256(receipt_raw).hexdigest(),'artifacts':result,'fit_order_complete_twice':True,
 'recorded_fit_point_batches':1024,'gpu_forwards':0,'optimizer_updates':0,'protected_artifacts_modified':False}))
