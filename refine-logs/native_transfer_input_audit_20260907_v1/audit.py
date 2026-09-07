import datetime,hashlib,json,os,time
from pathlib import Path
import torch
assert os.environ['CUDA_VISIBLE_DEVICES']==''
torch.set_num_threads(1)
root=Path(__file__).parent
m=json.loads((root/'input_manifest.json').read_text())
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
 return h.hexdigest()
started=time.time()
assert sha(Path(m['training_directory'])/'input_manifest.json')==m['training_manifest_sha256']
states={}
descriptions={}
for label in ['scan_backbone','protected_nr']:
 item=m[label]
 path=Path(item['path'])
 assert path.stat().st_size==item['bytes'] and sha(path)==item['sha256']
 payload=torch.load(str(path),map_location='cpu')
 raw=payload['model']
 assert raw and all(name.startswith('module.') for name in raw)
 states[label]={name[7:]:value for name,value in raw.items()}
 descriptions[label]={'path':str(path),'sha256':item['sha256'],'tensor_count':len(raw),
  'backbone_tensor_count':sum(name.startswith('module.backbone_net.') for name in raw),
  'selector_tensor_names':[name for name in states[label] if name.startswith('source_choice_selector.')],
  'has_range_or_local_state':any('local_visual.' in name for name in raw),
  'raw_native_prefix':'module.'}
 del payload
scan,nr=states['scan_backbone'],states['protected_nr']
common=set(scan)&set(nr)
mismatch=[{'name':name,'scan_shape':list(scan[name].shape),'nr_shape':list(nr[name].shape),
 'scan_dtype':str(scan[name].dtype),'nr_dtype':str(nr[name].dtype)} for name in sorted(common)
 if scan[name].shape!=nr[name].shape or scan[name].dtype!=nr[name].dtype]
source=Path(m['scan_source'])
assert sha(source/'local_visual_source_manifest.json')==m['scan_source_manifest_sha256']
sm=json.loads((source/'local_visual_source_manifest.json').read_text())
prep=Path(m['native_preparation'])
nm=json.loads((prep/'model_source/native_source_manifest.json').read_text())
selected=['main_utils.py','train_dist_mod.py','src/joint_det_dataset.py','src/grounding_evaluator.py']
sources={}
for name in selected:
 assert sha(source/name)==sm['files'][name]
 assert sha(prep/'model_source'/name)==nm['files'][name]
 sources[name]={'scan_sha256':sm['files'][name],'native_sha256':nm['files'][name],
   'equal':sm['files'][name]==nm['files'][name]}
annotation=json.loads((prep/'annotation_receipt.json').read_text())
for name,item in annotation['annotations_and_split_files'].items():assert sha(name)==item['sha256'],name
protocols={dset:{part:{key:value for key,value in data.items() if not isinstance(value,(list,dict))}
 for part,data in annotation['protocols'][dset].items() if isinstance(data,dict)} for dset in ['nr3d','sr3d']}
result={'schema':'mcln-native-transfer-input-audit-receipt-v1','status':'complete',
 'time_cst':datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
 'elapsed_seconds':time.time()-started,'artifacts':descriptions,
 'only_scan_keys':sorted(set(scan)-set(nr)),'only_nr_keys':sorted(set(nr)-set(scan)),
 'common_shape_dtype_mismatches':mismatch,'identical_parameter_schema':set(scan)==set(nr) and not mismatch,
 'selected_runtime_source_files':sources,'native_annotation_protocol_counts':protocols,
 'annotations_and_splits_rehashed':len(annotation['annotations_and_split_files']),
 'gpu_forwards':0,'checkpoint_writes':0,'optimizer_steps':0,'formal_rows_evaluated':0,
 'future_trained_candidate_not_loaded':True,
 'limitations':'Actual protected model schemas and native data-file identities only;not an end-to-end transfer load,not GPUpreflight,not transfer performance;no promotion or native training authorization change.',
 'manifest_sha256':sha(root/'input_manifest.json'),'audit_script_sha256':sha(__file__)}
with (root/'receipt.json').open('x') as f:json.dump(result,f,indent=2,sort_keys=True,allow_nan=False)
print(json.dumps(result))
