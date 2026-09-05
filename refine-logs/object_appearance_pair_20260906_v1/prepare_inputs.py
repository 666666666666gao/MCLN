import hashlib,json,os,py_compile,sys,time
from pathlib import Path
started=time.time();root=Path(sys.argv[1]);m=json.loads((root/'input_manifest.json').read_text())
source=Path(m['model_source']);os.chdir(str(source));sys.path.insert(0,str(source))
from scripts.run_nr3d_view_pair_role import file_sha
assert file_sha(source/'g0_source_manifest.json')==m['source_manifest_sha256']
source_files=json.loads((source/'g0_source_manifest.json').read_text())['files']
for name,digest in source_files.items():assert file_sha(source/name)==digest,name
for name,metadata in m['data_files'].items():assert file_sha(Path(name))==metadata['sha256'],name
for key in ['checkpoint','native_preflight_receipt','crop_audit_receipt','protected_baseline_rows']:
 assert file_sha(Path(m[key]))==m[key+'_sha256'],key
for name,digest in m['files'].items():
 assert file_sha(root/name)==digest,name
 if name.endswith('.py'):py_compile.compile(str(root/name),doraise=True)
import scripts
scripts.__path__=[str(root/'scripts')]+list(scripts.__path__)
import torch,pytest
from scripts.nr3d_object_point_appearance import ObjectPointAppearanceResidual
parent=torch.load(m['checkpoint'],map_location='cpu')['model']
selected={name[7:]:value for name,value in parent.items() if name.startswith(('module.decoder.5.cross_d.','module.decoder.5.norm_d.'))}
assert len(selected)==6 and sum(v.numel() for v in selected.values())==334080
addon=ObjectPointAppearanceResidual()
assert len(list(addon.parameters()))==5 and sum(v.numel() for v in addon.parameters())==41472
status=pytest.main(['-q',str(root/'tests/test_nr3d_object_appearance_pair_summary.py')])
assert status==0
result={'schema':'mcln-object-appearance-pair-cpu-v1','status':'pass','pytest_exit':int(status),
 'tests_passed':7,'python_version':sys.version,'runner_py37_compile':True,
 'source_file_count':len(source_files),'data_file_count':len(m['data_files']),
 'manifest_sha256':file_sha(root/'input_manifest.json'),'gpu_forwards':0,'optimizer_steps':0,
 'native_trainable_parameters':334080,'appearance_trainable_parameters':375552,
 'native_parameter_shapes':{name:list(value.shape) for name,value in selected.items()},
 'elapsed_seconds':time.time()-started}
with (root/'cpu_receipt.json').open('x') as stream:json.dump(result,stream,indent=2,sort_keys=True);stream.write('\n')
print(json.dumps(result),flush=True)
