import hashlib,json,os,py_compile,sys,time
from pathlib import Path
started=time.time();root=Path(__file__).resolve().parent;m=json.loads((root/'input_manifest.json').read_text())
source=Path(m['model_source']);os.chdir(str(source));sys.path.insert(0,str(source))
from scripts.run_nr3d_view_pair_role import file_sha
assert os.environ['CUDA_VISIBLE_DEVICES']=='' and sys.prefix==m['python_prefix']
assert file_sha(source/'g0_source_manifest.json')==m['source_manifest_sha256']
source_files=json.loads((source/'g0_source_manifest.json').read_text())['files']
for name,digest in source_files.items():assert file_sha(source/name)==digest,name
for name,metadata in m['data_files'].items():assert file_sha(Path(name))==metadata['sha256'],name
assert file_sha(Path(m['checkpoint']))==m['checkpoint_sha256']
assert file_sha(Path(m['sparse_preflight_receipt']))==m['sparse_preflight_receipt_sha256']
assert file_sha(Path(m['baseline_reference']))==m['baseline_reference_sha256']
for name,digest in m['runtime_receipts'].items():assert file_sha(Path(name))==digest,name
for name,digest in m['files'].items():
 assert file_sha(root/name)==digest,name
 if name.endswith('.py'):py_compile.compile(str(root/name),doraise=True)
import scripts
scripts.__path__=[str(root/'scripts')]+list(scripts.__path__)
import torch,spconv,cumm,pytest
assert torch.__version__=='1.10.2+cu111' and spconv.__version__=='2.3.6' and cumm.__version__=='0.4.11'
status=pytest.main(['-q','-p','no:cacheprovider',str(root/'tests/test_nr3d_sparse_point_pair_summary.py'),
 str(root/'tests/test_nr3d_point_voxel_mapping.py'),'--junitxml='+str(root/'tests.xml')])
assert status==0
import xml.etree.ElementTree as ET
tests=[{'name':c.attrib['name'],'passed':not list(c)} for c in ET.parse(str(root/'tests.xml')).getroot().findall('.//testcase')]
assert len(tests)==8 and all(row['passed'] for row in tests)
from scripts.nr3d_sparse_point_memory import SparsePointSuperpointResidual
addon=SparsePointSuperpointResidual()
new_shapes={name:list(p.shape) for name,p in addon.named_parameters()}
checkpoint=torch.load(m['checkpoint'],map_location='cpu')
shared={name:checkpoint['model']['module.'+name] for name in m['shared_parameter_names']}
assert len(shared)==16 and sum(t.numel() for t in shared.values())==1348960
assert len(new_shapes)==17 and sum(p.numel() for p in addon.parameters())==267936
result={'schema':'mcln-sparse-point-pair-cpu-v1','status':'pass','tests':tests,'pytest_exit':int(status),
 'python':sys.version,'prefix':sys.prefix,'torch':torch.__version__,'spconv':spconv.__version__,
 'runner_py37_compile':True,'source_file_count':len(source_files),'data_file_count':len(m['data_files']),
 'shared_parameter_shapes':{name:list(t.shape) for name,t in shared.items()},'sparse_parameter_shapes':new_shapes,
 'trainable_parameters':{'native':1348960,'sparse':1616896},'trainable_tensors':{'native':16,'sparse':33},
 'manifest_sha256':file_sha(root/'input_manifest.json'),'gpu_forwards':0,'optimizer_steps':0,
 'elapsed_seconds':time.time()-started}
with (root/'cpu_receipt.json').open('x') as f:json.dump(result,f,indent=2,sort_keys=True);f.write('\n')
print(json.dumps(result),flush=True)
