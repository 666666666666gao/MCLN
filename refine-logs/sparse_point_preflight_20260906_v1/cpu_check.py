import hashlib,json,os,py_compile,subprocess,sys,xml.etree.ElementTree as ET
from pathlib import Path
root=Path(__file__).resolve().parent
manifest=json.loads((root/'input_manifest.json').read_text())
def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
 return h.hexdigest()
assert os.environ['CUDA_VISIBLE_DEVICES']==''
assert sys.prefix==manifest['python_prefix']
for name,digest in manifest['files'].items():
 assert sha(root/name)==digest,name
 if name.endswith('.py'):py_compile.compile(str(root/name),doraise=True)
result=subprocess.run(['/root/miniconda3/envs/bdetr/bin/python','-m','pytest',
 'tests/test_nr3d_point_voxel_mapping.py','-q','-p','no:cacheprovider','--junitxml=tests.xml'],
 cwd=str(root),stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
(root/'pytest.txt').write_bytes(result.stdout)
print(result.stdout.decode(),flush=True)
assert result.returncode==0
tests=[{'test':c.attrib['name'],'outcome':'passed' if not list(c) else 'failed'}
 for c in ET.parse(str(root/'tests.xml')).getroot().findall('.//testcase')]
assert len(tests)==3 and all(row['outcome']=='passed' for row in tests)
source=Path(manifest['model_source'])
assert sha(source/'g0_source_manifest.json')==manifest['source_manifest_sha256']
source_files=json.loads((source/'g0_source_manifest.json').read_text())['files']
for name,digest in source_files.items():assert sha(source/name)==digest,name
for name,meta in manifest['data_files'].items():assert sha(Path(name))==meta['sha256'],name
assert sha(Path(manifest['checkpoint']))==manifest['checkpoint_sha256']
assert sha(Path(manifest['m3_receipt']))==manifest['m3_receipt_sha256']
for name,digest in manifest['runtime_receipts'].items():assert sha(Path(name))==digest,name
sys.path.insert(0,str(source))
import scripts
scripts.__path__=[str(root/'scripts')]+list(scripts.__path__)
import torch,spconv,cumm
assert torch.__version__=='1.10.2+cu111' and spconv.__version__=='2.3.6' and cumm.__version__=='0.4.11'
from scripts.nr3d_sparse_point_memory import SparsePointSuperpointResidual
addon=SparsePointSuperpointResidual()
parameters={name:{'shape':list(p.shape),'numel':p.numel()} for name,p in addon.named_parameters()}
assert addon.output.weight.count_nonzero()==0
receipt={'schema':'mcln-sparse-point-cpu-preflight-v1','status':'pass','tests':tests,
 'pytest_exit':result.returncode,'parameter_tensors':len(parameters),
 'parameter_count':sum(p['numel'] for p in parameters.values()),'parameters':parameters,
 'manifest_sha256':sha(root/'input_manifest.json'),'source_files_verified':len(source_files),
 'data_files_verified':len(manifest['data_files']),'parent_and_runtime_receipts_verified':True,
 'python':sys.version,'prefix':sys.prefix,'torch':torch.__version__,
 'spconv':spconv.__version__,'cumm':cumm.__version__,
 'pytest_python':'/root/miniconda3/envs/bdetr/bin/python',
 'gpu_forwards':0,'native_model_forwards':0,'optimizer_steps':0}
with (root/'cpu_receipt.json').open('x') as f:json.dump(receipt,f,indent=2,sort_keys=True)
print(json.dumps(receipt),flush=True)
