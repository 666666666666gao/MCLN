import hashlib,json,os,subprocess,sys,urllib.request
from pathlib import Path
import pkg_resources
root=Path(__file__).parent
import ssl
ca=Path(os.environ['SSL_CERT_FILE'])
assert hashlib.sha256(ca.read_bytes()).hexdigest()=='8e9482d461319198d2c5758d8ad29a1fb9dc0bc6850a24c57ffc83f6a8082cab'
context=ssl.create_default_context()
assert context.check_hostname and context.verify_mode==ssl.CERT_REQUIRED
before={d.key:d.version for d in pkg_resources.working_set}
metadata=json.loads((root/'wheel_metadata.json').read_text())
packages=[]
for package in metadata['packages']:
 assert len(package['py37_linux_x86_64_wheels'])==1
 wheel=package['py37_linux_x86_64_wheels'][0]
 path=root/wheel['filename']
 with urllib.request.urlopen(wheel['url'],timeout=120) as response,path.open('xb') as stream:
  for block in iter(lambda:response.read(1024*1024),b''):stream.write(block)
 raw=path.read_bytes()
 assert len(raw)==wheel['bytes'] and hashlib.sha256(raw).hexdigest()==wheel['sha256']
 packages.append(str(path))
 print('WHEEL VERIFIED '+wheel['filename'],flush=True)
venv=root/'venv'
subprocess.run([sys.executable,'-m','venv','--system-site-packages',str(venv)],check=True)
python=str(venv/'bin/python')
subprocess.run([python,'-m','pip','install','--no-input','--disable-pip-version-check','--index-url','https://pypi.org/simple']+packages,check=True)
check="""import json,sys,pkg_resources
import numpy,torch,transformers
import spconv.pytorch as sparse
import spconv,cumm
assert torch.__version__=='1.10.2+cu111' and numpy.__version__=='1.21.5' and transformers.__version__=='4.17.0'
assert spconv.__version__=='2.3.6' and cumm.__version__=='0.4.11'
assert sys.prefix==sys.argv[1]
result={'python':sys.version,'prefix':sys.prefix,'torch':torch.__version__,'numpy':numpy.__version__,
 'transformers':transformers.__version__,'spconv':spconv.__version__,'cumm':cumm.__version__,
 'torch_module_path':torch.__file__,'spconv_module_path':spconv.__file__,
 'installed_packages':{d.key:d.version for d in pkg_resources.working_set},
 'sparse_conv_class_available':sparse.SubMConv3d is not None}
print('SPARSE_IMPORT_RECEIPT '+json.dumps(result))
"""
result=subprocess.run([python,'-c',check,str(venv)],capture_output=True,text=True,check=True)
print(result.stdout,flush=True)
print(result.stderr,flush=True)
value=json.loads(next(line[len('SPARSE_IMPORT_RECEIPT '):] for line in result.stdout.splitlines() if line.startswith('SPARSE_IMPORT_RECEIPT ')))
basecheck=subprocess.run([sys.executable,'-c','import json,pkg_resources;print(json.dumps({d.key:d.version for d in pkg_resources.working_set}))'],capture_output=True,text=True,check=True)
assert json.loads(basecheck.stdout)==before
value.update(schema='mcln-sparse-runtime-preparation-v1',status='pass',base_package_inventory_unchanged=True,
 verified_download_ca_sha256=hashlib.sha256(ca.read_bytes()).hexdigest(),
 base_package_inventory_sha256=hashlib.sha256(json.dumps(before,sort_keys=True).encode()).hexdigest(),
 downloaded_wheels_sha256={Path(path).name:hashlib.sha256(Path(path).read_bytes()).hexdigest() for path in packages},
 gpu_forwards=0,native_model_forwards=0,optimizer_steps=0,sparse_kernel_runtime_tested=False)
with (root/'receipt.json').open('x') as stream:json.dump(value,stream,indent=2,sort_keys=True);stream.write('\n')
print('SPARSE RUNTIME PREP COMPLETE',json.dumps({k:v for k,v in value.items() if k!='installed_packages'}),flush=True)
