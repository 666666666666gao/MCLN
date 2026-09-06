import hashlib,json,os,py_compile,subprocess,sys,time
from pathlib import Path
directory=Path(sys.argv[1]);source=Path(sys.argv[2]);os.chdir(str(directory))
sys.path[:0]=[str(directory),str(source)]
files=json.loads((directory/'files.json').read_text())
for name,digest in files.items():
    assert hashlib.sha256((directory/name).read_bytes()).hexdigest()==digest,name
    py_compile.compile(str(directory/name),doraise=True)
source_raw=(source/'g0_source_manifest.json').read_bytes()
assert hashlib.sha256(source_raw).hexdigest()=='dcf333b0e1868a7eeaafaf7f0a7abdb664a34dda65966defc1ad244ce762b15d'
source_files=json.loads(source_raw)['files']
for name,digest in source_files.items():
    assert hashlib.sha256((source/name).read_bytes()).hexdigest()==digest,name
import scripts
scripts.__path__.append(str(source/'scripts'))
import torch,pytest
assert os.environ['CUDA_VISIBLE_DEVICES']=='' and torch.__version__=='1.10.2+cu111'
from scripts.nr3d_sparse_point_memory import SparsePointSuperpointResidual
addon=SparsePointSuperpointResidual()
assert len(addon.state_dict())==17 and sum(p.numel() for p in addon.parameters())==267936
class Results:
    def __init__(self): self.calls=[]
    def pytest_runtest_logreport(self,report):
        if report.when=='call': self.calls.append({'test':report.nodeid,'outcome':report.outcome})
results=Results();started=time.time()
status=pytest.main(['-q','--junitxml=tests.xml','tests/test_nr3d_sparse_formal_state.py',
    'tests/test_nr3d_sparse_native_formal.py','tests/test_nr3d_l1_native_formal_pair.py',
    'tests/test_nr3d_sparse_point_pair_summary.py'],plugins=[results])
help_result=subprocess.run([sys.executable,str(directory/'scripts/run_nr3d_sparse_native_formal.py'),'--help'],
    stdout=subprocess.PIPE,stderr=subprocess.STDOUT,universal_newlines=True)
receipt={'schema':'mcln-sparse-native-formal-cpu-v1','pytest_exit':int(status),'tests':results.calls,
    'help_exit':help_result.returncode,'help_output':help_result.stdout,'elapsed_seconds':time.time()-started,
    'python':sys.version,'torch':torch.__version__,'prefix':sys.prefix,'files':files,
    'source_file_count':len(source_files),'source_manifest_sha256':hashlib.sha256(source_raw).hexdigest(),
    'actual_sparse_parameter_tensors':17,'actual_sparse_parameters':267936,
    'fixtures_are_synthetic':True,'gpu_forwards':0,'optimizer_steps':0,'formal_evaluation_executed':False,
    'trained_endpoint_loaded':False,'live_training_files_modified':False}
with (directory/'receipt.json').open('x') as f: json.dump(receipt,f,indent=2,sort_keys=True);f.write('\n')
print(json.dumps(receipt),flush=True)
assert help_result.returncode==0
sys.exit(int(status))
