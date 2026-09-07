import datetime
import hashlib
import json
import os
from pathlib import Path
import shlex

import paramiko

repo = Path('C:/Users/gb/.codex_mcln_g0_20260905')
local = repo / 'refine-logs/mask_geometry_native_loss_cpu_20260907_v1'
remote = '/root/autodl-tmp/mcln_mask_geometry_native_loss_cpu_20260907_v1'
source = '/root/autodl-tmp/mcln_native_range_preparation_20260907_v1/model_source'
training = '/root/autodl-tmp/mcln_scanrefer_mask_geometry_pair_20260907_v1'
local.mkdir(exist_ok=False)
for directory in ['models', 'scripts', 'tests']:
    (local / directory).mkdir()
client = paramiko.SSHClient(); client.load_system_host_keys()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect('region-9.autodl.pro', port=33476, username='root',
               password=os.environ['MCLN_SSH_PASSWORD'], timeout=30)
sftp = client.open_sftp()
with sftp.open(source + '/native_source_manifest.json', 'rb') as stream:
    native_manifest = stream.read()
with sftp.open(source + '/models/__init__.py', 'rb') as stream:
    init = stream.read()
package = ('__path__ = [' + repr(remote + '/models') + ', ' + repr(source + '/models') + ']\n').encode() + init
(local / 'models/__init__.py').write_bytes(package)
(local / 'scripts/__init__.py').write_bytes(('__path__ = [' + repr(remote + '/scripts') + ', ' + repr(source + '/scripts') + ']\n').encode())
names = ['main_utils.py', 'models/losses.py', 'scripts/native_mask_geometry_supervision.py',
         'scripts/native_teacher_box_transfer.py', 'scripts/prototype_probability_geometry.py',
         'tests/test_native_mask_geometry_training.py', 'tests/test_native_mask_geometry_supervision.py',
         'tests/test_mcln_training_groups.py', 'tests/test_density_aware_target_box.py']
for name in names:
    (local / name).write_bytes((repo / name).read_bytes())
bootstrap = '''import datetime,hashlib,json,os,sys,time
from pathlib import Path
root=Path(REMOTE); source=Path(SOURCE); training=Path(TRAINING)
assert os.environ['CUDA_VISIBLE_DEVICES']==''
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
manifest=json.loads((root/'input_manifest.json').read_text())
assert sha(source/'native_source_manifest.json')==manifest['native_source_manifest_sha256']
for name,digest in json.loads((source/'native_source_manifest.json').read_text())['files'].items():
    assert sha(source/name)==digest,name
def check_running_source():
    assert sha(training/'input_manifest.json')==manifest['training_manifest_sha256']
    for name,digest in json.loads((training/'input_manifest.json').read_text())['files'].items():
        assert sha(training/name)==digest,name
check_running_source()
for name,digest in manifest['files'].items():assert sha(root/name)==digest,name
os.chdir(str(source)); sys.path[:0]=[str(root),str(source)]
import torch,pytest
torch.set_num_threads(1)
assert not torch.cuda.is_available()
started=time.time()
arguments=['-q','-p','no:cacheprovider',str(root/'tests/test_native_mask_geometry_training.py'),
str(root/'tests/test_native_mask_geometry_supervision.py'),
str(root/'tests/test_mcln_training_groups.py')+'::test_cli_defaults_expose_joint_mask_training_controls',
str(root/'tests/test_mcln_training_groups.py')+'::test_compute_loss_forwards_mask_and_consistency_scales',
str(root/'tests/test_density_aware_target_box.py')+'::test_default_off_integration_is_guarded_before_auxiliary_call']
status=pytest.main(arguments)
import main_utils,models.losses
assert Path(main_utils.__file__).resolve()==root/'main_utils.py'
assert Path(models.losses.__file__).resolve()==root/'models/losses.py'
check_running_source()
result={'schema':'mcln-native-mask-geometry-loss-cpu-v1','status':'pass' if status==0 else 'fail',
'time_cst':datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
'seconds':time.time()-started,'pytest_exit':int(status),'arguments':arguments,
'source_files_verified':len(json.loads((source/'native_source_manifest.json').read_text())['files']),
'current_patched_entrypoints_imported':True,'running_scan_source_unchanged':True,
'gpu_forwards':0,'optimizer_steps':0,'checkpoint_writes':0,'formal_rows':0,
'scope':'CPU native criterion integration with real Hungarian assignments and synthetic point/mask tensors; not dataset training',
'manifest_sha256':sha(root/'input_manifest.json')}
with (root/'receipt.json').open('x') as stream:json.dump(result,stream,indent=2,sort_keys=True)
print(json.dumps(result),flush=True)
raise SystemExit(status)
'''.replace('REMOTE', repr(remote)).replace('SOURCE', repr(source)).replace('TRAINING', repr(training))
(local / 'check_native_loss.py').write_bytes(bootstrap.encode())
files = {p.relative_to(local).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
         for p in local.rglob('*.py')}
manifest = {'schema': 'mcln-native-mask-geometry-loss-cpu-input-v1', 'files': files,
            'native_source': source, 'native_source_manifest_sha256': hashlib.sha256(native_manifest).hexdigest(),
            'training_manifest_sha256': '15f46411069a7172a55373c5c13075b22bcb4146d39251e9fca2c37ed5867eb3',
            'no_environment_install_or_rebuild': True, 'gpu_forwards': 0, 'optimizer_steps': 0}
(local / 'input_manifest.json').write_bytes((json.dumps(manifest, indent=2) + '\n').encode())
sftp.mkdir(remote)
for directory in ['models', 'scripts', 'tests']:
    sftp.mkdir(remote + '/' + directory)
for path in local.rglob('*'):
    if path.is_file():
        sftp.put(str(path), remote + '/' + path.relative_to(local).as_posix())
command = ('CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false '
           '/root/miniconda3/envs/bdetr/bin/python -u ' + shlex.quote(remote + '/check_native_loss.py') +
           ' > ' + shlex.quote(remote + '/cpu_tests.txt') + ' 2>&1')
_, output, error = client.exec_command(command, timeout=60)
status = output.channel.recv_exit_status()
for name in ['receipt.json', 'cpu_tests.txt']:
    size = sftp.stat(remote + '/' + name).st_size
    with sftp.open(remote + '/' + name, 'rb') as stream:
        stream.prefetch(file_size=size); raw = stream.read()
    (local / name).write_bytes(raw)
(local / 'stage_from_local.py').write_bytes(Path(__file__).read_bytes())
sftp.close(); client.close()
print((local / 'cpu_tests.txt').read_text(encoding='utf-8'), flush=True)
raise SystemExit(status)
