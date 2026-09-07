import datetime,hashlib,json,os,sys,time
from pathlib import Path
root=Path('/root/autodl-tmp/mcln_mask_geometry_native_loss_cpu_20260907_v1'); source=Path('/root/autodl-tmp/mcln_native_range_preparation_20260907_v1/model_source'); training=Path('/root/autodl-tmp/mcln_scanrefer_mask_geometry_pair_20260907_v1')
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
