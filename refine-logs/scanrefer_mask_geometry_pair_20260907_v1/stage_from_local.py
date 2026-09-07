import ast, hashlib, json, os, shlex
from pathlib import Path
import paramiko

repo=Path('C:/Users/gb/.codex_mcln_g0_20260905')
local=repo/'refine-logs/scanrefer_mask_geometry_pair_20260907_v1'
remote='/root/autodl-tmp/mcln_scanrefer_mask_geometry_pair_20260907_v1'
local.mkdir(exist_ok=False); (local/'scripts').mkdir(); (local/'tests').mkdir()
base=json.loads((repo/'refine-logs/scanrefer_mask_geometry_training_probe_20260907_v1/input_manifest.json').read_bytes())
names=['scripts/run_scanrefer_mask_geometry_pair.py','scripts/audit_scanrefer_mask_geometry_pair.py',
       'scripts/native_mask_geometry_supervision.py','scripts/prototype_probability_geometry.py',
       'scripts/mask_geometry_pair_support.py','scripts/native_teacher_box_transfer.py',
       'scripts/scanrefer_joint_readout.py','scripts/scanrefer_rec_evaluation.py',
       'scripts/scanrefer_data_contract.py','scripts/audit_scanrefer_joint_readout_pair.py',
       'tests/test_mask_geometry_pair_support.py','tests/test_native_mask_geometry_supervision.py']
files={}
for name in names:
    raw=(repo/name).read_bytes(); ast.parse(raw); (local/name).write_bytes(raw)
    files[name]=hashlib.sha256(raw).hexdigest()
(local/'scripts/__init__.py').write_bytes(b''); files['scripts/__init__.py']=hashlib.sha256(b'').hexdigest()
manifest={k:base[k] for k in ['model_source','source_manifest_sha256','artifacts','data_root','train_superpoint_files',
                            'split_salt','split_protocol','split_protocol_sha256','environment_reuse']}
probe=repo/'refine-logs/scanrefer_mask_geometry_training_probe_20260907_v1/receipt.json'
assert json.loads(probe.read_bytes())['status']=='pass'
manifest.update(schema='mcln-scanrefer-mask-geometry-gt-training-input-v1',
    core_learning_rate=1e-6, mask_geometry_loss_weight=1., weight_decay=.0005, clip_norm=.1,
    files=files, native_probe_receipt='/root/autodl-tmp/mcln_scanrefer_mask_geometry_training_probe_20260907_v1/receipt.json',
    native_probe_receipt_sha256=hashlib.sha256(probe.read_bytes()).hexdigest(),
    epochs=1,batch_size=12,steps_per_arm=2482,fit_rows=29778,holdout_rows=6887,formal_rows=0,
    readouts_frozen=True, model_mode='eval;84 allowed parameter tensors;all other parameters and buffers frozen',
    candidate_predeclared='native_gt_mask_geometry',control='native_gt',
    save_policy='84 core parameter delta plus actual optimizer;combine with protected E71 for1144-state reconstruction',
    selection='Existing GT-free native decision and frozen V99 system;record separately',
    module_gate='Candidate system REC025/050 no lower than identical baseline and native_gt;then fixed formal ScanRefer',
    formal_rec_hits_floor=[5572,4797],formal_rec_stretch=[59.,51.],formal_mask_floor=[58.70,50.70,44.72],
    nr3d_rec_floor=[59.82,51.38],sr3d_rec_floor=[68.43,57.30],nr3d_sr3d_mask_gate=False,
    purpose='Direct current-root Mask extent GT in training;actual hard geometry and frozen V99 remain the deployment output',
    auxiliary_geometry={'probability':'sigmoid','quantile':.005,'weight':1.,'matching':'current native Hungarian root index0'},
    hard_geometry_diagnostic={'source':'fused','threshold':0.,'quantile':.005,'min_points':5,'max_point_fraction':.5})
(local/'input_manifest.json').write_bytes((json.dumps(manifest,sort_keys=True,indent=2)+'\n').encode())
controller='''#!/usr/bin/env bash
set -u
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false
cd {remote}
flock -n /root/autodl-tmp/mcln_v99_backbone_gpu0.lock /root/miniconda3/envs/bdetr/bin/python -u scripts/run_scanrefer_mask_geometry_pair.py --manifest {remote}/input_manifest.json
status=$?
printf '%s\\n' "$status" > training.exit
if [ "$status" -eq 0 ]; then
  CUDA_VISIBLE_DEVICES= /root/miniconda3/envs/bdetr/bin/python -u scripts/audit_scanrefer_mask_geometry_pair.py --manifest {remote}/input_manifest.json
  status=$?
fi
printf '%s\\n' "$status" > controller.exit
exit "$status"
'''.format(remote=remote)
(local/'controller.sh').write_bytes(controller.encode())
c=paramiko.SSHClient(); c.load_system_host_keys(); c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect('region-9.autodl.pro',port=33476,username='root',password=os.environ['MCLN_SSH_PASSWORD'],timeout=30)
s=c.open_sftp(); s.mkdir(remote); s.mkdir(remote+'/scripts'); s.mkdir(remote+'/tests')
for p in local.rglob('*'):
    if p.is_file(): s.put(str(p),remote+'/'+p.relative_to(local).as_posix())
command='cd '+shlex.quote(remote)+' && CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH='+shlex.quote(remote+':'+manifest['model_source'])+' /root/miniconda3/envs/bdetr/bin/python -m pytest -q tests/test_mask_geometry_pair_support.py tests/test_native_mask_geometry_supervision.py'
_,o,e=c.exec_command(command); output=o.read().decode()+e.read().decode(); code=o.channel.recv_exit_status()
(local/'cpu_tests.txt').write_bytes(output.encode()); print(output,flush=True); assert code==0
check="import ast,json;from pathlib import Path;p=Path("+repr(remote)+");m=json.loads((p/'input_manifest.json').read_text());[ast.parse((p/n).read_text()) for n in m['files'] if n.endswith('.py')];print('Frozen scripts parse under original Python3.7')"
_,o,e=c.exec_command('/root/miniconda3/envs/bdetr/bin/python -c '+shlex.quote(check)+' && bash -n '+shlex.quote(remote+'/controller.sh'))
result=o.read().decode()+e.read().decode(); assert o.channel.recv_exit_status()==0,result
(local/'original_environment_syntax.txt').write_bytes(result.encode()); print(result,flush=True)
(local/'stage_from_local.py').write_bytes(Path(__file__).read_bytes())
s.close(); c.close(); print('STAGED; NOT LAUNCHED',remote,flush=True)
