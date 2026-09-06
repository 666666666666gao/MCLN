import hashlib
import json
import os
from pathlib import Path
import shlex

import paramiko


repo = Path('C:/Users/gb/.codex_mcln_g0_20260905')
old = repo / 'refine-logs/native_local_preflight_preparation_20260906_v1'
local = repo / 'refine-logs/native_local_preflight_preparation_20260906_v2'
remote = '/root/autodl-tmp/mcln_native_local_preflight_preparation_20260906_v2'
source = '/root/autodl-tmp/mcln_candidate_local_native_preparation_20260906_v1/model_source'
files = {name: (old / name).read_bytes() for name in ['annotation_receipt.json', 'preflight_rows.json', 'nr_contract.json']}
files['old_cpu_receipt.json'] = (old / 'receipt.json').read_bytes()
files['data_inputs.json'] = (repo / 'refine-logs/scanrefer_local_visual_official_20260906_v3/data_inputs.json').read_bytes()
files['run_native_candidate_local_preflight.py'] = (repo / 'scripts/run_native_candidate_local_preflight.py').read_bytes()
check = (old / 'check_inputs.py').read_text()
old_block = """    argv=list(contract['eval_argv'])
    for key,value in [('--dataset',dset),('--test_dataset',dset),('--expected_eval_sample_count',str(annotation['protocols'][dset]['val']['total_rows']))]:
        argv[argv.index(key)+1]=value"""
assert check.count(old_block) == 1
check = check.replace(old_block, "    argv=entry.build_probe_argv(contract['eval_argv'],dset,annotation['protocols'][dset]['val']['total_rows'],data_root)")
check = check.replace("contract=json.loads((root/'nr_contract.json').read_text())",
                      "contract=json.loads((root/'nr_contract.json').read_text())\ndata_root=entry.verify_probe_data_inputs(root)\nold_cpu=json.loads((root/'old_cpu_receipt.json').read_text())")
check = check.replace('    assert args.use_color', '    assert args.data_root==data_root\n    assert args.use_color')
check = check.replace("'is_view_dep':batch['is_view_dep'].tolist(),",
                      "'superpoint_sha256':[hashlib.sha256(p.numpy().tobytes()).hexdigest() for p in inputs['superpoint']],\n                            'superpoint_counts':[int(p.unique().numel()) for p in inputs['superpoint']],\n                            'is_view_dep':batch['is_view_dep'].tolist(),")
needle = "    results[dset]={'unique_rows':16,'constructed_point_samples':32,'records':records}"
assert check.count(needle) == 1
replacement = """    for current,previous in zip(records,old_cpu['protocols'][dset]['records']):
        for key in ['phase','rows','sample_dataset','scan_ids','point_sha256','is_view_dep','gt_objects','input_objects','token_map_shape']:
            assert current[key]==previous[key],(dset,key)
    results[dset]={'unique_rows':16,'constructed_point_samples':32,'records':records,'old_root_sampled_points_and_labels_equal':True}"""
check = check.replace(needle, replacement)
check = check.replace("'schema':'mcln-native-local-real-input-cpu-preparation-v1'", "'schema':'mcln-native-local-real-input-cpu-preparation-v2'")
check = check.replace("'source_manifest_sha256':expected['source_manifest_sha256'],'files':expected['files'],",
                      "'data_root':data_root,'old_root_point_samples_verified_equal':64,\n        'source_manifest_sha256':expected['source_manifest_sha256'],'files':expected['files'],")
files['check_inputs.py'] = check.encode()
files['prepare.py'] = Path(__file__).read_bytes()
expected = {'model_source': source,
            'source_manifest_sha256': '4af68cad46b52c9e250de17872a193485d75529378db983ad037779232e500fc',
            'files': {name: hashlib.sha256(raw).hexdigest() for name, raw in files.items()}}
files['expected.json'] = (json.dumps(expected, indent=2, sort_keys=True) + '\n').encode()
local.mkdir()
client = paramiko.SSHClient()
client.load_system_host_keys()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect('region-9.autodl.pro', port=33476, username='root', password=os.environ['MCLN_SSH_PASSWORD'], timeout=30)
sftp = client.open_sftp()
sftp.mkdir(remote)
for name, raw in files.items():
    (local / name).write_bytes(raw)
    with sftp.open(remote + '/' + name, 'wx') as stream:
        stream.write(raw)
command = ("CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false PYTHONDONTWRITEBYTECODE=1"
           + ' /root/miniconda3/envs/bdetr/bin/python -u ' + shlex.quote(remote + '/check_inputs.py') + ' ' + shlex.quote(remote))
_, output, error = client.exec_command(command, timeout=900)
text = output.read().decode()
error_text = error.read().decode()
status = output.channel.recv_exit_status()
(local / 'run.txt').write_text(text + error_text, encoding='utf-8')
with sftp.open(remote + '/run.txt', 'wx') as stream:
    stream.write((text + error_text).encode())
print(text + error_text, flush=True)
assert status == 0
with sftp.open(remote + '/receipt.json', 'rb') as stream:
    raw = stream.read()
    (local / 'receipt.json').write_bytes(raw)
receipt = json.loads(raw)
assert receipt['status'] == 'actual_cpu_inputs_pass' and receipt['old_root_point_samples_verified_equal'] == 64
sftp.close()
client.close()
print(json.dumps({'receipt_sha256': hashlib.sha256(raw).hexdigest(), 'files': expected['files'],
                  'data_root': receipt['data_root'], 'point_samples': 64, 'gpu_forwards': 0}))
