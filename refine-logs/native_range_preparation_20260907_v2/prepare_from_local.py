import hashlib
import json
import os
from pathlib import Path
import shlex

import paramiko


repo = Path('C:/Users/gb/.codex_mcln_g0_20260905')
local = repo / 'refine-logs/native_range_preparation_20260907_v2'
remote = '/root/autodl-tmp/mcln_native_range_preparation_20260907_v2'
old = repo / 'refine-logs/candidate_local_native_preparation_20260906_v1'
inputs = repo / 'refine-logs/native_local_preflight_preparation_20260906_v2'


def replace_once(text, before, after):
    assert text.count(before) == 1, before
    return text.replace(before, after)


entry = (repo / 'scripts/run_native_candidate_local_preflight.py').read_text()
entry = entry.replace('native-local-preflight', 'native-range-preflight')
entry = entry.replace('NATIVE LOCAL PREFLIGHT', 'NATIVE RANGE PREFLIGHT')
entry = replace_once(entry, "assert formal['schema'] == 'mcln-scanrefer-local-visual-official-v2'",
    "assert formal['schema'] == 'mcln-scanrefer-range-official-v1'\n"
    "    assert manifest['candidate_local_visual_variant'] == 'extent'\n"
    "    assert file_sha(manifest['scan_formal_audit']) == manifest['scan_formal_audit_sha256']\n"
    "    audit = json.loads(Path(manifest['scan_formal_audit']).read_text())\n"
    "    assert audit['schema'] == 'mcln-scanrefer-range-official-audit-v1'\n"
    "    assert audit['integrity_pass'] and audit['formal_rows'] == 9508\n"
    "    assert audit['receipt_sha256'] == manifest['scan_formal_receipt_sha256']\n"
    "    assert audit['promotion'] == formal['promotion']")
entry = replace_once(entry,
    "assert annotation['source_manifest_sha256'] == manifest['source_manifest_sha256']",
    "assert annotation['source_manifest_sha256'] == manifest['annotation_source_manifest_sha256']")
entry = replace_once(entry,
    "sys.argv = ['native-range-preflight'] + argv + ['--use_candidate_local_visual']",
    "sys.argv = ['native-range-preflight'] + argv + ['--use_candidate_local_visual',\n"
    "            '--candidate_local_visual_variant', manifest['candidate_local_visual_variant']]")
entry = replace_once(entry, "model = TrainTester.get_model(args).cuda()",
    "model = TrainTester.get_model(args).cuda()\n"
    "        assert type(model.decoder[-1].local_visual).__name__ == 'CandidateRangeVisual'\n"
    "        assert model.decoder[-1].local_visual.sampling == 'extent'")
entry = replace_once(entry, "'schema': 'mcln-native-candidate-local-gpu-preflight-v2'",
    "'schema': 'mcln-native-candidate-range-gpu-preflight-v1',\n"
    "              'candidate_local_visual_variant': manifest['candidate_local_visual_variant']")
(repo / 'scripts/run_native_candidate_range_preflight.py').write_text(entry, encoding='utf-8', newline='\n')

check = (old / 'check_native.py').read_text()
check = replace_once(check, "+ ['--use_candidate_local_visual']",
    "+ ['--use_candidate_local_visual', '--candidate_local_visual_variant', 'extent']")
check = replace_once(check, "reader = model.decoder[-1].local_visual",
    "reader = model.decoder[-1].local_visual\n"
    "    assert type(reader).__name__ == 'CandidateRangeVisual' and reader.sampling == 'extent'")
check = replace_once(check, "'schema': 'mcln-candidate-local-native-preparation-v1'",
    "'schema': 'mcln-candidate-range-native-preparation-v1', 'reader_variant': 'extent'")
check = replace_once(check, "assert not torch.cuda.is_initialized()",
    "import importlib.util\n"
    "probe = importlib.util.spec_from_file_location('range_preflight_entry', str(root / 'run_native_candidate_range_preflight.py'))\n"
    "probe_entry = importlib.util.module_from_spec(probe)\n"
    "probe.loader.exec_module(probe_entry)\n"
    "data_root = probe_entry.verify_probe_data_inputs(root)\n"
    "assert data_root == '/root/autodl-tmp/DATA_ROOT_mcln_meshsp/'\n"
    "assert not torch.cuda.is_initialized()")
check = replace_once(check, "'gpu_forwards': 0, 'real_dataset_rows': 0,",
    "'data_root': data_root, 'superpoint_files_verified': {'train': 1201, 'val': 312},\n"
    "    'gpu_forwards': 0, 'real_dataset_rows': 0,")

prepare = (old / 'prepare_source.py').read_text()
prepare = replace_once(prepare, "parent / 'local_visual_source_manifest.json'", "parent / 'native_source_manifest.json'")
prepare = replace_once(prepare, 'assert len(files) == 614', 'assert len(files) == 616')
prepare = replace_once(prepare, 'assert len(files) == 616\nfor name in files:', 'assert len(files) == 618\nfor name in files:')
prepare = replace_once(prepare, 'Prepared616sourcefiles; original Scan source unchanged', 'Prepared618sourcefiles; original native and live Scan sources unchanged')

overlays = ['main_utils.py', 'train_dist_mod.py', 'models/candidate_local_visual_training.py',
            'models/candidate_range_visual.py', 'tests/test_candidate_local_visual_training.py',
            'tests/test_candidate_range_visual.py']
files = {'overlays/' + name: (repo / name).read_bytes() for name in overlays}
files.update({name: (inputs / name).read_bytes() for name in ['nr_contract.json', 'annotation_receipt.json',
    'preflight_rows.json', 'data_inputs.json']})
files['run_native_candidate_range_preflight.py'] = entry.encode()
files['check_native.py'] = check.encode()
files['prepare_source.py'] = prepare.encode()
files['prepare_from_local.py'] = Path(__file__).read_bytes()
expected = json.loads((old / 'expected.json').read_text())
expected['parent_source'] = '/root/autodl-tmp/mcln_candidate_local_native_preparation_20260906_v1/model_source'
expected['parent_manifest_sha256'] = '4af68cad46b52c9e250de17872a193485d75529378db983ad037779232e500fc'
expected['overlay_files'] = {name: hashlib.sha256(files['overlays/' + name]).hexdigest() for name in overlays}
expected['prepared_files'] = {name: hashlib.sha256(raw).hexdigest() for name, raw in files.items()}
files['expected.json'] = (json.dumps(expected, indent=2, sort_keys=True) + '\n').encode()
controller = '\n'.join([
    '#!/usr/bin/env bash', 'set -u',
    "export CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false",
    'run_checks() {', 'set -e',
    '/root/miniconda3/envs/bdetr/bin/python ' + remote + '/prepare_source.py ' + remote,
    'cd ' + remote + '/model_source',
    '/root/miniconda3/envs/bdetr/bin/python -m pytest -q tests/test_candidate_local_visual_training.py tests/test_candidate_range_visual.py tests/test_mcln_training_groups.py tests/test_main_utils_source_choice_checkpoint.py',
    '/root/miniconda3/envs/bdetr/bin/python ' + remote + '/check_native.py ' + remote,
    '}', '(run_checks)', 'status=$?', 'printf "%s\\n" "$status" > ' + remote + '/controller.exit', 'exit "$status"', ''])
files['controller.sh'] = controller.encode()
local.mkdir()
client = paramiko.SSHClient()
client.load_system_host_keys()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect('region-9.autodl.pro', port=33476, username='root', password=os.environ['MCLN_SSH_PASSWORD'], timeout=30)
sftp = client.open_sftp()
sftp.mkdir(remote)
directories = {remote}
for name, raw in files.items():
    target = local / name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(raw)
    parts = name.split('/')[:-1]
    parent = remote
    for part in parts:
        parent += '/' + part
        if parent not in directories:
            sftp.mkdir(parent)
            directories.add(parent)
    with sftp.open(remote + '/' + name, 'wx') as stream:
        stream.write(raw)
command = 'screen -dmS mcln_native_range_cpu_prep_v2 bash -c ' + shlex.quote('exec bash ' + remote + '/controller.sh > ' + remote + '/run.log 2>&1')
_, output, error = client.exec_command(command, timeout=30)
assert output.channel.recv_exit_status() == 0, error.read().decode()
_, output, error = client.exec_command('screen -ls', timeout=30)
sessions = output.read().decode()
assert output.channel.recv_exit_status() == 0, error.read().decode()
matches = [line.strip() for line in sessions.splitlines() if '.mcln_native_range_cpu_prep_v2' in line]
assert len(matches) == 1, sessions
launch = {'screen_session': matches, 'remote_directory': remote,
          'expected_sha256': hashlib.sha256(files['expected.json']).hexdigest(),
          'gpu_visible_devices': '', 'scope': 'CPU source/model/checkpoint/optimizer preparation only; Scan promotion required for native GPU preflight.'}
raw = (json.dumps(launch, indent=2, sort_keys=True) + '\n').encode()
(local / 'launch.json').write_bytes(raw)
with sftp.open(remote + '/launch.json', 'wx') as stream:
    stream.write(raw)
sftp.close()
client.close()
print(json.dumps(launch), flush=True)
