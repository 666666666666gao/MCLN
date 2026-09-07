import datetime
import hashlib
import json
import os
from pathlib import Path
import shlex

import paramiko

repo = Path('C:/Users/gb/.codex_mcln_g0_20260905')
name = 'scanrefer_appearance_native_overlap_20260908_v1'
archive = repo/'refine-logs'/name
remote = '/root/autodl-tmp/mcln_'+name
training = '/root/autodl-tmp/mcln_scanrefer_object_appearance_pair_20260908_v1'
transfer = '/root/autodl-tmp/mcln_scanrefer_appearance_readout_transfer_preparation_20260908_v1'
source = '/root/autodl-tmp/mcln_scanrefer_object_appearance_native_20260908_v1/model_source'
data = '/root/autodl-tmp/DATA_ROOT_mcln_meshsp'
hash_bytes = lambda raw: hashlib.sha256(raw).hexdigest()
c = paramiko.SSHClient()
c.load_system_host_keys()
c.connect('region-9.autodl.pro', port=33476, username='root', password=os.environ['MCLN_SSH_PASSWORD'], timeout=30)
s = c.open_sftp()

def read(path):
    with s.open(path, 'rb') as stream:
        return stream.read()

assert read(training+'/native_queue.exit').strip() == b'0'
assert read(transfer+'/queue.exit').strip() == b'0'
annotation = data+'/ScanRefer/ScanRefer_filtered_train.json'
scene_list = data+'/ScanRefer/ScanRefer_filtered_train.txt'
source_files = {source+'/'+p: hash_bytes(read(source+'/'+p)) for p in ('src/visual_data_handlers.py', 'src/joint_det_dataset.py')}
inputs = {
    annotation: hash_bytes(read(annotation)), scene_list: hash_bytes(read(scene_list)),
    data+'/train_v3scans.pkl': '804415e82b740e92d4a8e3de4679575fe116cd1e27a9797e9da0977ad1c42be7',
    training+'/native_evaluation/rows.json': '4d8ecb693eafc8b4319350a100f0c0e78b16c27c7e36184bfae1d00c5855fc76',
    training+'/native_evaluation/receipt.json': '03b02f734b30659820191755ea8a4ca51778e24dab7c1c17bbf051557d9e9fb4',
    transfer+'/analysis/paired_rows.csv': '123d5e3037ba1824a25196c1cc567499c4fc1c756d1e9a4feb5c71afae37544a',
}
original = json.loads(read(training+'/input_manifest.json'))
inputs[original['split_protocol']] = original['split_protocol_sha256']
inputs.update(source_files)
files = {p: (repo/p).read_bytes() for p in (
    'scripts/analyze_scanrefer_appearance_native_overlap.py',
    'scripts/analyze_scanrefer_instance_overlap.py',
    'scripts/audit_scanrefer_stage_diagnostic.py',
    'tests/test_appearance_native_overlap.py',
    'tests/test_instance_overlap_diagnostic.py',
)}
files['scripts/__init__.py'] = b''
inputs.update({remote+'/'+p: hash_bytes(raw) for p, raw in files.items()})
manifest = {
    'schema': 'mcln-saved-native-appearance-overlap-v1', 'model_source': source,
    'annotations': annotation, 'annotation_scene_list': scene_list,
    'scene_pickle': data+'/train_v3scans.pkl',
    'native_rows': training+'/native_evaluation/rows.json',
    'native_receipt': training+'/native_evaluation/receipt.json',
    'split_protocol': original['split_protocol'],
    'paired_rows': transfer+'/analysis/paired_rows.csv', 'input_files': inputs,
    'counts_are_geometric_proxies_not_semantic_identity': True,
    'tie_atol': 1e-6, 'gpu_forwards': 0, 'formal_rows': 0, 'optimizer_steps': 0,
}
files['input_manifest.json'] = (json.dumps(manifest, indent=2, sort_keys=True)+'\n').encode()
controller = '''#!/usr/bin/env bash
set -euo pipefail
cd ROOT
export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
trap 'printf "%s\\n" "$?" > controller.exit' EXIT
/root/miniconda3/envs/bdetr/bin/python -m pytest -q tests/test_appearance_native_overlap.py tests/test_instance_overlap_diagnostic.py > cpu_tests.txt 2>&1
/root/miniconda3/envs/bdetr/bin/python -u -m scripts.analyze_scanrefer_appearance_native_overlap --directory .
'''.replace('ROOT', shlex.quote(remote))
files['controller.sh'] = controller.encode()
files['launch_from_local.py'] = Path(__file__).read_bytes()
check = "from pathlib import Path; import json,shutil; limit=int(Path('/sys/fs/cgroup/memory.max').read_text()); used=int(Path('/sys/fs/cgroup/memory.current').read_text()); free=shutil.disk_usage('/root/autodl-tmp').free; assert limit-used>12*1024**3 and free>150*1024**2; print(json.dumps(dict(memory_limit=limit,memory_used=used,free_bytes=free)))"
_, stdout, stderr = c.exec_command('/root/miniconda3/envs/bdetr/bin/python -c '+shlex.quote(check), timeout=30)
raw = stdout.read()
assert stdout.channel.recv_exit_status() == 0, stderr.read().decode()
resources = json.loads(raw)
archive.mkdir()
s.mkdir(remote)
s.mkdir(remote+'/scripts')
s.mkdir(remote+'/tests')
for path, blob in files.items():
    destination = archive/path
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(blob)
    with s.open(remote+'/'+path, 'wx') as stream:
        stream.write(blob)
    assert read(remote+'/'+path) == blob
command = 'exec bash '+shlex.quote(remote+'/controller.sh')+' > '+shlex.quote(remote+'/run.log')+' 2>&1'
_, stdout, stderr = c.exec_command('screen -dmS mcln_appearance_native_overlap_v1 bash -c '+shlex.quote(command), timeout=30)
assert stdout.channel.recv_exit_status() == 0, stderr.read().decode()
_, stdout, stderr = c.exec_command('screen -ls', timeout=30)
screens = stdout.read().decode()
assert stdout.channel.recv_exit_status() == 0, stderr.read().decode()
matches = [line.strip() for line in screens.splitlines() if '.mcln_appearance_native_overlap_v1' in line]
assert len(matches) == 1
launch = {
    'time_cst': datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
    'screen': matches[0], 'resources': resources,
    'manifest_sha256': hash_bytes(files['input_manifest.json']),
    'estimated_duration_seconds': 120, 'actual_completion_verified': False,
}
blob = (json.dumps(launch, indent=2)+'\n').encode()
(archive/'launch.json').write_bytes(blob)
with s.open(remote+'/launch.json', 'wx') as stream:
    stream.write(blob)
s.close()
c.close()
print(json.dumps(launch), flush=True)
