"""Assemble the next native source tree without changing the running Scan job."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


root = Path(sys.argv[1]).resolve()
spec = json.loads((root / 'input_manifest.json').read_text())
parent = Path(spec['parent_source'])
assert sha(parent / 'native_source_manifest.json') == spec['parent_manifest_sha256']
parent_files = json.loads((parent / 'native_source_manifest.json').read_text())['files']
assert len(parent_files) == 618
training = Path(spec['running_training'])


def check_training_source():
    assert sha(training / 'input_manifest.json') == spec['running_manifest_sha256']
    for name, digest in json.loads((training / 'input_manifest.json').read_text())['files'].items():
        assert sha(training / name) == digest, name


check_training_source()
source = root / 'model_source'
source.mkdir()
files = dict(parent_files)
for name, digest in parent_files.items():
    raw = (parent / name).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == digest, name
    target = source / name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(raw)
for name, digest in spec['overlays'].items():
    raw = (root / 'overlays' / name).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == digest, name
    target = source / name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(raw)
    files[name] = digest
for name, digest in files.items():
    assert sha(source / name) == digest, name
    (source / name).chmod(0o444)
source_manifest = dict(schema='mcln-native-mask-geometry-complete-source-v1',
    model_source=str(source), parent_source=str(parent),
    parent_manifest_sha256=spec['parent_manifest_sha256'],
    overlay_git_commit=spec['overlay_git_commit'], files=files)
(source / 'native_source_manifest.json').write_text(json.dumps(source_manifest, indent=2, sort_keys=True) + '\n')
environment = dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONPATH=str(source),
    PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', TOKENIZERS_PARALLELISM='false')
commands = [
    [sys.executable, 'train_dist_mod.py', '--help'],
    [sys.executable, str(root / 'check_complete_source.py'), str(source)],
]
results = []
for index, command in enumerate(commands):
    with (root / ('check_%d.txt' % index)).open('xb') as stream:
        process = subprocess.run(command, cwd=str(source), env=environment,
                                 stdout=stream, stderr=subprocess.STDOUT)
    results.append({'command': command, 'exit_code': process.returncode})
    assert process.returncode == 0, results[-1]
check_training_source()
for name, digest in files.items():
    assert sha(source / name) == digest, name
result = {'schema': 'mcln-native-mask-geometry-complete-source-preparation-v1',
    'time_cst': datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
    'status': 'pass', 'source_files': len(files), 'model_source': str(source),
    'source_manifest_sha256': sha(source / 'native_source_manifest.json'),
    'input_manifest_sha256': sha(root / 'input_manifest.json'), 'checks': results,
    'running_source_unchanged': True, 'canonical_core_overwritten': False,
    'gpu_forwards': 0, 'optimizer_steps': 0, 'checkpoint_writes': 0,
    'formal_rows': 0, 'actual_endpoint_bound': False,
    'scope': 'Complete native CLI and criterion source packaging; CPU synthetic tests only. Not real dataset training.'}
with (root / 'receipt.json').open('x') as stream:
    json.dump(result, stream, indent=2, sort_keys=True)
print(json.dumps(result), flush=True)
