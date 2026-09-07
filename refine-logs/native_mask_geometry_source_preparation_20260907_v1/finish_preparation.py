"""Finish the existing source check after correcting duplicate namespace paths."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

root = Path(sys.argv[1]).resolve()
source = root / 'model_source'
spec = json.loads((root / 'input_manifest.json').read_text())
manifest = json.loads((source / 'native_source_manifest.json').read_text())
training = Path(spec['running_training'])


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify():
    assert sha(training / 'input_manifest.json') == spec['running_manifest_sha256']
    for name, digest in json.loads((training / 'input_manifest.json').read_text())['files'].items():
        assert sha(training / name) == digest, name
    for name, digest in manifest['files'].items():
        assert sha(source / name) == digest, name


verify()
assert '--native_mask_geometry_supervision' in (root / 'check_0.txt').read_text()
environment = dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONPATH=str(source),
    PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', TOKENIZERS_PARALLELISM='false')
command = [sys.executable, str(root / 'check_complete_source_v2.py'), str(source)]
with (root / 'check_2.txt').open('xb') as stream:
    process = subprocess.run(command, cwd=str(source), env=environment,
                             stdout=stream, stderr=subprocess.STDOUT)
assert process.returncode == 0, process.returncode
verify()
result = {'schema': 'mcln-native-mask-geometry-complete-source-preparation-v1',
    'time_cst': datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
    'status': 'pass', 'source_files': len(manifest['files']), 'model_source': str(source),
    'source_manifest_sha256': sha(source / 'native_source_manifest.json'),
    'input_manifest_sha256': sha(root / 'input_manifest.json'),
    'checker_sha256': sha(root / 'check_complete_source_v2.py'),
    'check_command': command, 'check_exit': process.returncode,
    'cli_help_complete': True, 'normal_imports_inside_snapshot': True,
    'prior_failure': 'Namespace scripts path contained the same resolved directory three times; strict list-cardinality check was wrong. Only checker changed to compare the resolved directory set.',
    'source_changed_to_fix_check': False, 'running_source_unchanged': True,
    'canonical_core_overwritten': False, 'gpu_forwards': 0, 'optimizer_steps': 0,
    'checkpoint_writes': 0, 'formal_rows': 0, 'actual_endpoint_bound': False,
    'scope': 'Complete native CLI and criterion source packaging; CPU synthetic tests only. Not real dataset training.'}
with (root / 'receipt.json').open('x') as stream:
    json.dump(result, stream, indent=2, sort_keys=True)
print(json.dumps(result), flush=True)
