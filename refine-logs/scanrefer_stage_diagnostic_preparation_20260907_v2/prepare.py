import datetime
import hashlib
import json
import os
from pathlib import Path
import shlex
import xml.etree.ElementTree as ET

import paramiko

repo = Path('C:/Users/gb/.codex_mcln_g0_20260905')
local = repo / 'refine-logs/scanrefer_stage_diagnostic_preparation_20260907_v2'
remote = '/root/autodl-tmp/mcln_scanrefer_stage_diagnostic_preparation_20260907_v2'
source = '/root/autodl-tmp/mcln_scanrefer_local_visual_preflight_20260906_v2/model_source'
names = ['scripts/diagnose_scanrefer_readout_stages.py', 'scripts/scanrefer_stage_diagnostics.py',
         'scripts/trace_scanrefer_readout_stages.py', 'scripts/evaluate_scanrefer_local_visual_official.py',
         'tests/test_scanrefer_stage_diagnostics.py', 'tests/test_trace_scanrefer_readout_stages.py']
files = {name: (repo / name).read_bytes() for name in names}
files['scripts/__init__.py'] = b''
files['prepare.py'] = Path(__file__).read_bytes()
files['build_runner.py'] = Path('C:/Users/gb/.codex/tmp/build_mcln_stage_diagnostic_runner_20260907.py').read_bytes()
manifest = {'schema': 'mcln-scanrefer-stage-diagnostic-preparation-v1',
            'files': {name: hashlib.sha256(raw).hexdigest() for name, raw in files.items()},
            'existing_model_source': source,
            'scope': 'CPU unit tests and CLI loading only; no real data/model forward or metric claim.'}
files['manifest.json'] = (json.dumps(manifest, indent=2, sort_keys=True) + '\n').encode()
local.mkdir()
(local / 'scripts').mkdir()
(local / 'tests').mkdir()
client = paramiko.SSHClient()
client.load_system_host_keys()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect('region-9.autodl.pro', port=33476, username='root',
               password=os.environ['MCLN_SSH_PASSWORD'], timeout=30)
sftp = client.open_sftp()
sftp.mkdir(remote)
sftp.mkdir(remote + '/scripts')
sftp.mkdir(remote + '/tests')
for name, raw in files.items():
    (local / name).write_bytes(raw)
    with sftp.open(remote + '/' + name, 'wx') as stream:
        stream.write(raw)
command = ('cd ' + shlex.quote(remote)
           + " && export CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1"
           + ' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=' + shlex.quote(source)
           + ' && /root/miniconda3/envs/bdetr/bin/python -m pytest'
           + ' tests/test_scanrefer_stage_diagnostics.py tests/test_trace_scanrefer_readout_stages.py'
           + ' -q --junitxml=validation.xml > validation.log 2>&1'
           + ' && /root/miniconda3/envs/bdetr/bin/python scripts/diagnose_scanrefer_readout_stages.py --help > help.txt')
_, output, error = client.exec_command(command, timeout=60)
output.read()
error_text = error.read().decode()
status = output.channel.recv_exit_status()
for name in ['validation.log', 'validation.xml']:
    with sftp.open(remote + '/' + name, 'rb') as stream:
        (local / name).write_bytes(stream.read())
assert status == 0, (local / 'validation.log').read_text() + error_text
with sftp.open(remote + '/help.txt', 'rb') as stream:
    help_raw = stream.read()
assert b'--manifest MANIFEST' in help_raw
(local / 'help.txt').write_bytes(help_raw)
suites = list(ET.parse(str(local / 'validation.xml')).getroot().iter('testsuite'))
counts = {key: sum(int(suite.attrib[key]) for suite in suites)
          for key in ['tests', 'failures', 'errors', 'skipped']}
assert counts == {'tests': 13, 'failures': 0, 'errors': 0, 'skipped': 0}
receipt = {'schema': manifest['schema'],
           'time_cst': datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
           'status': 'original_environment_cpu_unit_tests_and_cli_pass', 'tests': counts,
           'gpu_forwards': 0, 'optimizer_updates': 0, 'checkpoint_writes': 0,
           'real_rows_traced': 0, 'training_polled': False, 'cli_help_pass': True,
           'manifest_sha256': hashlib.sha256(files['manifest.json']).hexdigest(),
           'validation_xml_sha256': hashlib.sha256((local / 'validation.xml').read_bytes()).hexdigest()}
raw = (json.dumps(receipt, indent=2, sort_keys=True) + '\n').encode()
(local / 'receipt.json').write_bytes(raw)
with sftp.open(remote + '/receipt.json', 'wx') as stream:
    stream.write(raw)
sftp.close()
client.close()
print(json.dumps(receipt))
