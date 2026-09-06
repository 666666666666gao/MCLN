import datetime
import hashlib
import json
import os
from pathlib import Path
import shlex
import xml.etree.ElementTree as ET

import paramiko


repo = Path('C:/Users/gb/.codex_mcln_g0_20260905')
local = repo / 'refine-logs/scanrefer_local_visual_official_20260906_v1/audit_preparation'
remote = '/root/autodl-tmp/mcln_scanrefer_local_visual_official_20260906_v1/audit_preparation'
names = ['scripts/audit_scanrefer_local_visual_official.py',
         'scripts/audit_scanrefer_joint_readout_pair.py',
         'scripts/evaluate_scanrefer_local_visual_official.py',
         'tests/test_audit_scanrefer_local_visual_official.py',
         'tests/test_evaluate_scanrefer_local_visual_official.py']
files = {name: (repo / name).read_bytes() for name in names}
files['scripts/__init__.py'] = b''
files['audit_terminal.py'] = Path('C:/Users/gb/.codex/tmp/audit_mcln_local_visual_official_20260906.py').read_bytes()
files['prepare.py'] = Path(__file__).read_bytes()
manifest = {'schema': 'mcln-scanrefer-formal-audit-preparation-v1',
            'files': {name: hashlib.sha256(raw).hexdigest() for name, raw in files.items()},
            'scope': 'Independent audit of either formal outcome;synthetic unit tests are not evaluation results.'}
files['manifest.json'] = (json.dumps(manifest, indent=2, sort_keys=True) + '\n').encode()
local.mkdir()
(local / 'scripts').mkdir()
(local / 'tests').mkdir()
client = paramiko.SSHClient()
client.load_system_host_keys()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect('region-9.autodl.pro', port=33476, username='root', password=os.environ['MCLN_SSH_PASSWORD'], timeout=30)
sftp = client.open_sftp()
sftp.mkdir(remote)
sftp.mkdir(remote + '/scripts')
sftp.mkdir(remote + '/tests')
for name, raw in files.items():
    (local / name).write_bytes(raw)
    with sftp.open(remote + '/' + name, 'wx') as stream:
        stream.write(raw)
command = ('cd ' + shlex.quote(remote)
           + " && CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1"
           + ' /root/miniconda3/envs/bdetr/bin/python -m pytest'
           + ' tests/test_audit_scanrefer_local_visual_official.py'
           + ' tests/test_evaluate_scanrefer_local_visual_official.py -q'
           + ' --junitxml=validation.xml > validation.log 2>&1')
_, output, error = client.exec_command(command, timeout=300)
output.read()
error_text = error.read().decode()
status = output.channel.recv_exit_status()
for remote_name, local_name in [('validation.log', 'validation.txt'), ('validation.xml', 'validation.xml')]:
    with sftp.open(remote + '/' + remote_name, 'rb') as stream:
        (local / local_name).write_bytes(stream.read())
assert status == 0, (local / 'validation.txt').read_text() + error_text
suites = list(ET.parse(str(local / 'validation.xml')).getroot().iter('testsuite'))
counts = {key: sum(int(suite.attrib[key]) for suite in suites) for key in ['tests', 'failures', 'errors', 'skipped']}
assert counts == {'tests': 23, 'failures': 0, 'errors': 0, 'skipped': 0}
receipt = {'schema': manifest['schema'], 'time_cst': datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
           'status': 'original_environment_unit_tests_pass', 'tests': counts,
           'gpu_forwards': 0, 'optimizer_steps': 0, 'checkpoint_writes': 0,
           'actual_formal_rows_audited': 0, 'manifest_sha256': hashlib.sha256(files['manifest.json']).hexdigest(),
           'validation_xml_sha256': hashlib.sha256((local / 'validation.xml').read_bytes()).hexdigest(),
           'formal_job_not_polled': True}
raw = (json.dumps(receipt, indent=2, sort_keys=True) + '\n').encode()
(local / 'receipt.json').write_bytes(raw)
with sftp.open(remote + '/receipt.json', 'wx') as stream:
    stream.write(raw)
sftp.close()
client.close()
print(json.dumps(receipt))
