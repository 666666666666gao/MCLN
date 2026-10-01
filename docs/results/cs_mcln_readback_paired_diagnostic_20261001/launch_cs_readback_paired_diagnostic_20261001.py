from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import shlex

import paramiko


main = Path(r'C:\Users\gb\.codex_mcln_g0_20260905')
directory = main / 'docs/results/cs_mcln_readback_paired_diagnostic_20261001'
cpu = json.loads((directory / 'cpu_fixture.json').read_text())
assert cpu['exit_code'] == 0 and not cpu['result']['cuda_initialized']
launch = json.loads((main / 'docs/results/cs_mcln_scanrefer_readback_20260930/launch.json').read_text())
remote = cpu['remote_diagnostic_directory']
python = '/root/miniconda3/envs/bdetr/bin/python'
parent = '/root/autodl-tmp/DATA_ROOT/output/preserved_best/mcln_pair_sweep/mcln_pair_default_rankblend010_2ep_best_acc025_epoch71_0.57993.pth'
command = ' '.join(shlex.quote(value) for value in (
    python, '-u', remote + '/run_cs_mcln_readback_posttrain_diagnostic.py',
    '--training-pid', '187859', '--not-before', '2026-10-04T01:45:00+08:00',
    '--run-dir', launch['output'], '--best-dir', launch['best_dir'],
    '--source', launch['source'], '--base', launch['immutable_base_source'],
    '--diagnostic', remote + '/diagnose_cs_mcln_readback.py',
    '--parent', parent, '--data-root', '/root/autodl-tmp/DATA_ROOT',
))
script = ('#!/usr/bin/env bash\nexport CUDA_VISIBLE_DEVICES=0\n' + command
          + ' > ' + shlex.quote(remote + '/controller.log') + ' 2>&1\n'
          + 'exit_code=$?\nprintf \'%s\\n\' "$exit_code" > '
          + shlex.quote(remote + '/controller.exit.txt') + '\nexit "$exit_code"\n')
client = paramiko.SSHClient()
client.load_system_host_keys()
client.connect('region-9.autodl.pro', port=33476, username='root',
               password=os.environ['MCLN_SSH_PASSWORD'], timeout=30)
sftp = client.open_sftp()
for name, expected in cpu['files_sha256'].items():
    with sftp.open(remote + '/' + name, 'rb') as stream:
        assert hashlib.sha256(stream.read()).hexdigest() == expected
launcher = remote + '/controller.sh'
with sftp.open(launcher, 'wb') as stream:
    stream.write(script.encode('utf-8'))
_, stdout, stderr = client.exec_command('bash -n ' + shlex.quote(launcher), timeout=30)
assert stdout.channel.recv_exit_status() == 0, stderr.read().decode()
session = 'cs_readback_paired_diag_20261001'
_, stdout, stderr = client.exec_command(
    'screen -dmS ' + shlex.quote(session) + ' bash ' + shlex.quote(launcher), timeout=30,
)
assert stdout.channel.recv_exit_status() == 0, stderr.read().decode()
_, stdout, stderr = client.exec_command(
    'sleep 2\npgrep -af ' + shlex.quote('[p]ython -u ' + remote + '/run_cs_mcln_readback_posttrain_diagnostic.py'),
    timeout=30,
)
process = stdout.read().decode().strip()
assert stdout.channel.recv_exit_status() == 0, stderr.read().decode()
with sftp.open(remote + '/controller.log', 'r') as stream:
    log = stream.read().decode()
assert 'WAIT_UNTIL 2026-10-04T01:45:00+08:00' in log and 'START_PHASE' not in log
receipt = {
    'launched_at_cst': datetime.now(timezone(timedelta(hours=8))).isoformat(),
    'session': session, 'process': process, 'command': command,
    'not_before_cst': '2026-10-04T01:45:00+08:00',
    'unfinished_training_poll_seconds': 300, 'status': 'WAIT_UNTIL',
    'controller_initial_log': log, 'gpu_sanity_performed': False,
    'full_diagnostic_performed': False, 'training_configuration_changed': False,
    'ordering': 'after R21 exits successfully; first12 then full9508; before common-numeric CS GPU control',
    'diagnostic_files_sha256': cpu['files_sha256'],
    'controller_launcher_sha256': hashlib.sha256(script.encode('utf-8')).hexdigest(),
}
(directory / 'posttrain_launch.json').write_text(json.dumps(receipt, indent=2) + '\n', encoding='utf-8')
(directory / 'controller.sh').write_text(script, encoding='utf-8')
sftp.close()
client.close()
print(json.dumps(receipt, indent=2), flush=True)
