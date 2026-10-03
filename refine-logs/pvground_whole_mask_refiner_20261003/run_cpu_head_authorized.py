"""First reviewed standalone head witness, CPU only and zero disk weights."""
import datetime
import json
import os
from pathlib import Path
import shlex
import time

import paramiko


local = Path(__file__).parent
assert not (local / 'cpu_receipt.json').exists()
review = json.loads((local / 'EXPERIMENT_CODE_REVIEW.json').read_bytes())
assert review['verdict'] in ('PASS', 'WARN') and not review['blocking_findings']
spec = json.loads((local.parent / 'pvground_candidate_normalization_20261003/normalized_spec.json').read_bytes())
remote = '/root/autodl-tmp/pvground_whole_mask_refiner_cpu_20261003'
client = paramiko.SSHClient()
client.load_system_host_keys()
client.connect('region-9.autodl.pro', port=33476, username='root',
               password=os.environ['MCLN_SSH_PASSWORD'], timeout=30)
sftp = client.open_sftp()
sftp.mkdir(remote)
names = ['whole_mask_range.py', 'pvground_candidate_box_refiner.py', 'pvground_tail_support_box_refiner.py',
         'pvground_whole_mask_box_refiner.py', 'cpu_head_witness.py', 'native_losses.py']
for name in names:
    raw = (local / name).read_bytes()
    with sftp.open(remote + '/' + name, 'wx') as stream:
        stream.write(raw)
    with sftp.open(remote + '/' + name, 'rb') as stream:
        assert stream.read() == raw
command = shlex.join(['env', 'CUDA_VISIBLE_DEVICES=', spec['runtime'] + '/venv/bin/python', '-B', '-u',
    remote + '/cpu_head_witness.py', '--fixtures', spec['reference_fixtures'], '--native-loss',
    remote + '/native_losses.py', '--output', remote + '/receipt.json'])
begin = time.time()
_, stdout, stderr = client.exec_command(command, timeout=180)
raw = stdout.read()
error = stderr.read()
code = stdout.channel.recv_exit_status()
(local / 'cpu.stdout').write_bytes(raw)
(local / 'cpu.stderr').write_bytes(error)
(local / 'cpu.exit').write_text(str(code) + '\n')
assert code == 0, error.decode()
with sftp.open(remote + '/receipt.json', 'rb') as stream:
    receipt = stream.read()
result = json.loads(receipt)
assert result['status'] == 'pass' and result['standalone_math_optimizer_updates'] == 2
assert result['PVGround_model_forwards'] == result['real_fit_optimizer_updates'] == result['weight_files_created'] == 0
assert not result['CUDA_initialized'] and not result['installed_in_actual_model']
assert sorted(sftp.listdir(remote)) == sorted(names + ['receipt.json'])
(local / 'cpu_receipt.json').write_bytes(receipt)
sftp.close()
client.close()
record = dict(time_cst=datetime.datetime.now().astimezone().isoformat(), remote=remote,
    seconds=time.time()-begin, exit_code=code, CPU_only=True, active_training_queried=False,
    active_training_modified=False, GPU_queried=False, real_training_launched=False,
    accuracy_results_available=False)
(local / 'cpu_execution.json').write_bytes((json.dumps(record, indent=2) + '\n').encode())
print(json.dumps(dict(execution=record, witness=result)))
