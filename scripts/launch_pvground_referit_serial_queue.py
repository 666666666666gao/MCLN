"""Arm the prepared REC-gated queue once, without starting models before Scan passes."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import shlex
import paramiko

repo = Path(__file__).resolve().parents[1]
archive = repo / 'refine-logs/pvground_referit_serial_queue_20260917_v1'
root = '/root/autodl-tmp/mcln_pvground_referit_serial_queue_20260917_v1'
client = paramiko.SSHClient()
client.load_system_host_keys()
client.connect('region-9.autodl.pro', port=33476, username='root',
               password=os.environ['MCLN_SSH_PASSWORD'], timeout=30)
sftp = client.open_sftp()


def read(path):
    with sftp.open(path, 'rb') as stream:
        return stream.read()


def command(value):
    _, out, err = client.exec_command(value, timeout=60)
    output, error = out.read(), err.read()
    assert out.channel.recv_exit_status() == 0, error.decode()
    return output


assert 'controller.pid' not in sftp.listdir(root)
assert 'controller.exit' not in sftp.listdir(root)
prep = json.loads(read(root + '/preparation_receipt.json'))
spec_raw = read(root + '/spec.json')
spec = json.loads(spec_raw)
assert prep['status'] == 'prepared' and prep['cpu_exit'] == 0 and prep['cpu_tests'] == 4
assert hashlib.sha256(spec_raw).hexdigest() == prep['spec_sha256']
assert read(root + '/controller.py') == (archive / 'controller.py').read_bytes()
for path, digest in spec['files'].items():
    assert hashlib.sha256(read(path)).hexdigest() == digest, path
dependencies = [(spec['scan_formal_root'], spec['scan_formal_pid'])]
dependencies.extend((item['probe_root'], item['probe_pid']) for item in spec['datasets'])
for directory, pid in dependencies:
    assert (directory + '/controller.py').encode() in read('/proc/' + str(pid) + '/cmdline')
free = int(command('df -B1 --output=avail /root/autodl-tmp').decode().splitlines()[-1])
assert free >= 3 * 1024**3, 'retain space for running Scan and serial ReferIt stages'
assert not (archive / 'launch_receipt.json').exists()
command('nohup /root/miniconda3/envs/bdetr/bin/python -u ' + shlex.quote(root + '/controller.py') +
        ' > ' + shlex.quote(root + '/controller.log') + ' 2>&1 < /dev/null &')
command('sleep 2')
pid = int(read(root + '/controller.pid'))
process = command('ps -p ' + str(pid) + ' -o pid,ppid,etime,args').decode()
assert (root + '/controller.py').encode() in read('/proc/' + str(pid) + '/cmdline')
assert 'controller.exit' not in sftp.listdir(root)
receipt = dict(status='waiting_for_scanrefer_rec', time_cst=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
               controller_pid=pid, process=process, disk_free_before_launch=free,
               spec_sha256=prep['spec_sha256'], queue_sha256=prep['queue_sha256'],
               first_check_cst=spec['first_check_cst'], poll_seconds=300,
               training_jobs_launched=0, model_forwards=0, formal_rows=0)
raw = (json.dumps(receipt, indent=2) + '\n').encode()
(archive / 'launch_receipt.json').write_bytes(raw)
with sftp.open(root + '/launch_receipt.json', 'wx') as stream:
    stream.write(raw)
assert read(root + '/launch_receipt.json') == raw
print(json.dumps(receipt), flush=True)
sftp.close()
client.close()
