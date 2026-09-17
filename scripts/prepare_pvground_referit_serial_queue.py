"""Install the serial queue and check it on CPU; this script launches no training."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import paramiko


repo = Path(__file__).resolve().parents[1]
root = '/root/autodl-tmp/mcln_pvground_referit_serial_queue_20260917_v1'
archive = repo / 'refine-logs/pvground_referit_serial_queue_20260917_v1'
archive.mkdir()
client = paramiko.SSHClient()
client.load_system_host_keys()
client.connect('region-9.autodl.pro', port=33476, username='root',
               password=os.environ['MCLN_SSH_PASSWORD'], timeout=30)
sftp = client.open_sftp()
sftp.mkdir(root)
sftp.mkdir(root + '/scripts')
sftp.mkdir(root + '/tests')


def remote_read(path):
    with sftp.open(path, 'rb') as stream:
        return stream.read()


def install(name, raw):
    path = archive / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    sftp.put(str(path), root + '/' + name)
    assert remote_read(root + '/' + name) == raw


for name in ['scripts/run_pvground_referit_serial_queue.py', 'tests/test_pvground_referit_serial_queue.py']:
    install(name, (repo / name).read_bytes())
runtime = '/root/autodl-tmp/mcln_pvground_runtime_20260908_v1'
scan = '/root/autodl-tmp/mcln_pvground_scanrefer_formal_20260917_rec_competition_v1'
spec = dict(runtime=runtime, env_spec_sha256='966235b2ead7fa5a1de63e9537745b457374a4e5749ac4a7a26566b7b630c82c',
            scan_formal_root=scan, scan_formal_pid=12668,
            first_check_cst='2026-09-17T23:13:44+08:00',
            gpu_lock='/root/autodl-tmp/mcln_v99_backbone_gpu0.lock', datasets=[], files={})
for short, dataset, pid in [('nr', 'nr3d', 13286), ('sr', 'sr3d', 13750)]:
    training = '/root/autodl-tmp/mcln_pvground_' + short + '_finetune_20260917_rec_competition_v1'
    endpoint = training.replace('_finetune_', '_endpoint_audit_')
    formal = training.replace('_finetune_', '_formal_')
    item = dict(dataset=dataset, training_root=training, audit_root=endpoint, formal_root=formal,
                probe_root='/root/autodl-tmp/mcln_pvground_' + short + '_rec_competition_preparation_20260917_v1',
                probe_pid=pid, training_free_bytes=2 * 350 * 1024**2 + 384 * 1024**2 + 768 * 1024**2,
                formal_free_bytes=512 * 1024**2 + 768 * 1024**2)
    spec['datasets'].append(item)
    for directory in [training, endpoint, formal]:
        for name in sftp.listdir(directory):
            if name.endswith('.py') or name == 'spec.json':
                path = directory + '/' + name
                spec['files'][path] = hashlib.sha256(remote_read(path)).hexdigest()
    train_spec = json.loads(remote_read(training + '/spec.json'))
    assert train_spec['env_spec_sha256'] == spec['env_spec_sha256']
    assert train_spec['scan_formal_root'] == scan
    assert 'controller.pid' not in sftp.listdir(training) and 'controller.exit' not in sftp.listdir(training)
queue_path = root + '/scripts/run_pvground_referit_serial_queue.py'
spec['files'][queue_path] = hashlib.sha256(remote_read(queue_path)).hexdigest()
install('spec.json', (json.dumps(spec, indent=2) + '\n').encode())
controller = """import os,subprocess
from pathlib import Path
root=Path(%r)
with (root/'controller.pid').open('x') as f:f.write(str(os.getpid())+'\\n')
run=subprocess.run(['/root/miniconda3/envs/bdetr/bin/python','-u',str(root/'scripts/run_pvground_referit_serial_queue.py'),'--spec',str(root/'spec.json')])
with (root/'controller.exit').open('x') as f:f.write(str(run.returncode)+'\\n')
raise SystemExit(run.returncode)
""" % root
compile(controller, 'controller.py', 'exec')
install('controller.py', controller.encode())
command = runtime + '/venv/bin/python ' + root + '/tests/test_pvground_referit_serial_queue.py'
_, stdout, stderr = client.exec_command(command, timeout=60)
output, error = stdout.read(), stderr.read()
code = stdout.channel.recv_exit_status()
install('cpu_check.stdout.log', output)
install('cpu_check.stderr.log', error)
assert code == 0, error.decode()
receipt = dict(status='prepared', time_cst=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
               cpu_tests=4, cpu_exit=code, training_jobs_launched=0, controller_launched=False,
               model_forwards=0, optimizer_steps=0, formal_rows=0,
               spec_sha256=hashlib.sha256(remote_read(root + '/spec.json')).hexdigest(),
               queue_sha256=spec['files'][queue_path])
install('preparation_receipt.json', (json.dumps(receipt, indent=2) + '\n').encode())
print(json.dumps(receipt), flush=True)
sftp.close()
client.close()
