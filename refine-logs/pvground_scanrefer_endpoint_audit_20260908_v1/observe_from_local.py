import datetime
import json
import os
from pathlib import Path
import shlex
import paramiko

root = '/root/autodl-tmp/mcln_pvground_scanrefer_finetune_20260908_v1'
archive = Path('C:/Users/gb/.codex_mcln_g0_20260905/refine-logs/pvground_scanrefer_finetune_20260908_v1')
client = paramiko.SSHClient()
client.load_system_host_keys()
client.connect('region-9.autodl.pro', port=33476, username='root', password=os.environ['MCLN_SSH_PASSWORD'], timeout=30)
sftp = client.open_sftp()
names = sftp.listdir(root)
probe = "import json,shutil,subprocess; print(json.dumps({'processes':subprocess.check_output(['ps','-eo','pid,ppid,etimes,args']).decode(),'disk_free':shutil.disk_usage('/root/autodl-tmp').free,'gpu':subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader']).decode()}))"
_, stdout, stderr = client.exec_command('/root/miniconda3/envs/bdetr/bin/python -c ' + shlex.quote(probe), timeout=30)
state = json.loads(stdout.read())
assert stdout.channel.recv_exit_status() == 0, stderr.read().decode()
state['processes'] = [line for line in state['processes'].splitlines() if root in line and ' -c ' not in line]
with sftp.open(root + '/run.log', 'rb') as stream:
    stream.prefetch()
    raw = stream.read()
(archive / 'run.log').write_bytes(raw)
record = {'time_cst': datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
          'state': state, 'files': names, 'terminal': 'controller.exit' in names, 'tail': raw.decode().splitlines()[-28:]}
for name in ['controller.exit', 'imports.json', 'capacity.json', 'training_contract.json', 'receipt.json', 'train.jsonl']:
    if name in names:
        sftp.get(root + '/' + name, str(archive / name))
if record['terminal']:
    record['exit'] = (archive / 'controller.exit').read_text().strip()
for phase in ['initial', 'terminal']:
    if phase in names:
        phase_files = sftp.listdir(root + '/' + phase)
        record[phase + '_files'] = phase_files
        if 'receipt.json' in phase_files:
            (archive / phase).mkdir(exist_ok=True)
            sftp.get(root + '/' + phase + '/receipt.json', str(archive / phase / 'receipt.json'))
            record[phase + '_receipt'] = json.loads((archive / phase / 'receipt.json').read_bytes())
audit_root = '/root/autodl-tmp/mcln_pvground_scanrefer_endpoint_audit_20260908_v1'
audit_archive = archive.parent / 'pvground_scanrefer_endpoint_audit_20260908_v1'
audit_names = sftp.listdir(audit_root)
audit_record = {'files': audit_names}
for name in ['initial_audit.json', 'audit.json', 'audit.exit', 'controller.exit', 'dependency_failure.json', 'run.log']:
    if name in audit_names:
        sftp.get(audit_root + '/' + name, str(audit_archive / name))
        if name.endswith('.json'):
            audit_record[name] = json.loads((audit_archive / name).read_bytes())
        elif name == 'run.log':
            audit_record['tail'] = (audit_archive / name).read_text().splitlines()[-12:]
        else:
            audit_record[name] = (audit_archive / name).read_text().strip()
record['cpu_audit'] = audit_record
stamp = record['time_cst'].replace(':', '').replace('-', '').replace('+', '_').split('.')[0]
raw_record = (json.dumps(record, indent=2) + '\n').encode()
(archive / ('observation_' + stamp + '.json')).write_bytes(raw_record)
(archive / 'observation_latest.json').write_bytes(raw_record)
sftp.close()
client.close()
print(json.dumps(record), flush=True)
