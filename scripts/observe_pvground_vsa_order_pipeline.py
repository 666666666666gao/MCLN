import datetime
import json
import os
from pathlib import Path
import shlex
import paramiko

root = '/root/autodl-tmp/mcln_pvground_scanrefer_finetune_20260908_vsaorder_v1'
audit_root = '/root/autodl-tmp/mcln_pvground_scanrefer_endpoint_audit_20260908_vsaorder_v1'
formal_root = '/root/autodl-tmp/mcln_pvground_scanrefer_formal_20260908_vsaorder_v1'
archive = Path(__file__).resolve().parents[1] / 'refine-logs/pvground_scanrefer_finetune_20260908_vsaorder_v1'
client = paramiko.SSHClient()
client.load_system_host_keys()
client.connect('region-9.autodl.pro', port=33476, username='root', password=os.environ['MCLN_SSH_PASSWORD'], timeout=30)
sftp = client.open_sftp()
names = sftp.listdir(root)
probe = "import json,shutil,subprocess; print(json.dumps({'processes':subprocess.check_output(['ps','-eo','pid,ppid,etimes,args']).decode(),'disk_free':shutil.disk_usage('/root/autodl-tmp').free,'gpu':subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader']).decode()}))"
_, stdout, stderr = client.exec_command('/root/miniconda3/envs/bdetr/bin/python -c ' + shlex.quote(probe), timeout=30)
state = json.loads(stdout.read())
assert stdout.channel.recv_exit_status() == 0, stderr.read().decode()
state['processes'] = [line for line in state['processes'].splitlines()
                      if any(path in line for path in [root, audit_root, formal_root]) and ' -c ' not in line]
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
audit_archive = archive.parent / 'pvground_scanrefer_endpoint_audit_20260908_vsaorder_v1'
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
formal_archive = archive.parent / 'pvground_scanrefer_formal_20260908_vsaorder_v1'
formal_names = sftp.listdir(formal_root)
formal_record = {'files': formal_names}
for name in ['decision.json', 'protocol.json', 'imports.json', 'receipt.json', 'audit.json',
             'evaluation.exit', 'audit.exit', 'controller.exit', 'run.log']:
    if name in formal_names:
        sftp.get(formal_root + '/' + name, str(formal_archive / name))
        if name.endswith('.json'):
            formal_record[name] = json.loads((formal_archive / name).read_bytes())
        elif name == 'run.log':
            formal_record['tail'] = (formal_archive / name).read_text().splitlines()[-16:]
        else:
            formal_record[name] = (formal_archive / name).read_text().strip()
for arm in ['published_parent', 'fit_terminal']:
    if arm in formal_names:
        arm_names = sftp.listdir(formal_root + '/' + arm)
        formal_record[arm] = {'files': arm_names}
        if 'receipt.json' in arm_names:
            (formal_archive / arm).mkdir(exist_ok=True)
            sftp.get(formal_root + '/' + arm + '/receipt.json', str(formal_archive / arm / 'receipt.json'))
            formal_record[arm]['receipt'] = json.loads((formal_archive / arm / 'receipt.json').read_bytes())
record['formal_evaluation'] = formal_record
stamp = record['time_cst'].replace(':', '').replace('-', '').replace('+', '_').split('.')[0]
raw_record = (json.dumps(record, indent=2) + '\n').encode()
(archive / ('observation_' + stamp + '.json')).write_bytes(raw_record)
(archive / 'observation_latest.json').write_bytes(raw_record)
sftp.close()
client.close()
print(json.dumps(record), flush=True)
