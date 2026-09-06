import datetime
import hashlib
import json
import os
from pathlib import Path

import paramiko

local = Path('C:/Users/gb/.codex_mcln_g0_20260905/refine-logs/scanrefer_stage_diagnostic_20260907_v1')
remote = '/root/autodl-tmp/mcln_scanrefer_stage_diagnostic_20260907_v1'
launch = json.loads((local / 'launch.json').read_bytes())
pid = int(launch['screen_session'][0].split('.', 1)[0])
client = paramiko.SSHClient()
client.load_system_host_keys()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect('region-9.autodl.pro', port=33476, username='root',
               password=os.environ['MCLN_SSH_PASSWORD'], timeout=30)
sftp = client.open_sftp()
_, output, error = client.exec_command('ps -p ' + str(pid) + ' -o pid,stat,etime,args', timeout=30)
process = output.read().decode()
status = output.channel.recv_exit_status()
assert status in (0, 1), error.read().decode()
live = status == 0 and 'mcln_scanrefer_stage_diagnostic_v1' in process
with sftp.open(remote + '/run.log', 'rb') as stream:
    stream.seek(max(0, stream.stat().st_size - 64000))
    log = stream.read()
progress = {}
for line in log.decode().splitlines():
    for label in ['SCANREFER STAGE DIAGNOSTIC ', 'SCANREFER STAGE DIAGNOSTIC COMPLETE ']:
        if line.startswith(label + '{'):
            progress[label.strip()] = json.loads(line[len(label):])
now = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8)))
result = {'time_cst': now.isoformat(), 'screen_live': live, 'process': process,
          'progress': progress, 'manifest_sha256': launch['manifest_sha256']}
if not live:
    with sftp.open(remote + '/controller.exit', 'rb') as stream:
        raw = stream.read()
    (local / 'controller.exit').write_bytes(raw)
    result['controller_exit'] = int(raw.strip())
stem = 'progress_' + now.strftime('%Y%m%d_%H%M%S')
(local / (stem + '.json')).write_text(json.dumps(result, indent=2, sort_keys=True), encoding='utf-8')
(local / (stem + '.txt')).write_bytes(log)
sftp.close()
client.close()
print(json.dumps(result))
print('\n'.join(log.decode().splitlines()[-20:]))
