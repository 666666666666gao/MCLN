import datetime
import hashlib
import json
import os
from pathlib import Path
import paramiko

local = Path('C:/Users/gb/.codex_mcln_g0_20260905/refine-logs/scanrefer_geometry_gradient_probe_20260907_v1')
remote = '/root/autodl-tmp/mcln_scanrefer_geometry_gradient_probe_20260907_v1'
c = paramiko.SSHClient()
c.load_system_host_keys()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect('region-9.autodl.pro', port=33476, username='root', password=os.environ['MCLN_SSH_PASSWORD'], timeout=30)
s = c.open_sftp()
names = s.listdir(remote)
for name in ['run.log', 'controller.exit', 'receipt.json']:
    if name in names:
        raw = s.open(remote + '/' + name, 'rb').read()
        (local / name).write_bytes(raw)
_, out, err = c.exec_command("ps -p 62152,62154,62155 -o pid,ppid,stat,etime,args\nnvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader\ndf -B1 /root/autodl-tmp")
live = out.read().decode()
observation = {'time_cst': datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(), 'live': live,
    'controller_exit': (local / 'controller.exit').read_text().strip() if 'controller.exit' in names else None,
    'log_tail': (local / 'run.log').read_text(errors='replace').splitlines()[-8:]}
(local / 'observation.json').write_text(json.dumps(observation, indent=2) + '\n', encoding='utf-8')
if 'receipt.json' in names:
    receipt = json.loads((local / 'receipt.json').read_bytes())
    observation['receipt'] = {'status': receipt['status'], 'rows': receipt['rows'],
        'max_gpu_mib': receipt['max_gpu_mib'], 'sha256': hashlib.sha256((local / 'receipt.json').read_bytes()).hexdigest(),
        'valid_per_variant': [sum(row['valid_per_variant'][i] for row in receipt['observations']) for i in range(7)],
        'gradient_norms': [{name: {group: value['norm'] for group, value in item['groups'].items()}
            for name, item in row['loss_gradients'].items()} for row in receipt['observations']]}
s.close()
c.close()
print(json.dumps(observation), flush=True)
