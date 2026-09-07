import datetime
import hashlib
import json
import os
from pathlib import Path
import paramiko

local = Path('C:/Users/gb/.codex_mcln_g0_20260905/refine-logs/scanrefer_mask_geometry_training_probe_20260907_v1')
remote = '/root/autodl-tmp/mcln_scanrefer_mask_geometry_training_probe_20260907_v1'
c = paramiko.SSHClient()
c.load_system_host_keys()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect('region-9.autodl.pro', port=33476, username='root', password=os.environ['MCLN_SSH_PASSWORD'], timeout=30)
s = c.open_sftp()
names = s.listdir(remote)
for name in ['run.log', 'controller.exit', 'receipt.json']:
    if name in names:
        (local / name).write_bytes(s.open(remote + '/' + name, 'rb').read())
_, out, err = c.exec_command('ps -p 62496,62498,62499 -o pid,ppid,stat,etime,args\nnvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader\ndf -B1 /root/autodl-tmp')
observation = {'time_cst': datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(), 'live': out.read().decode(),
    'controller_exit': (local / 'controller.exit').read_text().strip() if 'controller.exit' in names else None,
    'log_tail': (local / 'run.log').read_text(errors='replace').splitlines()[-8:]}
(local / 'observation.json').write_text(json.dumps(observation, indent=2) + '\n', encoding='utf-8')
if 'receipt.json' in names:
    receipt = json.loads((local / 'receipt.json').read_bytes())
    observation['receipt'] = {'status': receipt['status'], 'rows': receipt['rows'], 'gradient_cosines': [row['gradient_cosine'] for row in receipt['observations']],
        'parameters': len(receipt['requires_grad_parameters']), 'max_gpu_mib': receipt['max_gpu_mib'],
        'updates': {arm: {'changed_parameters': len(row['changed_tensors']), 'post_native_loss': row['post_native_loss'],
            'post_geometry_loss': row['post_geometry_loss'], 'steps': row['steps']} for arm, row in receipt['updates'].items()},
        'sha256': hashlib.sha256((local / 'receipt.json').read_bytes()).hexdigest()}
s.close()
c.close()
print(json.dumps(observation), flush=True)
