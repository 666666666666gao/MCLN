import datetime
import hashlib
import json
import os
from pathlib import Path
import paramiko

local = Path('C:/Users/gb/.codex_mcln_g0_20260905/refine-logs/scanrefer_probability_geometry_probe_20260907_v1')
remote = '/root/autodl-tmp/mcln_scanrefer_probability_geometry_probe_20260907_v1'
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
_, out, err = c.exec_command("ps -eo pid,ppid,stat,etime,args | grep '[p]robe_scanrefer_probability_geometry'\nscreen -ls\nnvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader")
live = out.read().decode()
observation = {'time_cst': datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(), 'live': live,
    'controller_exit': (local / 'controller.exit').read_text().strip() if 'controller.exit' in names else None,
    'log_tail': (local / 'run.log').read_text(errors='replace').splitlines()[-8:],
    'launch_observer_issue': 'Initial post-launch ps grep retained the previous probe name; corrected read-only observation, no relaunch.'}
(local / 'observation.json').write_text(json.dumps(observation, indent=2) + '\n', encoding='utf-8')
if 'receipt.json' in names:
    receipt = json.loads((local / 'receipt.json').read_bytes())
    observation['receipt'] = {'status': receipt['status'], 'summary': receipt['summary'], 'gradient_records': receipt['gradient_records'],
        'max_gpu_mib': receipt['max_gpu_mib'], 'sha256': hashlib.sha256((local / 'receipt.json').read_bytes()).hexdigest()}
s.close()
c.close()
print(json.dumps(observation), flush=True)
