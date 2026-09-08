"""Observe the existing fit-support controller and collect bounded evidence."""
import datetime
import json
import os
from pathlib import Path
import shlex
import paramiko

repo = Path(__file__).resolve().parents[1]
root = '/root/autodl-tmp/mcln_pvground_fit_support_20260908_v2'
archive = repo / 'refine-logs/pvground_fit_support_20260908_v2'
c = paramiko.SSHClient(); c.load_system_host_keys()
c.connect('region-9.autodl.pro', port=33476, username='root', password=os.environ['MCLN_SSH_PASSWORD'], timeout=30)
s = c.open_sftp(); names = s.listdir(root)
record = dict(time_cst=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat())
for name in ['controller.pid', 'controller.exit', 'run.log', 'receipt.json', 'rows.jsonl']:
    if name in names:
        s.get(root + '/' + name, str(archive / name))
        if name.endswith('.exit'): record[name] = (archive / name).read_text().strip()
        if name == 'receipt.json': record[name] = json.loads((archive / name).read_bytes())
        if name == 'run.log': record['log_tail'] = (archive / name).read_text().splitlines()[-8:]
pid = int((archive / 'controller.pid').read_text())
probe = "from pathlib import Path;import json,subprocess;pid=" + str(pid) + ";p=Path('/proc')/str(pid);r={'pid':pid,'live':p.exists()};r['cmdline']=p.joinpath('cmdline').read_bytes().replace(b'\\0',b' ').decode() if p.exists() else '';r['children']=p.joinpath('task',str(pid),'children').read_text() if p.exists() else '';r['gpu']=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader']).decode();print(json.dumps(r))"
_, out, err = c.exec_command('/root/miniconda3/envs/bdetr/bin/python -c ' + shlex.quote(probe), timeout=30)
record['process'] = json.loads(out.read()); assert out.channel.recv_exit_status() == 0, err.read().decode()
s.close(); c.close()
(archive / 'observation_latest.json').write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
print(json.dumps(record), flush=True)
