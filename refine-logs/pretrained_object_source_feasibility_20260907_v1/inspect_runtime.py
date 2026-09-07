import hashlib
import json
import os
from pathlib import Path
import shlex
import paramiko

repo = Path('C:/Users/gb/.codex_mcln_g0_20260905')
outdir = repo / 'refine-logs/pretrained_object_source_feasibility_20260907_v1'
outdir.mkdir()
c = paramiko.SSHClient()
c.load_system_host_keys()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect('region-9.autodl.pro', port=33476, username='root', password=os.environ['MCLN_SSH_PASSWORD'], timeout=30)
code = '''import datetime,importlib.util,json,os,shutil,socket,subprocess,sys
import pkg_resources
names=['torch','torchvision','transformers','dgl','timm','einops','omegaconf','torch_redstone','open_clip','clip','ftfy','huggingface_hub']
versions={p.key:p.version for p in pkg_resources.working_set}
print(json.dumps(dict(time_cst=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),uid=os.getuid(),host=socket.gethostname(),python=sys.version,packages={n:dict(available=importlib.util.find_spec(n) is not None,version=versions.get(n.replace('_','-'))) for n in names},disk=shutil.disk_usage('/root/autodl-tmp')._asdict(),gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,memory.total,utilization.gpu','--format=csv,noheader,nounits']).decode(),research_processes=[s for s in subprocess.check_output(['ps','-eo','pid,ppid,args']).decode().splitlines() if 'python -u run_' in s and '-c ' not in s])))
'''
_, stdout, stderr = c.exec_command('/root/miniconda3/envs/bdetr/bin/python -c ' + shlex.quote(code), timeout=40)
raw, error = stdout.read(), stderr.read()
assert stdout.channel.recv_exit_status() == 0, error.decode()
profile = json.loads(raw)
assert profile['uid'] == 0
(outdir / 'environment_inventory.json').write_bytes(raw)
(outdir / 'inspect_runtime.py').write_bytes(Path(__file__).read_bytes())
print(json.dumps(profile), flush=True)
c.close()
