import os
import shlex
from pathlib import Path
import paramiko

local = Path('C:/Users/gb/.codex_mcln_g0_20260905/refine-logs/native_mask_geometry_source_preparation_20260907_v1')
remote = '/root/autodl-tmp/mcln_native_mask_geometry_source_preparation_20260907_v1'
c = paramiko.SSHClient()
c.load_system_host_keys()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect('region-9.autodl.pro', port=33476, username='root', password=os.environ['MCLN_SSH_PASSWORD'], timeout=30)
s = c.open_sftp()
for name in ['finish_preparation.py', 'check_complete_source_v2.py']:
    s.put(str(local / name), remote + '/' + name)
cmd = 'CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 /root/miniconda3/envs/bdetr/bin/python ' + shlex.quote(remote + '/finish_preparation.py') + ' ' + shlex.quote(remote)
_, out, err = c.exec_command(cmd, timeout=55)
body, error = out.read(), err.read()
status = out.channel.recv_exit_status()
(local / 'finish_stdout.txt').write_bytes(body)
(local / 'finish_stderr.txt').write_bytes(error)
print((body + error).decode(), flush=True)
for entry in s.listdir_attr(remote):
    if entry.filename in ['receipt.json', 'check_2.txt']:
        s.get(remote + '/' + entry.filename, str(local / entry.filename))
s.close()
c.close()
raise SystemExit(status)
