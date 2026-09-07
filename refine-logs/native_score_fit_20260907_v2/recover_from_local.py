import json
import os
from pathlib import Path
import shlex
import paramiko

repo = Path('C:/Users/gb/.codex_mcln_g0_20260905')
archive = repo / 'refine-logs/native_score_fit_20260907_v2'
remote = '/root/autodl-tmp/mcln_native_score_fit_20260907_v2'
c = paramiko.SSHClient()
c.load_system_host_keys()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect('region-9.autodl.pro', port=33476, username='root', password=os.environ['MCLN_SSH_PASSWORD'], timeout=30)
s = c.open_sftp()
s.put(str(repo / 'scripts/recover_native_score_fit_receipt.py'), remote + '/recover_native_score_fit_receipt.py')
command = ['/root/miniconda3/envs/bdetr/bin/python', remote + '/recover_native_score_fit_receipt.py', '--directory', remote]
_, out, err = c.exec_command(' '.join(shlex.quote(v) for v in command), timeout=45)
stdout, stderr = out.read(), err.read()
code = out.channel.recv_exit_status()
(archive / 'recovery_stdout.txt').write_bytes(stdout)
(archive / 'recovery_stderr.txt').write_bytes(stderr)
(archive / 'recovery_exit.json').write_bytes(json.dumps({'exit': code, 'command': command}).encode())
assert code == 0, stderr.decode()
(archive / 'recover_from_local.py').write_bytes(Path(__file__).read_bytes())
print(stdout.decode(), flush=True)
s.close()
c.close()
