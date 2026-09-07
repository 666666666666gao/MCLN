import datetime
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import paramiko

repo = Path('C:/Users/gb/.codex_mcln_g0_20260905')
archive = repo / 'refine-logs/native_score_fit_20260907_v2'
remote = '/root/autodl-tmp/mcln_native_score_fit_20260907_v2'
c = paramiko.SSHClient()
c.load_system_host_keys()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect('region-9.autodl.pro', port=33476, username='root', password=os.environ['MCLN_SSH_PASSWORD'], timeout=30)
s = c.open_sftp()
names = s.listdir(remote)
launch = json.loads((archive / 'launch.json').read_text())
pids = ','.join(line.split()[0] for line in launch['processes'])
_, out, err = c.exec_command('ps -p ' + pids + ' -o pid,ppid,etime,args', timeout=30)
processes = out.read().decode()
process_exit = out.channel.recv_exit_status()
with s.open(remote + '/run.log', 'rb') as stream:
    log = stream.read()
(archive / 'run_log.txt').write_bytes(log)
observation = {'time_cst': datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
               'processes': processes, 'process_exit': process_exit, 'files': names}
if 'controller.exit' not in names:
    (archive / 'latest_observation.json').write_bytes(json.dumps(observation, indent=2).encode())
    print(json.dumps(observation), flush=True)
    print(log.decode()[-2500:], flush=True)
    s.close()
    c.close()
    raise SystemExit(0)
with s.open(remote + '/controller.exit', 'rb') as stream:
    code = int(stream.read())
observation['controller_exit'] = code
(archive / 'terminal_observation.json').write_bytes(json.dumps(observation, indent=2).encode())
assert code == 1  # Completed v2 rows; its final receipt path error is explicitly recovered.
for name in ['receipt.json', 'protocol.json']:
    s.get(remote + '/' + name, str(archive / name))
receipt = json.loads((archive / 'receipt.json').read_text())
assert receipt['recovery']['original_controller_exit'] == 1
for shard in receipt['shards']:
    name = shard['file']
    s.get(remote + '/' + name, str(archive / name))
    assert hashlib.sha256((archive / name).read_bytes()).hexdigest() == shard['sha256']
analyzer = 'scripts/analyze_native_score_fit_audit.py'
s.put(str(repo / analyzer), remote + '/analyze_native_score_fit_audit.py')
command = ['/root/miniconda3/envs/bdetr/bin/python', remote + '/analyze_native_score_fit_audit.py',
           '--directory', remote, '--output', remote + '/independent_analysis.json']
_, out, err = c.exec_command(' '.join(shlex.quote(v) for v in command), timeout=45)
analysis_out, analysis_err = out.read(), err.read()
code = out.channel.recv_exit_status()
(archive / 'analysis_stdout.txt').write_bytes(analysis_out)
(archive / 'analysis_stderr.txt').write_bytes(analysis_err)
assert code == 0, analysis_err.decode()
s.get(remote + '/independent_analysis.json', str(archive / 'independent_analysis.json'))
(archive / 'collect_from_local.py').write_bytes(Path(__file__).read_bytes())
print(analysis_out.decode(), flush=True)
print(json.dumps({'observation': observation, 'receipt_sha256': hashlib.sha256((archive / 'receipt.json').read_bytes()).hexdigest(),
                  'analysis_sha256': hashlib.sha256((archive / 'independent_analysis.json').read_bytes()).hexdigest()}), flush=True)
s.close()
c.close()
