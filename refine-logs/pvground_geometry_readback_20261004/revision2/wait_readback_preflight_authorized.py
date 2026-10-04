"""One scheduled read-only observer; no restarts or short remote polling."""
import datetime
import json
import os
from pathlib import Path
import shlex
import time
import paramiko

local = Path(__file__).resolve().parent
launch = json.loads((local / 'readback_preflight_launch.json').read_bytes())
assert launch['execution_status'] == 'LAUNCHED_NOT_COMPLETED'
first = datetime.datetime.fromisoformat(launch['time_cst']).timestamp() + launch['first_check_seconds']
delay = max(0, first - time.time())
print(json.dumps(dict(status='SCHEDULED_READ_ONLY_WAIT', first_check_epoch=first,
    wait_seconds=delay, later_poll_seconds=240)), flush=True)
time.sleep(delay)
probe = '''
import json,shutil,subprocess,sys
from pathlib import Path
root=Path(sys.argv[1]);pid=int(sys.argv[2]);state=json.loads((root/'status.json').read_bytes())
result=dict(status=state,controller_alive=Path('/proc/%d'%pid).exists(),
    directory_free_bytes=shutil.disk_usage(root).free,system_free_bytes=shutil.disk_usage('/').free,
    gpu=subprocess.check_output(['nvidia-smi','--query-gpu=utilization.gpu,memory.used','--format=csv,noheader,nounits'],text=True).strip())
exitfile=root/'controller.exit'
result['exitcode']=int(exitfile.read_text()) if exitfile.exists() else None
phase=state['phase'].split('/');log=root/phase[0]/(phase[1]+'.log')
result['log_tail']=log.read_text(errors='replace').splitlines()[-18:] if log.exists() else []
print(json.dumps(result))
'''
pid = launch['process'].split()[0]
spec = json.loads((local / 'evidence_visible_preflight_spec.json').read_bytes())
count = 0
while True:
    client = paramiko.SSHClient()
    client.load_system_host_keys()
    client.connect('region-9.autodl.pro', port=33476, username='root',
        password=os.environ['MCLN_SSH_PASSWORD'], timeout=30)
    command = shlex.join([spec['runtime'] + '/venv/bin/python', '-c', probe, launch['root'], pid])
    _, stdout, stderr = client.exec_command(command, timeout=60)
    raw = stdout.read()
    assert stdout.channel.recv_exit_status() == 0, stderr.read().decode()
    client.close()
    result = json.loads(raw)
    count += 1
    result['time_cst'] = datetime.datetime.now().astimezone().isoformat()
    (local / ('readback_preflight_observation_%02d.json' % count)).write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result), flush=True)
    if result['status']['status'] != 'running':
        assert not result['controller_alive'] and result['exitcode'] is not None
        (local / 'readback_preflight_wait.json').write_text(json.dumps(dict(
            time_cst=result['time_cst'], terminal=result, remote_queries=count,
            no_restart=True, observer_closed=True), indent=2) + '\n', encoding='utf-8')
        break
    time.sleep(240)
