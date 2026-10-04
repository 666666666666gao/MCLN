"""One ETA-based formal observer; no relaunches and240-second later intervals."""
import datetime
import json
import os
from pathlib import Path
import shlex
import time
import paramiko

local = Path(__file__).resolve().parent
launch = json.loads((local / 'launch.json').read_bytes())
first = datetime.datetime.fromisoformat(launch['time_cst']).timestamp() + launch['first_check_seconds']
print(json.dumps(dict(status='SCHEDULED_READ_ONLY_WAIT', first_check_epoch=first,
    wait_seconds=max(0, first - time.time()), later_poll_seconds=240)), flush=True)
time.sleep(max(0, first - time.time()))
spec = json.loads((local / 'evidence_hidden_fit_spec.json').read_bytes())
probe = '''
import json,shutil,subprocess,sys
from pathlib import Path
root=Path(sys.argv[1]);pid=int(sys.argv[2]);state=json.loads((root/'status.json').read_bytes())
result=dict(status=state,controller_alive=Path('/proc/%d'%pid).exists(),
    directory_free_bytes=shutil.disk_usage(root).free,system_free_bytes=shutil.disk_usage('/').free,
    gpu=subprocess.check_output(['nvidia-smi','--query-gpu=utilization.gpu,memory.used','--format=csv,noheader,nounits'],text=True).strip())
exitfile=root/'controller.exit'
result['exitcode']=int(exitfile.read_text()) if exitfile.exists() else None
phase=state['stage'].split('/');log=root/phase[0]/(phase[1]+'.log')
result['log_tail']=log.read_text(errors='replace').splitlines()[-12:] if log.exists() else []
print(json.dumps(result))
'''
count = 0
while True:
    client = paramiko.SSHClient()
    client.load_system_host_keys()
    client.connect('region-9.autodl.pro', port=33476, username='root',
        password=os.environ['MCLN_SSH_PASSWORD'], timeout=30)
    _, stdout, stderr = client.exec_command(shlex.join([spec['runtime'] + '/venv/bin/python', '-c', probe,
        launch['root'], launch['process'].split()[0]]), timeout=60)
    raw = stdout.read()
    assert stdout.channel.recv_exit_status() == 0, stderr.read().decode()
    client.close()
    result = json.loads(raw)
    result['time_cst'] = datetime.datetime.now().astimezone().isoformat()
    count += 1
    (local / ('observation_%02d.json' % count)).write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result), flush=True)
    if result['status']['status'] != 'running':
        assert not result['controller_alive'] and result['exitcode'] is not None
        (local / 'wait.json').write_text(json.dumps(dict(time_cst=result['time_cst'], terminal=result,
            remote_queries=count, observer_closed=True, no_restart=True), indent=2) + '\n', encoding='utf-8')
        break
    time.sleep(240)
