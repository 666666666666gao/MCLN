"""Read only the specific launched preflight controller and its phase exits."""
import datetime
import json
import os
from pathlib import Path
import shlex

import paramiko


local = Path(__file__).parent
launch = json.loads((local / 'preflight_launch.json').read_bytes())
spec = json.loads((local / 'whole_range_spec.json').read_bytes())
probe = '''
import json,subprocess,sys
from pathlib import Path
root=Path(sys.argv[1]);pid=int(sys.argv[2])
process=subprocess.run(['ps','-p',str(pid),'-o','pid=,args='],stdout=subprocess.PIPE,text=True)
status=json.loads((root/'status.json').read_bytes())
exit_path=root/'controller.exit'
phases={}
for arm in ('local_range','whole_range'):
    directory=root/arm
    phases[arm]=dict(exit_code=(int((directory/'preflight.exit').read_text()) if (directory/'preflight.exit').exists() else None),
        receipt_available=(directory/'preflight.json').exists(),
        log_tail=((directory/'preflight.log').read_text()[-6000:] if (directory/'preflight.log').exists() else ''))
print(json.dumps(dict(status=status,controller_alive=process.returncode==0,
    controller_process=process.stdout.strip(),controller_exit=(int(exit_path.read_text()) if exit_path.exists() else None),phases=phases)))
'''
client = paramiko.SSHClient()
client.load_system_host_keys()
client.connect('region-9.autodl.pro', port=33476, username='root',
    password=os.environ['MCLN_SSH_PASSWORD'], timeout=30)
command = shlex.join([spec['runtime'] + '/venv/bin/python', '-c', probe,
    launch['root'], launch['process'].split()[0]])
_, stdout, stderr = client.exec_command(command, timeout=60)
raw = stdout.read()
assert stdout.channel.recv_exit_status() == 0, stderr.read().decode()
record = json.loads(raw)
record['observed_cst'] = datetime.datetime.now().astimezone().isoformat()
name = 'preflight_observation_' + datetime.datetime.now().strftime('%H%M%S') + '.json'
with (local / name).open('x', encoding='utf-8') as handle:
    handle.write(json.dumps(record, indent=2) + '\n')
client.close()
print(json.dumps(record), flush=True)
