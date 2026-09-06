import json
import os
from pathlib import Path

import paramiko

root=Path('C:/Users/gb/.codex_mcln_g0_20260905/refine-logs/scanrefer_range_preflight_20260907_v1')
client=paramiko.SSHClient()
client.load_system_host_keys()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect('region-9.autodl.pro',port=33476,username='root',password=os.environ['MCLN_SSH_PASSWORD'],timeout=30)
code="""import datetime,hashlib,json,subprocess
from pathlib import Path
root=Path('/root/autodl-tmp/mcln_scanrefer_range_preflight_20260907_v1')
processes=subprocess.check_output(['ps','-p','46874,46877','-o','pid,ppid,stat,lstart,etime,args']).decode()
assert 'run_scanrefer_range_preflight.py' in processes and 'SCREEN -DmS mcln_scanrefer_range_preflight_v1' in processes
value={'time_cst':datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),'screen_session':['46874.mcln_scanrefer_range_preflight_v1'],'python_pid':46877,'processes':processes,'manifest_sha256':hashlib.sha256((root/'input_manifest.json').read_bytes()).hexdigest(),'formal_rows':0,'checkpoint_writes':0,'preflight_started':True,'training_started':False,'launcher_observation_timeout':True,'launcher_waited_for_screen_exit':True,'actual_job_not_restarted':True,'controller_sha256':hashlib.sha256((root/'controller.sh').read_bytes()).hexdigest()}
with (root/'launch.json').open('x') as stream: json.dump(value,stream,indent=2,sort_keys=True)
print(json.dumps(value))
"""
_,out,err=client.exec_command('/root/miniconda3/envs/bdetr/bin/python - <<\'PY\'\n'+code+'\nPY',timeout=30)
raw=out.read()
assert out.channel.recv_exit_status()==0,err.read().decode()
value=json.loads(raw)
sftp=client.open_sftp()
with sftp.open('/root/autodl-tmp/mcln_scanrefer_range_preflight_20260907_v1/launch.json','rb') as stream:
    (root/'launch.json').write_bytes(stream.read())
sftp.close()
client.close()
print(json.dumps(value))
