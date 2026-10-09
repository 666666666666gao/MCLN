"""Verify the first actual fit update after the estimated warm startup interval."""
import datetime,json,os,shlex,time
from pathlib import Path
import paramiko
p=Path(__file__).resolve().parent
assert not (p/'FIT_FIRST_UPDATE.json').exists()
launch=json.loads((p/'fit_launch.json').read_bytes())
first=datetime.datetime.fromisoformat(launch['time_cst']).timestamp()+240
(p/'FIT_FIRST_UPDATE_WAIT.json').write_text(json.dumps(dict(first_check_cst=datetime.datetime.fromtimestamp(first).astimezone().isoformat(),poll_seconds=240,expected_startup_seconds=240,estimate_basis='Same warm parent and dataset setup as passed M0; first fit batch checked once after4min, then240s only if not yet updated',remote_reads=0),indent=2)+'\n')
time.sleep(max(0,first-time.time()))
code=r"""import datetime,json,sys
from pathlib import Path
r=Path(sys.argv[1]);pid=int(sys.argv[2]);proc=Path('/proc')/str(pid)
status=json.loads((r/'fit_status.json').read_bytes());exitcode=int((r/'fit_controller.exit').read_text()) if (r/'fit_controller.exit').exists() else None
first=None
if (r/'train.jsonl').exists():
 with (r/'train.jsonl').open() as stream:
  line=stream.readline()
 if line:first=json.loads(line)
print(json.dumps(dict(time_cst=datetime.datetime.now().astimezone().isoformat(),controller_alive=proc.exists(),status=status,exitcode=exitcode,first_update=first)))
"""
reads=0
while True:
 c=paramiko.SSHClient();c.load_system_host_keys();c.connect('region-9.autodl.pro',port=33476,username='root',password=os.environ['MCLN_SSH_PASSWORD'],timeout=30)
 _,out,err=c.exec_command(shlex.join(['/root/autodl-tmp/mcln_pvground_runtime_20260908_v1/venv/bin/python','-B','-c',code,launch['root'],str(launch['controller_pid'])]),timeout=30)
 raw=out.read();returncode=out.channel.recv_exit_status();errors=err.read();c.close();assert returncode==0,errors.decode()
 j=json.loads(raw);reads+=1
 (p/'FIT_STARTUP_OBSERVATION.json').write_bytes(raw)
 if j['first_update'] is not None:
  assert j['first_update']['step']==1
  (p/'FIT_FIRST_UPDATE.json').write_bytes(raw)
  print(json.dumps(dict(status='ACTUAL_FORMAL_FIT_FIRST_UPDATE_OBSERVED',time_cst=j['time_cst'],queries=reads,first_update=j['first_update'])),flush=True);break
 if j['exitcode'] is not None:
  (p/'FIT_STARTUP_CLOSED_WITHOUT_UPDATE.json').write_bytes(raw)
  raise SystemExit('fit closed without first update; no restart attempted')
 assert j['controller_alive'];time.sleep(240)
