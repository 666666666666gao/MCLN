import datetime
import hashlib
import json
import os
from pathlib import Path
import shlex
import paramiko

repo = Path('C:/Users/gb/.codex_mcln_g0_20260905')
root = '/root/autodl-tmp/mcln_pvground_scanrefer_endpoint_audit_20260908_v1'
training = '/root/autodl-tmp/mcln_pvground_scanrefer_finetune_20260908_v1'
archive = repo/'refine-logs/pvground_scanrefer_endpoint_audit_20260908_v1'
files = {'audit.py': repo/'scripts/audit_pvground_scanrefer_finetune.py',
         'plan.md': repo/'docs/PVG_SCANREFER_ENDPOINT_AUDIT_PLAN_2026-09-08.md'}
spec = {'training_root': training, 'training_controller_pid': 5874,
        'training_spec_sha256': hashlib.sha256((repo/'refine-logs/pvground_scanrefer_finetune_20260908_v1/spec.json').read_bytes()).hexdigest(),
        'first_check_cst': '2026-09-08T08:02:30+08:00', 'poll_seconds': 300,
        'files': {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in files.items()},
        'formal_evaluation_launch': False}
controller = '''import datetime,hashlib,importlib.util,json,subprocess,time
from pathlib import Path
root=Path(ROOT)
spec=json.loads((root/'spec.json').read_bytes())
training=Path(spec['training_root'])
for name,digest in spec['files'].items():
    assert hashlib.sha256((root/name).read_bytes()).hexdigest()==digest,name
assert hashlib.sha256((training/'spec.json').read_bytes()).hexdigest()==spec['training_spec_sha256']
deadline=datetime.datetime.fromisoformat(spec['first_check_cst']).timestamp()
time.sleep(max(0,deadline-time.time()))
module_spec=importlib.util.spec_from_file_location('endpoint_auditor',str(root/'audit.py'))
module=importlib.util.module_from_spec(module_spec);module_spec.loader.exec_module(module)
input_spec=json.loads((training/'spec.json').read_bytes())
manifest=json.loads(Path(input_spec['input_manifest']).read_bytes())
assert module.sha(manifest['split_protocol'])==manifest['split_protocol_sha256']
ids=json.loads(Path(manifest['split_protocol']).read_bytes())['row_ids']['holdout']
initial_done=False
while True:
    if not initial_done and (training/'initial/receipt.json').is_file():
        rows,metrics,checks=module.audit_stage(training,'initial',ids)
        result={'integrity_pass':True,'stage':'initial','formal_rows':0,'metrics':metrics,'checks':checks,
                'receipt_sha256':module.sha(training/'initial/receipt.json'),
                'time_cst':datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()}
        with (root/'initial_audit.json').open('x') as stream:json.dump(result,stream,indent=2);stream.write('\\n')
        print('PVG_INITIAL_AUDIT_COMPLETE '+json.dumps(result),flush=True)
        initial_done=True
    if (training/'controller.exit').is_file():
        code=int((training/'controller.exit').read_text())
        if code!=0:
            result={'status':'dependency_failed','training_exit':code,'formal_rows':0}
            (root/'dependency_failure.json').write_text(json.dumps(result)+'\\n')
            print('PVG_DEPENDENCY_FAILED '+json.dumps(result),flush=True)
            raise SystemExit(code)
        command=['/root/miniconda3/envs/bdetr/bin/python','-u',str(root/'audit.py'),'--root',str(training),'--out',str(root/'audit.json')]
        result=subprocess.run(command)
        (root/'audit.exit').write_text(str(result.returncode)+'\\n')
        raise SystemExit(result.returncode)
    process=Path('/proc')/str(spec['training_controller_pid'])/'cmdline'
    assert process.is_file() and (str(training)+'/controller.py').encode() in process.read_bytes(),'original training controller disappeared without an exit receipt'
    time.sleep(spec['poll_seconds'])
'''.replace('ROOT', repr(root))
compile(controller, 'queue.py', 'exec')
wrapper = '''import subprocess
from pathlib import Path
root=Path(ROOT)
result=subprocess.run(['/root/miniconda3/envs/bdetr/bin/python','-u',str(root/'queue.py')])
(root/'controller.exit').write_text(str(result.returncode)+'\\n')
raise SystemExit(result.returncode)
'''.replace('ROOT', repr(root))
compile(wrapper, 'controller.py', 'exec')
client = paramiko.SSHClient()
client.load_system_host_keys()
client.connect('region-9.autodl.pro', port=33476, username='root', password=os.environ['MCLN_SSH_PASSWORD'], timeout=30)
sftp = client.open_sftp()
with sftp.open(training+'/spec.json', 'rb') as stream:
    assert hashlib.sha256(stream.read()).hexdigest() == spec['training_spec_sha256']
_, stdout, stderr = client.exec_command('pgrep -af '+shlex.quote('^/root/miniconda3/envs/bdetr/bin/python -u '+training+'/controller.py$'), timeout=30)
process = stdout.read().decode().strip()
assert stdout.channel.recv_exit_status() == 0 and process.startswith('5874 '), stderr.read().decode()
archive.mkdir()
sftp.mkdir(root)
data = {name: path.read_bytes() for name, path in files.items()}
data.update({'spec.json': (json.dumps(spec, indent=2)+'\n').encode(), 'queue.py': controller.encode(), 'controller.py': wrapper.encode()})
for name, raw in data.items():
    assert os.environ['MCLN_SSH_PASSWORD'].encode() not in raw
    (archive/name).write_bytes(raw)
    with sftp.open(root+'/'+name, 'wx') as stream:
        stream.write(raw)
    with sftp.open(root+'/'+name, 'rb') as stream:
        assert stream.read() == raw
screen = 'mcln_pvg_endpoint_audit_v1'
inner = 'exec /root/miniconda3/envs/bdetr/bin/python -u '+shlex.quote(root+'/controller.py')+' > '+shlex.quote(root+'/run.log')+' 2>&1'
_, stdout, stderr = client.exec_command('screen -dmS '+screen+' bash -c '+shlex.quote(inner), timeout=30)
assert stdout.channel.recv_exit_status() == 0, stderr.read().decode()
_, stdout, stderr = client.exec_command('pgrep -af '+shlex.quote('^/root/miniconda3/envs/bdetr/bin/python -u '+root+'/controller.py$'), timeout=30)
process = stdout.read().decode().strip()
assert stdout.channel.recv_exit_status() == 0 and process, stderr.read().decode()
record = {'time_cst': datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
          'process': process, 'screen': screen, 'first_check_cst': spec['first_check_cst'], 'poll_seconds': 300,
          'formal_rows': 0, 'gpu_forwards': 0, 'training_modified': False}
raw = (json.dumps(record, indent=2)+'\n').encode()
(archive/'launch.json').write_bytes(raw)
(archive/'launch_from_local.py').write_bytes(Path(__file__).read_bytes())
with sftp.open(root+'/launch.json', 'wx') as stream:
    stream.write(raw)
sftp.close()
client.close()
print('PVG_ENDPOINT_AUDIT_QUEUED '+json.dumps(record), flush=True)
