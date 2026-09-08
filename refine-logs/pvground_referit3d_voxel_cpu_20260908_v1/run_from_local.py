import datetime
import hashlib
import json
import os
from pathlib import Path
import shlex

import paramiko

repo = Path('C:/Users/gb/.codex_mcln_g0_20260905')
archive = repo / 'refine-logs/pvground_referit3d_voxel_cpu_20260908_v1'
archive.mkdir()
root = '/root/autodl-tmp/mcln_pvground_referit3d_voxel_cpu_20260908_v1'
runtime = '/root/autodl-tmp/mcln_pvground_runtime_20260908_v1'
selection_root = '/root/autodl-tmp/mcln_referit3d_appearance_getitem_20260908_v1'
audit = (repo / 'scripts/audit_pvground_referit3d_voxels.py').read_bytes()
compile(audit, 'audit.py', 'exec')
spec = dict(runtime=runtime, selection_root=selection_root,
    selection_manifest_sha256=hashlib.sha256((repo / 'refine-logs/referit3d_appearance_getitem_20260908_v1/manifest.json').read_bytes()).hexdigest(),
    seed=2027, batch_size=4, rows='the already-fixed 12 language and 4 detection samples per dataset',
    augmentation=[False, True], train_test_processors='compare both on each same actual sample',
    script_sha256=hashlib.sha256(audit).hexdigest(),
    scope='CPU real input integration only; no model, optimizer, checkpoints, or formal rows',
    time_cst=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat())
controller = '''import datetime
import json
import os
from pathlib import Path
import subprocess

root = Path(__file__).parent
runtime = Path("/root/autodl-tmp/mcln_pvground_runtime_20260908_v1")
environment = os.environ.copy()
environment.update(json.loads((runtime / "env_spec.json").read_bytes())["env"])
environment.update(CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
(root / "controller.pid").write_text(str(os.getpid()) + "\\n")
with (root / "audit.log").open("xb") as log:
    result = subprocess.run([str(runtime / "venv/bin/python"), "-u", str(root / "audit.py"), "--spec", str(root / "spec.json")], env=environment, stdout=log, stderr=subprocess.STDOUT)
(root / "audit.exit").write_text(str(result.returncode) + "\\n")
(root / "controller.exit").write_text(str(result.returncode) + "\\n")
raise SystemExit(result.returncode)
'''.encode()
compile(controller, 'controller.py', 'exec')
files = {'audit.py': audit, 'controller.py': controller, 'spec.json': (json.dumps(spec, indent=2) + '\n').encode()}
client = paramiko.SSHClient()
client.load_system_host_keys()
client.connect('region-9.autodl.pro', port=33476, username='root', password=os.environ['MCLN_SSH_PASSWORD'], timeout=30)
sftp = client.open_sftp()
with sftp.open(runtime + '/env_spec.json', 'rb') as stream:
    runtime_spec = json.loads(stream.read())
assert hashlib.sha256(json.dumps(runtime_spec, sort_keys=True, separators=(',', ':')).encode()).hexdigest() == '966235b2ead7fa5a1de63e9537745b457374a4e5749ac4a7a26566b7b630c82c'
with sftp.open(selection_root + '/manifest.json', 'rb') as stream:
    assert hashlib.sha256(stream.read()).hexdigest() == spec['selection_manifest_sha256']
preflight_code = "import datetime,json,os,shutil,socket; print(json.dumps(dict(time_cst=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),uid=os.getuid(),hostname=socket.gethostname(),disk_free=shutil.disk_usage('/root/autodl-tmp').free,main_controllers={str(pid):os.path.exists('/proc/'+str(pid)) for pid in [5874,6398,6898]})))"
_, out, err = client.exec_command('/root/miniconda3/bin/python -c ' + shlex.quote(preflight_code), timeout=30)
preflight = json.loads(out.read())
assert out.channel.recv_exit_status() == 0 and not err.read()
assert preflight['uid'] == 0 and preflight['disk_free'] > 1024**3 and all(preflight['main_controllers'].values())
(archive / 'preflight.json').write_text(json.dumps(preflight, indent=2) + '\n')
sftp.mkdir(root)
for name, raw in files.items():
    (archive / name).write_bytes(raw)
    with sftp.open(root + '/' + name, 'wb') as stream:
        stream.write(raw)
    with sftp.open(root + '/' + name, 'rb') as stream:
        assert stream.read() == raw
argv = ['screen', '-dmS', 'mcln_pvg_referit_voxel_cpu_v1', '/root/miniconda3/bin/python', root + '/controller.py']
_, out, err = client.exec_command(' '.join(shlex.quote(item) for item in argv), timeout=30)
assert out.channel.recv_exit_status() == 0 and not out.read() and not err.read()
observation = dict(root=root, screen='mcln_pvg_referit_voxel_cpu_v1', argv=argv,
    time_cst=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
    estimated_seconds=180, planned_first_result_check_seconds=180,
    spec_sha256=hashlib.sha256(files['spec.json']).hexdigest(), main_scan_source_changed=False)
(archive / 'launch.json').write_text(json.dumps(observation, indent=2) + '\n')
print('CPU_AUDIT_LAUNCHED ' + json.dumps(dict(preflight=preflight, launch=observation)), flush=True)
sftp.close()
client.close()
