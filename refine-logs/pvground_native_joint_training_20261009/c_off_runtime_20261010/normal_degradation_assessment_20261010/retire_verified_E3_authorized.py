"""Delete only the archived inferior E3 latest; the separate cache copy remains owned."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess

root = Path(__file__).resolve().parent
native = root.parent
assert not (root/'EXACT_E3_WEIGHT_RETIREMENT.json').exists()
panel = json.loads((root/'PANEL_RESULT.json').read_bytes())
assert panel['panel_exit_code'] == 0
panel_receipt = json.loads((root/'actual_panel/results/PANEL_RECEIPT.json').read_bytes())
assert panel_receipt['status'] == 'READ_ONLY_RETAINED_E0_VS_FIXED_E3_SAME64_VALIDATION_INPUTS_COMPLETE'
assert panel_receipt['new_optimizer_steps'] == 0 and len(panel_receipt['results']) == 2
archive = json.loads((native/'normal_terminal_recovery_20261010/ARCHIVE_COMPLETE.json').read_bytes())
latest = next(row for row in archive['files'] if row['identity']['path'].endswith('/latest.pth'))
best = next(row for row in archive['files'] if row['identity']['path'].endswith('/best.pth'))
rows = [latest]
expected_paths = {
    '/root/autodl-tmp/pvground_native_joint_training_20261010/normal/logs/scanrefer/extremal_support/1791573559/latest.pth'}
assert {row['identity']['path'] for row in rows} == expected_paths
assert sum(row['identity']['bytes'] for row in rows) == 841676832
for row in rows:
    path = Path(row['local'])
    assert path.stat().st_size == row['identity']['bytes']
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):
            value.update(block)
    assert value.hexdigest() == row['identity']['sha256']
spec = json.loads((native/'init_manifests/extremal_support.json').read_bytes())
protected = [spec[key] for key in ('official_checkpoint','g_checkpoint','support_checkpoint','span_checkpoint')]
protected += [best['identity']['path'],
    '/root/autodl-tmp/mcln_pvground_nr_checkpoint_inspection_20260908_v1/PV-Ground_NR3D.pth',
    '/root/autodl-tmp/mcln_pvground_sr_checkpoint_inspection_20260908_v1/PV-Ground_SR3D.pth']
assert not expected_paths.intersection(protected)
code = r'''import datetime,hashlib,json,shutil,sys
from pathlib import Path
b=json.load(sys.stdin)
status=json.loads(Path('/root/autodl-tmp/pvground_native_joint_training_20261010/normal/normal_status.json').read_bytes())
assert status['status']=='complete' and status['exit_code']==0
receipt=json.loads(Path('/root/autodl-tmp/pvground_normal_degradation_panel_20261010/results/PANEL_RECEIPT.json').read_bytes())
assert receipt['new_optimizer_steps']==0 and receipt['actual_neural_forwards']==16
expected={
 '/root/autodl-tmp/pvground_native_joint_training_20261010/normal/logs/scanrefer/extremal_support/1791573559/latest.pth'}
assert {row['identity']['path'] for row in b['files']}==expected
assert not expected.intersection(b['protected']) and all(Path(p).is_file() for p in b['protected'])
paths=[]
for row in b['files']:
 identity=row['identity'];p=Path(identity['path'])
 assert p.resolve()==p and not p.is_symlink() and p.stat().st_size==identity['bytes']
 assert p.parts[:3]==('/','root','autodl-tmp')
 h=hashlib.sha256()
 with p.open('rb') as f:
  for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
 assert h.hexdigest()==identity['sha256'];paths.append(p)
before=shutil.disk_usage('/root/autodl-tmp').free
for p in paths:p.unlink()
assert all(not p.exists() for p in paths) and all(Path(p).is_file() for p in b['protected'])
result=dict(status='ONLY_ARCHIVED_INFERIOR_E3_WEIGHT_RETIRED',
 time_cst=datetime.datetime.now().astimezone().isoformat(),files=b['files'],deleted_count=1,
 deleted_bytes=sum(row['identity']['bytes'] for row in b['files']),
 data_free_before=before,data_free_after=shutil.disk_usage('/root/autodl-tmp').free,
 best_and_all_author_G_support_span_dependencies_preserved=True,
 datasets_runtime_and_V99_untouched=True,local_full_archives_preserved=True,new_neural_calls=0)
destination=Path('/root/autodl-tmp/pvground_normal_degradation_panel_20261010/e3_cleanup_receipt.json')
assert not destination.exists();destination.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))
'''
witness = json.loads((native.parent/'SCP_TRANSPORT_WITNESS.json').read_bytes())
environment = dict(os.environ,
    SSH_ASKPASS='C:/Users/gb/.codex/tmp/pvground_native_joint_training_20261009/NativeSshAskPass.exe',
    SSH_ASKPASS_REQUIRE='force',DISPLAY='codex-byte-transfer')
runtime = '/root/autodl-tmp/mcln_pvground_runtime_20260908_v1/venv/bin/python'
command = ['C:/Windows/System32/OpenSSH/ssh.exe','-T','-p','33476','-o','ProxyCommand=none',
    '-o','StrictHostKeyChecking=yes','-o','UserKnownHostsFile=C:/Users/gb/.codex/tmp/pvground_native_joint_training_20261009/native_scp_known_hosts',
    '-o','HostKeyAlgorithms='+witness['negotiated_host_key_algorithm'],'-o','NumberOfPasswordPrompts=1',
    'root@region-9.autodl.pro',shlex.join([runtime,'-B','-u','-c',code])]
response = subprocess.run(command,env=environment,input=json.dumps(dict(files=rows,protected=protected)).encode(),
    stdout=subprocess.PIPE,stderr=subprocess.PIPE,creationflags=subprocess.CREATE_NO_WINDOW)
(root/'E3_RETIREMENT_STDOUT.json').write_bytes(response.stdout)
(root/'E3_RETIREMENT_STDERR.txt').write_bytes(response.stderr)
(root/'E3_RETIREMENT_EXIT.json').write_text(json.dumps(dict(exit_code=response.returncode))+'\n',encoding='utf-8')
assert response.returncode == 0,'Inspect exact remote artifact/receipt state before any cleanup repair'
result = json.loads(response.stdout)
(root/'EXACT_E3_WEIGHT_RETIREMENT.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
print(json.dumps({key:result[key] for key in ('status','time_cst','deleted_count','deleted_bytes','data_free_after')}),flush=True)
