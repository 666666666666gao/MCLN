"""Resume the actually timed-out archive, transferring only missing or incomplete files."""
import datetime
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shlex
import tarfile
import paramiko


local = Path(__file__).resolve().parent
destination = local / 'complete_fit'
recovery = local / 'collector_recovery'
assert destination.is_dir() and not (destination / 'INTAKE.json').exists()
assert not recovery.exists()
wait = json.loads((local / 'fit_wait.json').read_bytes())
assert wait['observer_closed'] and wait['terminal']['exitcode'] == 0
assert not wait['terminal']['controller_alive']
spec = json.loads((local / 'pair_spec.json').read_bytes())
recovery.mkdir()
client = paramiko.SSHClient()
client.load_system_host_keys()
client.connect('region-9.autodl.pro', port=33476, username='root',
               password=os.environ['MCLN_SSH_PASSWORD'], timeout=30)
inventory_code = r'''import hashlib,json,sys
from pathlib import Path
root=Path(sys.argv[1]);assert root.resolve()==root
assert (root/'fit_controller.exit').read_text().strip()=='0'
assert json.loads((root/'fit_status.json').read_bytes())['status']=='complete'
files=[]
for path in sorted(root.rglob('*')):
    if not path.is_file() or path.suffix in ('.pth','.pt','.tmp'):continue
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    files.append(dict(name=str(path.relative_to(root)),bytes=path.stat().st_size,sha256=h.hexdigest()))
print(json.dumps(dict(files=files,weights_copied=0)))
'''
_, stdout, stderr = client.exec_command(shlex.join([
    spec['runtime'] + '/venv/bin/python', '-B', '-c', inventory_code, spec['root']]), timeout=180)
manifest_bytes = stdout.read()
error = stderr.read().decode()
assert stdout.channel.recv_exit_status() == 0, error
manifest = json.loads(manifest_bytes)
assert len(manifest['files']) == 2423
assert sum(row['bytes'] for row in manifest['files']) == 230030167
assert sum(row['name'].endswith('.npz') for row in manifest['files']) == 2378
(recovery / 'REMOTE_MANIFEST.json').write_bytes(manifest_bytes)
expected = {row['name']: row for row in manifest['files']}
assert len(expected) == len(manifest['files'])
existing = {path.relative_to(destination).as_posix() for path in destination.rglob('*') if path.is_file()}
assert existing.issubset(expected)
verified, remaining, incomplete = [], [], []
for row in manifest['files']:
    path = destination.joinpath(*PurePosixPath(row['name']).parts)
    assert destination.resolve() in path.resolve().parents
    if path.exists():
        raw = path.read_bytes()
        if len(raw) == row['bytes'] and hashlib.sha256(raw).hexdigest() == row['sha256']:
            verified.append(row['name'])
            continue
        incomplete.append(dict(name=row['name'],bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest()))
    remaining.append(row)
record = dict(status='ACTUAL_TIMEOUT_PARTIAL_ARCHIVE_CLASSIFIED',
    time_cst=datetime.datetime.now().astimezone().isoformat(),
    original_native_session=95046, original_collector_pid=33052, original_exitcode=1,
    original_failure='paramiko.channel.recv PipeTimeout -> socket.timeout -> TimeoutError in tar member read',
    verified_files=len(verified),verified_bytes=sum(expected[name]['bytes'] for name in verified),
    remaining_files=len(remaining),remaining_bytes=sum(row['bytes'] for row in remaining),
    incomplete_files=incomplete,verified_files_retransmitted=0,weights_copied=0,
    neural_forwards=0,optimizer_updates=0,automatic_retries=0)
(recovery / 'PARTIAL_CLASSIFICATION.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
print(json.dumps(record),flush=True)
archive_code = r'''import json,subprocess,sys
from pathlib import Path
b=json.load(sys.stdin);root=Path(b['root']);assert root.resolve()==root
assert (root/'fit_controller.exit').read_text().strip()=='0'
assert json.loads((root/'fit_status.json').read_bytes())['status']=='complete'
for name in b['names']:
    path=root/name;assert root in path.resolve().parents and path.is_file()
    assert path.suffix not in ('.pth','.pt','.tmp')
subprocess.run(['tar','-cf','-','-C',str(root)]+b['names'],check=True,stdout=sys.stdout.buffer)
'''
stdin, stdout, stderr = client.exec_command(shlex.join([
    spec['runtime'] + '/venv/bin/python', '-B', '-c', archive_code]), timeout=180)
# The original channel actually timed out during this large transfer; no per-read deadline.
stdout.channel.settimeout(None)
stdin.write(json.dumps(dict(root=spec['root'],names=[row['name'] for row in remaining])))
stdin.channel.shutdown_write()
remaining_by_name = {row['name']: row for row in remaining}
seen = []
with tarfile.open(fileobj=stdout, mode='r|') as archive:
    for member in archive:
        assert member.isfile() and member.name in remaining_by_name
        row = remaining_by_name[member.name]
        assert member.size == row['bytes']
        path = destination.joinpath(*PurePosixPath(member.name).parts)
        assert destination.resolve() in path.resolve().parents
        path.parent.mkdir(parents=True, exist_ok=True)
        h, count = hashlib.sha256(), 0
        stream = archive.extractfile(member)
        # Only a classified missing or incomplete file can be written here.
        with path.open('wb') as output:
            for block in iter(lambda:stream.read(4*1024*1024), b''):
                output.write(block);h.update(block);count += len(block)
        assert count == row['bytes'] and h.hexdigest() == row['sha256']
        seen.append(member.name)
assert len(seen) == len(remaining) and set(seen) == set(remaining_by_name)
error = stderr.read().decode()
assert stdout.channel.recv_exit_status() == 0, error
client.close()
for row in manifest['files']:
    path = destination.joinpath(*PurePosixPath(row['name']).parts)
    assert path.stat().st_size == row['bytes']
    assert hashlib.sha256(path.read_bytes()).hexdigest() == row['sha256']
record = dict(time_cst=datetime.datetime.now().astimezone().isoformat(),
    status='CLOSED_FIT_ARTIFACTS_COLLECTED',files=manifest['files'],
    files_copied=len(manifest['files']),total_bytes=sum(row['bytes'] for row in manifest['files']),
    weights_copied=0,model_or_optimizer_replayed=False,
    verified_original_files_retained=len(verified),resumed_files=len(seen),
    source='original_partial_plus_one_missing_files_transfer',original_collector_exitcode=1)
(destination / 'INTAKE.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
(recovery / 'RECOVERY_COMPLETE.json').write_text(json.dumps(
    {key:value for key,value in record.items() if key!='files'},indent=2)+'\n',encoding='utf-8')
print(json.dumps({key:value for key,value in record.items() if key!='files'}),flush=True)
