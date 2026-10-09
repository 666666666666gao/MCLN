"""Stream a genuinely closed phase and verify every archived byte and hash."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shlex
import tarfile

import paramiko


runner = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument('--phase', choices=('preflight', 'fit'), required=True)
phase = parser.parse_args().phase
wait = json.loads((runner / (phase + '_wait.json')).read_bytes())
terminal = wait['terminal']
assert wait['observer_closed'] and not terminal['controller_alive']
assert terminal['exitcode'] == 0 and terminal['status']['status'] == 'complete'
spec = json.loads((runner / 'pair_spec.json').read_bytes())
destination = runner / ('preflight_complete' if phase == 'preflight' else 'complete_fit')
assert not destination.exists()
destination.mkdir()
client = paramiko.SSHClient()
client.load_system_host_keys()
client.connect('region-9.autodl.pro', port=33476, username='root',
               password=os.environ['MCLN_SSH_PASSWORD'], timeout=30)
code = '''import hashlib,json,subprocess,sys
from pathlib import Path
root=Path(sys.argv[1]);phase=sys.argv[2];assert root.resolve()==root
assert (root/(phase+'_controller.exit')).read_text().strip()=='0'
assert json.loads((root/(phase+'_status.json')).read_bytes())['status']=='complete'
files=[]
for path in sorted(root.rglob('*')):
 if not path.is_file() or path.suffix in ('.pt','.tmp'):continue
 name=str(path.relative_to(root));digest=hashlib.sha256()
 with path.open('rb') as stream:
  for block in iter(lambda:stream.read(8*1024*1024),b''):digest.update(block)
 files.append(dict(name=name,bytes=path.stat().st_size,sha256=digest.hexdigest()))
print(json.dumps(dict(files=files,weights_copied=sum(row['name'].endswith('/terminal.pth') for row in files))),flush=True)
subprocess.run(['tar','-cf','-','-C',str(root)]+[row['name'] for row in files],check=True,stdout=sys.stdout.buffer)
'''
_, stdout, stderr = client.exec_command(shlex.join([spec['runtime'] + '/venv/bin/python',
    '-B', '-c', code, spec['root'], phase]), timeout=120)
manifest = json.loads(stdout.readline())
expected = {entry['name']: entry for entry in manifest['files']}
assert manifest['weights_copied'] == (0 if phase == 'preflight' else 2)
assert {name for name in expected if name.endswith('.pth')} == (set() if phase == 'preflight' else
    {'whole_support/terminal.pth', 'extremal_support/terminal.pth'})
seen = []
with tarfile.open(fileobj=stdout, mode='r|') as archive:
    for member in archive:
        assert member.isfile() and member.name in expected
        entry = expected[member.name]
        assert member.size == entry['bytes']
        path = destination.joinpath(*PurePosixPath(member.name).parts)
        assert destination.resolve() in path.resolve().parents
        path.parent.mkdir(parents=True, exist_ok=True)
        digest, size = hashlib.sha256(), 0
        stream = archive.extractfile(member)
        with path.open('xb') as output:
            for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
                output.write(block)
                digest.update(block)
                size += len(block)
        assert size == entry['bytes'] and digest.hexdigest() == entry['sha256']
        seen.append(member.name)
assert len(seen) == len(expected) and set(seen) == set(expected)
error = stderr.read()
exit_code = stdout.channel.recv_exit_status()
(destination / 'COLLECT_STDERR.txt').write_bytes(error)
(destination / 'COLLECT_EXIT.json').write_text(json.dumps(dict(exit_code=exit_code)) + '\n')
assert exit_code == 0, error.decode()
client.close()
record = dict(status='CLOSED_SPAN_PHASE_ARTIFACTS_COLLECTED', phase=phase,
    time_cst=datetime.datetime.now().astimezone().isoformat(), files=manifest['files'],
    files_copied=len(seen), total_bytes=sum(row['bytes'] for row in expected.values()),
    weights_copied=manifest['weights_copied'], model_or_optimizer_replayed=False)
(destination / 'INTAKE.json').write_text(json.dumps(record, indent=2) + '\n')
print(json.dumps(dict(status=record['status'], phase=phase, files_copied=len(seen),
    bytes=record['total_bytes'], weights_copied=record['weights_copied'])), flush=True)
