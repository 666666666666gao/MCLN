"""Stream the closed actual artifacts directly; never copy nonbest checkpoints."""
import datetime
import hashlib
import json
import os
from pathlib import Path,PurePosixPath
import shlex
import tarfile
import paramiko

local=Path(__file__).resolve().parent
wait=json.loads((local/'fit_wait.json').read_bytes())
terminal=wait['terminal']
assert wait['observer_closed'] and not terminal['controller_alive']
assert terminal['exitcode']==0 and terminal['status']['status']=='complete'
spec=json.loads((local/'control_spec.json').read_bytes())
root=str(PurePosixPath(spec['root']).parent)
destination=local/'complete_fit'
assert not destination.exists()
destination.mkdir()
client=paramiko.SSHClient();client.load_system_host_keys()
client.connect('region-9.autodl.pro',port=33476,username='root',password=os.environ['MCLN_SSH_PASSWORD'],timeout=30)
code='''import hashlib,json,subprocess,sys
from pathlib import Path
root=Path(sys.argv[1]);assert root.resolve()==root
assert (root/'fit_controller.exit').read_text().strip()=='0'
assert json.loads((root/'fit_status.json').read_bytes())['status']=='complete'
files=[]
for path in sorted(root.rglob('*')):
    if not path.is_file() or path.suffix in ('.pth','.pt','.tmp'):continue
    name=str(path.relative_to(root));digest=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):digest.update(block)
    files.append(dict(name=name,bytes=path.stat().st_size,sha256=digest.hexdigest()))
print(json.dumps(dict(files=files,weights_copied=0)),flush=True)
subprocess.run(['tar','-cf','-','-C',str(root)]+[row['name'] for row in files],check=True,stdout=sys.stdout.buffer)
'''
_,stdout,stderr=client.exec_command(shlex.join([spec['runtime']+'/venv/bin/python','-B','-c',code,root]),timeout=120)
manifest=json.loads(stdout.readline())
expected={entry['name']:entry for entry in manifest['files']}
seen=[]
with tarfile.open(fileobj=stdout,mode='r|') as archive:
    for member in archive:
        assert member.isfile() and member.name in expected
        entry=expected[member.name]
        assert member.size==entry['bytes']
        path=destination.joinpath(*PurePosixPath(member.name).parts)
        assert destination.resolve() in path.resolve().parents
        path.parent.mkdir(parents=True,exist_ok=True)
        digest=hashlib.sha256();size=0
        stream=archive.extractfile(member)
        with path.open('xb') as output:
            for block in iter(lambda:stream.read(4*1024*1024),b''):
                output.write(block);digest.update(block);size+=len(block)
        assert size==entry['bytes'] and digest.hexdigest()==entry['sha256']
        seen.append(member.name)
assert len(seen)==len(expected) and set(seen)==set(expected)
assert stdout.channel.recv_exit_status()==0,stderr.read().decode()
client.close()
record=dict(time_cst=datetime.datetime.now().astimezone().isoformat(),status='CLOSED_FIT_ARTIFACTS_COLLECTED',
    files=manifest['files'],files_copied=len(seen),total_bytes=sum(row['bytes'] for row in expected.values()),
    weights_copied=0,model_or_optimizer_replayed=False)
(destination/'INTAKE.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
print(json.dumps(dict(status=record['status'],files_copied=len(seen),bytes=record['total_bytes'],weights_copied=0)),flush=True)
