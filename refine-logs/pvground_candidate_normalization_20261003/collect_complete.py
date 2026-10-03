"""Collect terminal evidence only after the original one-arm controller exits."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import shlex

import paramiko


def main():
    local = Path(__file__).parent
    output = local / 'complete'
    assert not output.exists(), 'preserve any earlier intake'
    launch = json.loads((local / 'launch.json').read_bytes())
    spec = json.loads((local / 'normalized_spec.json').read_bytes())
    probe = r'''
import hashlib,json,shutil,subprocess,sys
from pathlib import Path
root=Path(sys.argv[1]); controller=int(sys.argv[2])
def sha(path):
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024**2),b''): digest.update(block)
    return digest.hexdigest()
status=json.loads((root/'status.json').read_bytes())
assert status['status']=='complete', 'original controller has not completed'
assert [row['mode'] for row in status['completed']]==['train','formal']
assert (root/'controller.exit').read_text().strip()=='0'
assert not Path('/proc/%d'%controller).exists(), 'original controller is still live'
directory=root/'normalized'
train=json.loads((directory/'receipt.json').read_bytes())
formal=json.loads((directory/'formal/receipt.json').read_bytes())
assert train['status']=='complete' and train['training_steps']==3723
assert train['fit_seen_exactly_once'] and train['fit_rows']==29778
assert formal['status']=='pass' and formal['rows']==9508
for mode in ('train','formal'):
    assert (directory/(mode+'.exit')).read_text().strip()=='0'
checkpoint=directory/'terminal.pth'; digest=sha(checkpoint)
assert digest==train['terminal_sha256']
assert not any((directory/name).exists() for name in ('latest.pth','latest.pth.tmp','terminal.pth.tmp'))
names=['status.json','controller.log','controller.exit','launch.json','preflight/preflight.json']
names.extend('normalized/'+name for name in ('spec.json','receipt.json','train.jsonl','train.log','train.exit','formal.log','formal.exit','load.json','imports.json'))
for stage in ('initial','terminal','formal'):
    names.extend('normalized/'+stage+'/'+name for name in ('receipt.json','rows.jsonl'))
sources=['run.py','controller.py','pvground_candidate_consistency.py','pvground_semantic_assignment.py','pvground_task_observation_query.py','pvground_observation_query.py','pvground_source_query.py']
names.extend(sources)
files={('source/'+name if name in sources else name):{'path':str(root/name),'bytes':(root/name).stat().st_size,'sha256':sha(root/name)} for name in names}
imports=json.loads((directory/'imports.json').read_bytes())
for name,path in imports['files'].items():
    source=Path(path); observed=sha(source)
    assert observed==imports['sha256'][name], ('import changed',name)
    if name not in ('pvground_candidate_consistency','pvground_semantic_assignment','pvground_task_observation_query','pvground_observation_query','pvground_source_query'):
        files['source/imported/'+name+'.py']={'path':str(source),'bytes':source.stat().st_size,'sha256':observed}
parent=Path(json.loads((directory/'spec.json').read_bytes())['base_terminal'])
assert sha(parent)==train['base_terminal_sha256']
gpu=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader'],text=True).strip()
print(json.dumps(dict(status=status,controller_exit=0,controller_alive=False,files=files,
    checkpoint={'path':str(checkpoint),'bytes':checkpoint.stat().st_size,'sha256':digest},
    original_g_sha256=sha(parent),directory_free_bytes=shutil.disk_usage(root).free,gpu_compute=gpu)))
'''
    client = paramiko.SSHClient()
    client.load_system_host_keys()
    client.connect('region-9.autodl.pro', port=33476, username='root',
                   password=os.environ['MCLN_SSH_PASSWORD'], timeout=30)
    command = shlex.join([spec['runtime'] + '/venv/bin/python', '-c', probe,
                          launch['root'], launch['process'].split()[0]])
    _, stdout, stderr = client.exec_command(command, timeout=180)
    raw = stdout.read()
    assert stdout.channel.recv_exit_status() == 0, stderr.read().decode()
    witness = json.loads(raw)
    output.mkdir()
    sftp = client.open_sftp()
    for relative, expected in witness['files'].items():
        destination = output / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        sftp.get(expected['path'], str(destination))
        content = destination.read_bytes()
        assert len(content) == expected['bytes'], relative
        assert hashlib.sha256(content).hexdigest() == expected['sha256'], relative
    sftp.close()
    client.close()
    witness.update(time_cst=datetime.datetime.now().astimezone().isoformat(),
                   remote_root=launch['root'], gpu_forward_executed=False,
                   optimizer_updates_executed=0, checkpoints_downloaded=False,
                   weights_deleted=False)
    (output / 'INTAKE.json').write_text(json.dumps(witness, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(dict(status='complete_evidence_collected', files=len(witness['files']),
                          output=str(output), checkpoint_copies=0, weights_deleted=0)))


if __name__ == '__main__':
    main()
