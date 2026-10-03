"""Collect a completed pair's evidence; never run a forward or copy checkpoints."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import shlex

import paramiko


def main():
    local = Path(__file__).parent
    output = local/'complete'
    assert not output.exists(), 'do not overwrite an earlier intake'
    launch = json.loads((local/'pair_launch.json').read_bytes())
    spec = json.loads((local/'g_control_spec.json').read_bytes())
    code = r'''
import hashlib,json,shutil,sys
from pathlib import Path
root=Path(sys.argv[1])
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024**2),b''):h.update(block)
    return h.hexdigest()
status=json.loads((root/'pair_status.json').read_bytes())
expected=[('g_control','train'),('g_consistent','train'),('g_control','formal'),('g_consistent','formal')]
assert status['status']=='complete','the pair has not completed'
assert [(x['arm'],x['mode']) for x in status['completed']]==expected
names=['pair_status.json','preflight/preflight.json']
checkpoints={}
for arm in ('g_control','g_consistent'):
    directory=root/arm
    train=json.loads((directory/'receipt.json').read_bytes())
    formal=json.loads((directory/'formal/receipt.json').read_bytes())
    assert train['status']=='complete' and train['training_steps']==3723
    assert train['fit_seen_exactly_once'] and train['fit_rows']==29778
    assert formal['status']=='pass' and formal['rows']==9508
    for mode in ('train','formal'):
        assert (directory/(mode+'.exit')).read_text().strip()=='0'
    checkpoint=directory/'terminal.pth'
    digest=sha(checkpoint)
    assert digest==train['terminal_sha256']
    assert not (directory/'latest.pth').exists()
    assert not (directory/'latest.pth.tmp').exists()
    checkpoints[arm]={'path':str(checkpoint),'bytes':checkpoint.stat().st_size,'sha256':digest}
    names.extend(arm+'/'+name for name in ('spec.json','receipt.json','train.jsonl','train.log','train.exit','formal.log','formal.exit','load.json','imports.json'))
    for stage in ('initial','terminal','formal'):
        names.extend(arm+'/'+stage+'/'+name for name in ('receipt.json','rows.jsonl'))
sources=['run.py','pair.py','pvground_candidate_consistency.py','pvground_semantic_assignment.py','pvground_task_observation_query.py','pvground_observation_query.py','pvground_source_query.py']
names.extend(sources)
files={name:{'bytes':(root/name).stat().st_size,'sha256':sha(root/name)} for name in names}
print(json.dumps({'status':status,'files':files,'sources':sources,'checkpoints':checkpoints,'directory_free_bytes':shutil.disk_usage(str(root)).free}))
'''
    client = paramiko.SSHClient()
    client.load_system_host_keys()
    client.connect('region-9.autodl.pro', port=33476, username='root',
                   password=os.environ['MCLN_SSH_PASSWORD'], timeout=30)
    command = shlex.join([spec['runtime']+'/venv/bin/python', '-c', code, launch['root']])
    _, stdout, stderr = client.exec_command(command, timeout=180)
    raw = stdout.read()
    assert stdout.channel.recv_exit_status() == 0, stderr.read().decode()
    witness = json.loads(raw)
    output.mkdir()
    sftp = client.open_sftp()
    downloads = {}
    for name, expected in witness['files'].items():
        relative = 'source/'+name if name in witness['sources'] else name
        destination = output/relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        sftp.get(launch['root']+'/'+name, str(destination))
        content = destination.read_bytes()
        assert len(content) == expected['bytes'], name
        assert hashlib.sha256(content).hexdigest() == expected['sha256'], name
        downloads[relative] = expected
    sftp.close()
    client.close()
    witness.update(time_cst=datetime.datetime.now().astimezone().isoformat(),
                   remote_root=launch['root'], downloads=downloads,
                   gpu_forward_executed=False, optimizer_updates_executed=0,
                   checkpoints_downloaded=False, weights_deleted=False)
    (output/'INTAKE.json').write_text(json.dumps(witness, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({'status':'complete_evidence_collected', 'files':len(downloads),
                      'output':str(output), 'checkpoint_copies':0, 'weights_deleted':0}))


if __name__ == '__main__':
    main()
