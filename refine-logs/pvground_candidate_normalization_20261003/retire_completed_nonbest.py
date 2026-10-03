"""Retire this one formally evaluated nonleading endpoint, without an archive."""
import datetime
import json
import os
from pathlib import Path
import shlex

import paramiko


local = Path(__file__).parent
assert not (local / 'nonbest_retirement_receipt.json').exists()
intake = json.loads((local / 'complete/INTAKE.json').read_bytes())
analysis = json.loads((local / 'analysis/SUMMARY.json').read_bytes())
audit = json.loads((local / 'analysis/EXPERIMENT_AUDIT.json').read_bytes())
assert audit['verdict'].upper() in ('PASS', 'WARN') and not audit['blocking_issues']
assert analysis['retention_recommendation']['score_leader'] == 'original_g'
assert not analysis['retention_recommendation']['normalized_endpoint_keep']
assert intake['status']['status'] == 'complete' and intake['controller_exit'] == 0
assert not intake['controller_alive'] and analysis['finished_cst'] == intake['status']['finished_cst']
spec = json.loads((local / 'normalized_spec.json').read_bytes())
launch = json.loads((local / 'launch.json').read_bytes())
root = '/root/autodl-tmp/pvground_candidate_normalization_20261003'
assert intake['remote_root'] == launch['root'] == root
item = intake['checkpoint']
assert item['path'] == root + '/normalized/terminal.pth'
probe = r'''
import datetime,hashlib,json,shutil,sys
from pathlib import Path
root=Path(sys.argv[1]);pid=int(sys.argv[2]);item=json.loads(sys.argv[3]);parent=json.loads(sys.argv[4])
assert not Path('/proc/%d'%pid).exists()
assert json.loads((root/'status.json').read_bytes())['status']=='complete'
assert (root/'controller.exit').read_text().strip()=='0'
receipt=root/'nonbest_retirement_receipt.json'
assert not receipt.exists()
def sha(path):
    result=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(4*1024**2),b''):result.update(block)
    return result.hexdigest()
path=Path(item['path'])
assert str(path.resolve())==item['path'] and path==root/'normalized/terminal.pth'
assert path.stat().st_size==item['bytes'] and sha(path)==item['sha256']
train=json.loads((root/'normalized/receipt.json').read_bytes())
formal=json.loads((root/'normalized/formal/receipt.json').read_bytes())
assert train['status']=='complete' and train['training_steps']==3723 and train['fit_seen_exactly_once']
assert train['terminal_sha256']==item['sha256']
assert formal['status']=='pass' and formal['rows']==9508
assert sha(parent['path'])==parent['sha256']
before=shutil.disk_usage(root).free
path.unlink()
assert not path.exists() and sha(parent['path'])==parent['sha256']
result=dict(time_cst=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
    deleted_count=1,deleted_bytes=item['bytes'],file=item,directory_free_before=before,
    directory_free_after=shutil.disk_usage(root).free,weights_deleted=True,
    local_archive_created=False,original_g_sha256_verified_before_and_after=True,
    user_authorization='及时清理我们产生的无用权重，只保留指标最好权重',
    protected_parent_and_V99_not_deletion_targets=True)
receipt.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(result,ensure_ascii=False))
'''
client = paramiko.SSHClient()
client.load_system_host_keys()
client.connect('region-9.autodl.pro', port=33476, username='root',
               password=os.environ['MCLN_SSH_PASSWORD'], timeout=30)
command = shlex.join([spec['runtime'] + '/venv/bin/python', '-c', probe, root,
    launch['process'].split()[0], json.dumps(item),
    json.dumps(dict(path=spec['base_terminal'], sha256=spec['base_terminal_sha256']))])
_, stdout, stderr = client.exec_command(command, timeout=180)
raw = stdout.read()
assert stdout.channel.recv_exit_status() == 0, stderr.read().decode()
result = json.loads(raw)
sftp = client.open_sftp()
with sftp.open(root + '/nonbest_retirement_receipt.json', 'rb') as stream:
    receipt = stream.read()
assert json.loads(receipt) == result
(local / 'nonbest_retirement_receipt.json').write_bytes(receipt)
sftp.close()
client.close()
print(json.dumps(dict(deleted_count=result['deleted_count'],deleted_bytes=result['deleted_bytes'],
    original_g_retained=True,local_archive_created=False)))
