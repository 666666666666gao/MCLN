"""Retire only the two archived, formally evaluated nonleading pair endpoints."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import shlex
import paramiko

local = Path(__file__).parent
assert not (local / 'pair_endpoint_cleanup_receipt.json').exists()
archive = json.loads((local / 'pair_endpoint_archive_receipt.json').read_bytes())
analysis = json.loads((local / 'analysis/SUMMARY.json').read_bytes())
audit = json.loads((local / 'TERMINAL_EXPERIMENT_AUDIT.json').read_bytes())
assert archive['archive_complete'] and not archive['weights_deleted']
assert audit['verdict'].upper() in ('PASS', 'WARN') and not audit['blocking_issues']
assert analysis['retention_recommendation']['score_leader'] == 'original_g'
spec = json.loads((local / 'g_control_spec.json').read_bytes())
root = '/root/autodl-tmp/pvground_candidate_consistency_20261003'
expected = {root + '/g_control/terminal.pth', root + '/g_consistent/terminal.pth'}
assert {item['path'] for item in archive['files']} == expected
for item in archive['files']:
    path = Path(item['local'])
    assert path.stat().st_size == item['bytes']
    with path.open('rb') as stream:
        assert hashlib.file_digest(stream, 'sha256').hexdigest() == item['sha256']
code = r'''
import hashlib,json,shutil,sys
from pathlib import Path
root=Path(sys.argv[1]);items=json.loads(sys.argv[2]);parent=json.loads(sys.argv[3])
assert json.loads((root/'pair_status.json').read_bytes())['status']=='complete'
def sha(path):
    result=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(4*1024*1024),b''):result.update(block)
    return result.hexdigest()
assert sha(parent['path'])==parent['sha256']
before=shutil.disk_usage(str(root)).free
for item in items:
    p=Path(item['path'])
    assert str(p.resolve())==item['path'] and p.parent.parent==root
    assert p.name=='terminal.pth' and p.stat().st_size==item['bytes']
    assert sha(p)==item['sha256']
for item in items:Path(item['path']).unlink()
assert all(not Path(item['path']).exists() for item in items)
assert sha(parent['path'])==parent['sha256']
print(json.dumps(dict(deleted_count=len(items),deleted_bytes=sum(i['bytes'] for i in items),
    directory_free_before=before,directory_free_after=shutil.disk_usage(str(root)).free,
    original_g_sha256_verified_before_and_after=True,weights_deleted=True)))
'''
client = paramiko.SSHClient()
client.load_system_host_keys()
client.connect('region-9.autodl.pro', port=33476, username='root',
               password=os.environ['MCLN_SSH_PASSWORD'], timeout=30)
command = shlex.join([spec['runtime'] + '/venv/bin/python', '-c', code, root,
                      json.dumps(archive['files']), json.dumps({'path': spec['base_terminal'],
                                                              'sha256': spec['base_terminal_sha256']})])
_, stdout, stderr = client.exec_command(command, timeout=180)
raw = stdout.read()
assert stdout.channel.recv_exit_status() == 0, stderr.read().decode()
receipt = json.loads(raw)
receipt.update(time_cst=datetime.datetime.now().astimezone().isoformat(), files=archive['files'],
               user_authorization='及时清理磁盘空间，只保留权重最高的权重',
               local_archives_complete_and_rehashed=True, original_g_remains_best=True,
               V99_chain_and_required_parent_paths_not_deletion_targets=True)
raw = (json.dumps(receipt, ensure_ascii=False, indent=2) + '\n').encode('utf-8')
(local / 'pair_endpoint_cleanup_receipt.json').write_bytes(raw)
with client.open_sftp().open(root + '/pair_endpoint_cleanup_receipt.json', 'wx') as stream:
    stream.write(raw)
client.close()
print(json.dumps({key: value for key, value in receipt.items() if key != 'files'}, ensure_ascii=False))
