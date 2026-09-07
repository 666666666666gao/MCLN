import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import paramiko

repo = Path('C:/Users/gb/.codex_mcln_g0_20260905')
archive = repo / 'refine-logs/native_score_contract_cpu_20260907_v1'
remote = '/root/autodl-tmp/mcln_native_score_contract_cpu_20260907_v1'
source = '/root/autodl-tmp/mcln_native_mask_geometry_source_preparation_20260907_v1/model_source'
old = json.loads((archive / 'source_hashes.json').read_bytes())
c = paramiko.SSHClient()
c.load_system_host_keys()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect('region-9.autodl.pro', port=33476, username='root', password=os.environ['MCLN_SSH_PASSWORD'], timeout=30)
s = c.open_sftp()
hashes = {}
differences = {}
for name in old:
    with s.open(source + '/' + name, 'rb') as stream:
        raw = stream.read()
    local = (repo / name).read_bytes()
    git = subprocess.check_output(['git', 'show', 'HEAD:' + name], cwd=str(repo))
    assert raw.replace(b'\r\n', b'\n') == local.replace(b'\r\n', b'\n') == git.replace(b'\r\n', b'\n'), name
    hashes[name] = hashlib.sha256(raw).hexdigest()
    differences[name] = {'local_sha256': old[name], 'remote_sha256': hashes[name],
                         'git_sha256': hashlib.sha256(git).hexdigest(),
                         'local_CRLF': local.count(b'\r\n'), 'remote_CRLF': raw.count(b'\r\n'),
                         'normalized_source_exact': True}
(archive / 'source_binding_line_endings.json').write_text(json.dumps(differences, indent=2) + '\n', encoding='utf-8')
(archive / 'source_hashes_runtime.json').write_text(json.dumps(hashes, indent=2) + '\n', encoding='utf-8')
s.put(str(archive / 'source_hashes_runtime.json'), remote + '/source_hashes_runtime.json')
argv = ['/root/miniconda3/envs/bdetr/bin/python', remote + '/audit_native_rec_score_semantics.py',
        '--source-root', source, '--expected-hashes', remote + '/source_hashes_runtime.json', '--output', remote + '/receipt.json']
command = 'CUDA_VISIBLE_DEVICES= ' + ' '.join(shlex.quote(v) for v in argv) + ' > ' + shlex.quote(remote + '/audit_2.log') + ' 2>&1'
_, out, err = c.exec_command(command, timeout=45)
out.read()
code = out.channel.recv_exit_status()
with s.open(remote + '/audit_2.log', 'rb') as stream:
    log = stream.read()
(archive / 'audit_2.log').write_bytes(log)
(archive / 'exit_2.json').write_text(json.dumps({'exit_code': code, 'command': command}, indent=2) + '\n', encoding='utf-8')
assert code == 0, log.decode()
with s.open(remote + '/receipt.json', 'rb') as stream:
    raw = stream.read()
(archive / 'receipt.json').write_bytes(raw)
for name, filename in [('run_from_local.py', 'run_native_score_contract_cpu_20260907.py'),
                       ('finish_from_local.py', Path(__file__).name)]:
    (archive / name).write_bytes((Path(__file__).parent / filename).read_bytes())
result = json.loads(raw)
print(json.dumps({'status': result['status'], 'cases': result['synthetic_cases'],
                  'receipt_sha256': hashlib.sha256(raw).hexdigest(), 'source_binding': differences}), flush=True)
s.close()
c.close()
