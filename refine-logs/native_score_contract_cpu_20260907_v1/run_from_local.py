import hashlib
import json
import os
from pathlib import Path
import shlex
import paramiko

repo = Path('C:/Users/gb/.codex_mcln_g0_20260905')
archive = repo / 'refine-logs/native_score_contract_cpu_20260907_v1'
archive.mkdir()
source = '/root/autodl-tmp/mcln_native_mask_geometry_source_preparation_20260907_v1/model_source'
remote = '/root/autodl-tmp/mcln_native_score_contract_cpu_20260907_v1'
names = ['models/rec_candidate_adapter.py', 'models/source_choice_adapter.py',
         'src/grounding_evaluator.py', 'models/losses.py', 'train_dist_mod.py', 'src/joint_det_dataset.py']
hashes = {n: hashlib.sha256((repo / n).read_bytes()).hexdigest() for n in names}
(archive / 'source_hashes.json').write_text(json.dumps(hashes, indent=2) + '\n', encoding='utf-8')
c = paramiko.SSHClient()
c.load_system_host_keys()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect('region-9.autodl.pro', port=33476, username='root', password=os.environ['MCLN_SSH_PASSWORD'], timeout=30)
s = c.open_sftp()
s.mkdir(remote)
s.put(str(archive / 'source_hashes.json'), remote + '/source_hashes.json')
s.put(str(repo / 'scripts/audit_native_rec_score_semantics.py'), remote + '/audit_native_rec_score_semantics.py')
argv = ['/root/miniconda3/envs/bdetr/bin/python', remote + '/audit_native_rec_score_semantics.py',
        '--source-root', source, '--expected-hashes', remote + '/source_hashes.json', '--output', remote + '/receipt.json']
command = 'CUDA_VISIBLE_DEVICES= ' + ' '.join(shlex.quote(v) for v in argv) + ' > ' + shlex.quote(remote + '/audit.log') + ' 2>&1'
_, out, err = c.exec_command(command, timeout=45)
out.read()
code = out.channel.recv_exit_status()
with s.open(remote + '/audit.log', 'rb') as stream:
    log = stream.read()
(archive / 'audit.log').write_bytes(log)
(archive / 'exit.json').write_text(json.dumps({'exit_code': code, 'remote_directory': remote,
                                               'source_directory': source, 'command': command}, indent=2) + '\n', encoding='utf-8')
assert code == 0, log.decode()
with s.open(remote + '/receipt.json', 'rb') as stream:
    raw = stream.read()
(archive / 'receipt.json').write_bytes(raw)
(archive / 'run_from_local.py').write_bytes(Path(__file__).read_bytes())
result = json.loads(raw)
assert result['status'] == 'pass' and not result['metric_gain_proven']
print(json.dumps({'status': result['status'], 'cases': result['synthetic_cases'],
                  'receipt_sha256': hashlib.sha256(raw).hexdigest()}), flush=True)
s.close()
c.close()
