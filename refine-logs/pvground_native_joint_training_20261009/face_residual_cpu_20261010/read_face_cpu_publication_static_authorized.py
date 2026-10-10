"""Verify only publication artifacts after an uncertain transport closure."""
import base64
import datetime
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess

root = Path(__file__).resolve().parent
prior = json.loads((root/'referit_author_core_cpu_publication.json').read_bytes())
project = Path('C:/Users/gb/.codex_mcln_g0_20260905')
doc = 'docs/MCLN_CURRENT_COMPLETE_HANDOFF_2026-08-15.md'
old = (project/doc).read_bytes()
assert hashlib.sha256(old).hexdigest() == prior['doc_sha256']
note = (root/'HANDOFF_FACE_RESIDUAL_CPU_20261010.md').read_text(encoding='utf-8')
new = old+('\n\n'+note.replace('\r\n','\n')).replace('\n','\r\n').encode()
prefix = 'refine-logs/pvground_native_joint_training_20261009/face_residual_cpu_20261010/'
names = ['HANDOFF_FACE_RESIDUAL_CPU_20261010.md','ACTUAL_FACE_RESIDUAL_CPU_AUDIT_CLOSURE.json',
    'record_actual_face_residual_cpu_closure.py','prepare_face_cpu_transport_fix.py',
    'read_face_cpu_static_root_authorized.py','prepare_face_residual_cpu_publication.py',
    'publish_face_residual_cpu_authorized.py']
names.extend(p.relative_to(root).as_posix() for p in (root/'face_residual_preparation_20261010').rglob('*')
    if p.is_file() and p.name != 'RAW_STDERR.txt')
hashes = {prefix+n:hashlib.sha256((root/n).read_bytes()).hexdigest() for n in sorted(set(names))}
hashes[prefix+'.gitattributes'] = hashlib.sha256(b'** -text whitespace=-blank-at-eol,-space-before-tab,cr-at-eol,-blank-at-eof\n').hexdigest()
payload = dict(doc=doc,old_sha256=prior['doc_sha256'],new_sha256=hashlib.sha256(new).hexdigest(),
    prefix=prefix,file_sha256=hashes)
(root/'FACE_CPU_PUBLICATION_EXPECTED_AT_FAILURE.json').write_text(json.dumps(payload,indent=2)+'\n')
code = r'''import hashlib,json,sys
from pathlib import Path
b=json.load(sys.stdin);p=Path('/home/gb/new butd/butd_detr-main/MCLN-main');d=p/b['doc'];leaf=p/b['prefix']
actual=hashlib.sha256(d.read_bytes()).hexdigest();present=0;missing=[];mismatch=[]
for n,h in b['file_sha256'].items():
 q=p/n
 if not q.is_file():missing.append(n)
 elif hashlib.sha256(q.read_bytes()).hexdigest()!=h:mismatch.append(n)
 else:present+=1
print(json.dumps(dict(doc_sha256=actual,old_document=actual==b['old_sha256'],new_document=actual==b['new_sha256'],
 prefix_exists=leaf.exists(),verified_files=present,expected_files=len(b['file_sha256']),missing=missing,mismatched=mismatch,
 training_status_queries=0)))
'''
witness = json.loads((root.parent/'SCP_TRANSPORT_WITNESS.json').read_bytes())
environment = dict(os.environ,SSH_ASKPASS='C:/Users/gb/.codex/tmp/pvground_native_joint_training_20261009/NativeSshAskPass.exe',
    SSH_ASKPASS_REQUIRE='force',DISPLAY='codex-byte-transfer')
argv = ['C:/Windows/System32/OpenSSH/ssh.exe','-T','-p','33476','-o','ProxyCommand=none',
    '-o','StrictHostKeyChecking=yes','-o','UserKnownHostsFile=C:/Users/gb/.codex/tmp/pvground_native_joint_training_20261009/native_scp_known_hosts',
    '-o','HostKeyAlgorithms='+witness['negotiated_host_key_algorithm'],'-o','NumberOfPasswordPrompts=1',
    'root@region-9.autodl.pro',shlex.join(['/root/autodl-tmp/mcln_pvground_runtime_20260908_v1/venv/bin/python','-B','-u','-c',code])]
response = subprocess.run(argv,env=environment,input=json.dumps(payload).encode(),stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,creationflags=subprocess.CREATE_NO_WINDOW)
(root/'FACE_CPU_PUBLICATION_STATIC_PRIVATE_STDERR.txt').write_bytes(response.stderr)
(root/'FACE_CPU_PUBLICATION_STATIC_RAW_STDOUT.json').write_bytes(response.stdout)
(root/'FACE_CPU_PUBLICATION_STATIC_EXIT.json').write_text(json.dumps(dict(exit_code=response.returncode))+'\n')
assert response.returncode == 0
record = json.loads(response.stdout)
record.update(time_cst=datetime.datetime.now().astimezone().isoformat())
(root/'FACE_CPU_PUBLICATION_STATIC_READ.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps({k:v for k,v in record.items() if k not in ('missing','mismatched')}))
