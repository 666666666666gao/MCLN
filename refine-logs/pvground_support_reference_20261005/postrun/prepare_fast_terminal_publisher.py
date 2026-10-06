"""Replace repeated closed-text readback with the existing SHA256 evidence."""
import ast
import hashlib
import json
from pathlib import Path

root=Path(__file__).resolve().parents[1]
old=root/'postrun/publish_reference_terminal.py'
new=root/'postrun/publish_reference_terminal_fast.py'
assert not new.exists()
source=old.read_text(encoding='utf-8')


def change(text,before,after):
    assert text.count(before)==1,before
    return text.replace(before,after)


source=change(source,'import subprocess\nimport paramiko','import subprocess\nimport shlex\nimport paramiko')
source=change(source,"    'postrun/publish_reference_terminal.py','TERMINAL_PUBLICATION_PREPARATION.json']",
    "    'postrun/publish_reference_terminal.py','TERMINAL_PUBLICATION_PREPARATION.json',\n"
    "    'postrun/publish_reference_terminal_fast.py','postrun/prepare_fast_terminal_publisher.py',\n"
    "    'PUBLICATION_TRANSFER_OPTIMIZATION.json']")
source=change(source,'for name, raw in payloads.items():\n    for repo in repos[:2]:',
'''existing={name:hashlib.sha256(raw).hexdigest() for name,raw in payloads.items()
    if name.startswith(prefix+'complete/') and name!=prefix+'complete/INTAKE.json'}
probe="""
import hashlib,json,sys
from pathlib import Path
project=Path(sys.argv[1]);expected=json.loads(sys.argv[2])
for name,digest in expected.items():
    assert hashlib.sha256((project/name).read_bytes()).hexdigest()==digest,name
print(json.dumps(dict(verified_existing_closed_texts=len(expected))))
"""
runtime=json.loads((local/'control_spec.json').read_bytes())['runtime']
_,stdout,stderr=client.exec_command(shlex.join([runtime+'/venv/bin/python','-B','-c',probe,project,json.dumps(existing)]),timeout=120)
verified=json.loads(stdout.read())
assert stdout.channel.recv_exit_status()==0,stderr.read().decode()
assert verified['verified_existing_closed_texts']==len(existing)
for name, raw in payloads.items():
    for repo in repos[:2]:''')
source=change(source,"    with sftp.open(remote, 'rb') as stream:\n        assert stream.read() == raw",
    "    if name not in existing:\n        with sftp.open(remote, 'rb') as stream:\n            assert stream.read() == raw")
source=change(source,'remote_complete_reuses_closed_data_disk_originals=True)',
    "remote_complete_reuses_closed_data_disk_originals=True,\n              existing_closed_payloads_verified_by_remote_SHA256=len(existing))")
ast.parse(source)
new.write_text(source,encoding='utf-8')
record=dict(status='PUBLICATION_TRANSFER_OPTIMIZATION_PREPARED_NOT_EXECUTED',
    original_source=str(old),original_source_sha256=hashlib.sha256(old.read_bytes()).hexdigest(),
    optimized_source=str(new),optimized_source_sha256=hashlib.sha256(new.read_bytes()).hexdigest(),
    change='Existing complete payloads use one remote CPU SHA256 comparison; document byte readback and all other publication checks unchanged.',
    original_source_unchanged=True,neural_or_weights_changed=False,published=False)
(root/'PUBLICATION_TRANSFER_OPTIMIZATION.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
print(json.dumps(record))
