"""Read-only inspection after the actual Doc80 remote receiver failure."""
import hashlib
import json
import os
from pathlib import Path
import shlex
import paramiko

local=Path(__file__).resolve().parent
client=paramiko.SSHClient();client.load_system_host_keys()
client.connect('region-9.autodl.pro',port=33476,username='root',password=os.environ['MCLN_SSH_PASSWORD'],timeout=30)
code='''import hashlib,json
from pathlib import Path
project=Path('/home/gb/new butd/butd_detr-main/MCLN-main')
doc=project/'docs/MCLN_CURRENT_COMPLETE_HANDOFF_2026-08-15.md'
root=project/'refine-logs/pvground_mask_extent_diagnostic_20261006'
print(json.dumps(dict(doc_sha256=hashlib.sha256(doc.read_bytes()).hexdigest(),
                     evidence_exists=root.exists(),evidence_paths=[str(p.relative_to(root)) for p in root.rglob('*') if p.is_file()])))
'''
runtime=json.loads((local/'spec.json').read_bytes())['runtime']
_,stdout,stderr=client.exec_command(shlex.join([runtime+'/venv/bin/python','-B','-c',code]),timeout=120)
raw=stdout.read();err=stderr.read().decode();exitcode=stdout.channel.recv_exit_status()
record=dict(exitcode=exitcode,stdout=raw.decode(),stderr=err,publication_native_session=87407,publication_native_exit=1)
(local/'PUBLICATION_FAILURE_INSPECTION.json').write_text(json.dumps(record,indent=2)+'\n')
client.close();print(json.dumps(record))
