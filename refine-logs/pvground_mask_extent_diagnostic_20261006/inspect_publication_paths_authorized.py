"""Read the actual relocated evidence root after the path guard rejection."""
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
logs=project/'refine-logs';root=logs/'pvground_mask_extent_diagnostic_20261006'
doc=project/'docs/MCLN_CURRENT_COMPLETE_HANDOFF_2026-08-15.md'
print(json.dumps(dict(project_resolved=str(project.resolve()),logs_is_symlink=logs.is_symlink(),
                     logs_resolved=str(logs.resolve()),evidence_resolved=str(root.resolve()),
                     evidence_exists=root.exists(),doc_sha256=hashlib.sha256(doc.read_bytes()).hexdigest())))
'''
runtime=json.loads((local/'spec.json').read_bytes())['runtime']
_,stdout,stderr=client.exec_command(shlex.join([runtime+'/venv/bin/python','-B','-c',code]),timeout=120)
raw=stdout.read();assert stdout.channel.recv_exit_status()==0,stderr.read().decode()
record=json.loads(raw);record.update(publication_native_session=21394,publication_native_exit=1,
                                   rejection='Remote receiver line9 resolved path containment check before any evidence write')
(local/'PUBLICATION_PATH_WITNESS.json').write_text(json.dumps(record,indent=2)+'\n')
client.close();print(json.dumps(record))
