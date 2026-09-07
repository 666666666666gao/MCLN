import datetime
import hashlib
import json
import os
from pathlib import Path
import shlex
import paramiko

temporary = Path('C:/Users/gb/.codex/tmp')
proof = json.loads((temporary / 'pvground_runtime_bundle_receipt_20260908.json').read_bytes())
root = proof['remote_root']
remote_bundle = '/root/autodl-tmp/pvground_runtime_bundle_20260908_v1.tar.gz'
archive = Path('C:/Users/gb/.codex_mcln_g0_20260905/refine-logs/pvground_runtime_20260908_v1')
assert list(p.name for p in archive.iterdir()) == ['launch_failed.py']
(archive / 'bootstrap_failure.json').write_bytes((json.dumps({'observed_time_cst': datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(), 'phase':'after bundle transfer, before extraction or build', 'error':'SyntaxError caused by BUNDLE replacement inside RUNTIME_BUNDLE_VERIFIED string', 'remote_build_started':False, 'recovery':'reuse verified transferred bundle; fix literal placeholder substitution'}, indent=2)+'\n').encode())
c = paramiko.SSHClient()
c.load_system_host_keys()
c.connect('region-9.autodl.pro', port=33476, username='root', password=os.environ['MCLN_SSH_PASSWORD'], timeout=30)
s = c.open_sftp()
source = (temporary / 'launch_pvg_runtime_20260908.py').read_text(encoding='utf-8')
tail = 'bootstrap = ' + source.split('bootstrap = ', 1)[1]
exec(compile(tail, 'resume_verified_runtime_bundle_bootstrap', 'exec'))
