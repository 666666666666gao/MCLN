"""Preserve failed attempt evidence and apply the proven Windows SSH paths."""
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parent
f = root/'face_residual_preparation_20261010'
launcher = f/'run_face_residual_cpu_authorized.py'
old = launcher.read_bytes()
review = json.loads((f/'source_review/EXPERIMENT_CODE_REVIEW.json').read_bytes())
old_hash = hashlib.sha256(old).hexdigest()
assert any(Path(name).name == launcher.name and digest == old_hash
           for name,digest in review['audited_input_hashes'].items())
archive = f/'transport_attempt1'
assert not archive.exists()
archive.mkdir()
(archive/'REVIEWED_LAUNCHER_ATTEMPT1.py').write_bytes(old)
for extension in ('json','md'):
    (archive/('EXPERIMENT_CODE_REVIEW_R1.'+extension)).write_bytes(
        (f/'source_review'/('EXPERIMENT_CODE_REVIEW.'+extension)).read_bytes())
failed = f/'cpu_execution'
exit_code = json.loads((failed/'TRANSPORT_EXIT.json').read_bytes())['exit_code']
stderr = (failed/'RAW_STDERR.txt').read_bytes()
assert exit_code == 255 and (failed/'RAW_STDOUT.json').stat().st_size == 0
assert b'Host key verification failed' in stderr
for extension in ('RAW_STDERR.txt','RAW_STDOUT.json','TRANSPORT_EXIT.json'):
    (archive/extension).write_bytes((failed/extension).read_bytes())
replacements = {
    "attempt = root/'cpu_execution'": "attempt = root/'cpu_execution_attempt2'",
    "SSH_ASKPASS=str(v.parent/'NativeSshAskPass.exe')":
        "SSH_ASKPASS='C:/Users/gb/.codex/tmp/pvground_native_joint_training_20261009/NativeSshAskPass.exe'",
    "'UserKnownHostsFile='+str(v.parent/'native_scp_known_hosts')":
        "'UserKnownHostsFile=C:/Users/gb/.codex/tmp/pvground_native_joint_training_20261009/native_scp_known_hosts'",
}
text = old.decode('utf-8')
for before,after in replacements.items():
    assert text.count(before) == 1
    text = text.replace(before,after)
launcher.write_text(text,encoding='utf-8')
record = dict(status='FAILED_SSH_TRANSPORT_PRESERVED_FIXED_PATHS_REVIEW_AND_STATIC_ROOT_READ_PENDING',
    transport_exit=255,stdout_bytes=0,stderr_bytes=len(stderr),error_kind='HOST_KEY_VERIFICATION_FAILED',
    old_launcher_sha256=old_hash,new_launcher_sha256=hashlib.sha256(launcher.read_bytes()).hexdigest(),
    archived_old_launcher=str(archive/'REVIEWED_LAUNCHER_ATTEMPT1.py'),
    cause_to_verify='Resolved Windows junction path entered SSH option/askpass; replace with unchanged known working C:/ forward-slash route',
    strict_host_key_checking_preserved=True,model_and_checker_changed=False,
    remote_CPU_execution_not_proven=True,current_training_queries=0,GPU_calls=0)
(f/'TRANSPORT_FIX_PREPARATION.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
print(json.dumps(record))
