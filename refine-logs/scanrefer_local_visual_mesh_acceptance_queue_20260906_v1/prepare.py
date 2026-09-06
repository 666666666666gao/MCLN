import ast
import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time


repo = Path('C:/Users/gb/.codex_mcln_g0_20260905')
root = repo / 'refine-logs/scanrefer_local_visual_mesh_acceptance_queue_20260906_v1'
post = repo / 'refine-logs/scanrefer_local_visual_mesh_posttraining_20260906_v1'
formal_name = 'scanrefer_local_visual_mesh_official_20260906_v1'
old_audit = (repo / 'refine-logs/scanrefer_local_visual_official_20260906_v3/audit_terminal.py').read_text(encoding='utf-8')
audit = old_audit.replace('scanrefer_local_visual_official_20260906_v3', formal_name)
anchor = "assert hashlib.sha256(manifest_raw).hexdigest() == 'e2b0350fb67ce7e57193394f730cb173161a9706cbd134ed405d071411923fb9'"
assert audit.count(anchor) == 1
binding = '''post = repo / 'refine-logs/scanrefer_local_visual_mesh_posttraining_20260906_v1'
preparation_raw = (post / 'preparation.json').read_bytes()
assert hashlib.sha256(preparation_raw).hexdigest() == '642f8facc62c30b623b37296672bc4c5385c947dabebc15d88eefc65712eec84'
preparation = json.loads(preparation_raw)
launch_raw = (post / 'executed.json').read_bytes()
launch = json.loads(launch_raw)
assert hashlib.sha256(manifest_raw).hexdigest() == launch['manifest_sha256']
assert manifest['training_directory'] == preparation['training_directory']
assert remote == preparation['formal_directory']
assert manifest['data_root'] == '/root/autodl-tmp/DATA_ROOT_mcln_meshsp/'
assert manifest['files'] == {name: preparation['files'][name]
                             for name in preparation['formal_files'] if name != 'scripts/__init__.py'}'''
audit = audit.replace(anchor, binding)
anchor = 'sftp = client.open_sftp()\n'
assert audit.count(anchor) == 1
audit = audit.replace(anchor, anchor + '''with sftp.open(remote + '/input_manifest.json', 'rb') as stream:
    assert stream.read() == manifest_raw
with sftp.open('/root/autodl-tmp/mcln_scanrefer_local_visual_mesh_posttraining_20260906_v1/executed.json', 'rb') as stream:
    assert stream.read() == launch_raw
''')
with (root / 'audit_terminal.py').open('x', encoding='utf-8', newline='\n') as stream:
    stream.write(audit)
native_path = repo / 'refine-logs/native_local_preflight_preparation_20260906_v2/launch_conditional.py'
native = native_path.read_text(encoding='utf-8')
assert native.count('mcln_scanrefer_local_visual_official_20260906_v3') == 1
native = native.replace('mcln_scanrefer_local_visual_official_20260906_v3', 'mcln_' + formal_name)
with (root / 'launch_native_conditional.py').open('x', encoding='utf-8', newline='\n') as stream:
    stream.write(native)
sha = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
dependencies = [
    'refine-logs/scanrefer_local_visual_mesh_posttraining_20260906_v1/preparation.json',
    'refine-logs/scanrefer_local_visual_mesh_posttraining_20260906_v1/queue_launch.json',
    'refine-logs/scanrefer_local_visual_mesh_posttraining_20260906_v1/observation_schedule.json',
    'refine-logs/scanrefer_local_visual_mesh_posttraining_20260906_v1/observe_posttraining.py',
    'refine-logs/native_local_preflight_preparation_20260906_v2/receipt.json',
    'scripts/run_native_candidate_local_preflight.py',
    'scripts/evaluate_scanrefer_local_visual_official.py',
]
files = ['continue_after_formal.py', 'audit_terminal.py', 'launch_native_conditional.py']
for name in files:
    source = (root / name).read_text(encoding='utf-8')
    ast.parse(source, filename=name)
    compile(source, name, 'exec')
assert 'mcln_scanrefer_local_visual_official_20260906_v3' not in native
assert native.replace('mcln_' + formal_name, 'mcln_scanrefer_local_visual_official_20260906_v3') == native_path.read_text(encoding='utf-8')
sys.path.insert(0, str(repo))
from scripts.evaluate_scanrefer_local_visual_official import promotion_check, row_metrics
old_rows = json.loads((repo / 'refine-logs/scanrefer_local_visual_official_20260906_v3/result/rows.json').read_bytes())
old_gate = promotion_check(row_metrics(old_rows['protected_v99']), row_metrics(old_rows['local_v99']))
assert not old_gate['advance_to_nr3d_sr3d_rec']
started = time.monotonic()
child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(5)'],
                         creationflags=subprocess.CREATE_NO_WINDOW)
subprocess.check_call(['powershell.exe', '-NoProfile', '-NonInteractive', '-Command',
                       'Wait-Process -Id ' + str(child.pid) + ' -ErrorAction Stop'],
                      creationflags=subprocess.CREATE_NO_WINDOW)
assert child.wait() == 0 and time.monotonic() - started >= 4
plan = {
    'schema': 'mcln-scanrefer-mesh-acceptance-queue-v1',
    'time_cst': datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
    'existing_observer_pid': 50584,
    'formal_directory': '/root/autodl-tmp/mcln_' + formal_name,
    'files': {name: sha(root / name) for name in files},
    'dependencies': {name: sha(repo / name) for name in dependencies},
    'compile_checks': len(files),
    'native_launcher_change': 'Only rebind formal_root from completed negative v3 to the fixed current mesh repeat.',
    'native_launcher_other_source_identical': True,
    'actual_old9508_negative_gate_recomputed': old_gate,
    'actual_windows_process_wait_completed': True,
    'waiting_does_not_poll_remote_training': True,
    'first_formal_check_seconds_after_launch': 1440,
    'formal_poll_seconds': 240,
    'checkpoint_writes': 0,
    'scope': 'Wait existing launch collector, observe actual formal screen, independent CPU audit, conditionally disposable native GPU preflight. No direct Nr/Sr training or gate changes.',
}
with (root / 'plan.json').open('x', encoding='utf-8') as stream:
    json.dump(plan, stream, indent=2, sort_keys=True)
    stream.write('\n')
(root / 'prepare.py').write_bytes(Path(__file__).read_bytes())
print(json.dumps(plan))
