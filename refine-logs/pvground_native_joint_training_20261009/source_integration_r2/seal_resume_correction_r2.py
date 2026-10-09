"""Seal only the actual normal-training resume-directory correction."""
import ast
import datetime
import difflib
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parent
prior = json.loads((root / 'METRIC_PREPARATION.json').read_bytes())
report = json.loads((root / 'source_review/SOURCE_REVIEW_R1.json').read_bytes())
source = root / 'source'
current = {}
for relative in prior['source_files']:
    path = source / relative
    ast.parse(path.read_text(encoding='utf-8'), filename=relative)
    current[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
changed = [relative for relative in current if current[relative] != prior['source_files'][relative]]
assert changed == ['main_utils.py']
snapshot_root = root / 'source_review/r1_snapshots'
matches = [path for path in snapshot_root.rglob('*.py')
           if hashlib.sha256(path.read_bytes()).hexdigest() == prior['source_files']['main_utils.py']]
assert len(matches) == 1
old = matches[0].read_text(encoding='utf-8')
new = (source / 'main_utils.py').read_text(encoding='utf-8')
diff = ''.join(difflib.unified_diff(old.splitlines(True), new.splitlines(True),
                                 fromfile='R1/main_utils.py', tofile='R2/main_utils.py'))
(root / 'RESUME_CORRECTION_R2.diff').write_text(diff, encoding='utf-8')
receipt = dict(status='B1_RESUME_DIRECTORY_CORRECTED_STATIC_PARSE_ONLY_REVIEW_PENDING',
    time_cst=datetime.datetime.now().astimezone().isoformat(),
    changed_source_files=changed, source_files=current,
    previous_source_receipt_sha256=hashlib.sha256((root / 'METRIC_PREPARATION.json').read_bytes()).hexdigest(),
    previous_review_sha256=hashlib.sha256((root / 'source_review/SOURCE_REVIEW_R1.json').read_bytes()).hexdigest(),
    correction='Training resume uses the checkpoint directory and requires its retained best.pth; fresh-run directory policy unchanged',
    actual_resume_executed=False, actual_gpu_preflight=False,
    native_joint_training_started=False, current_frozen_pair_changed=False,
    three_effective_contributions=False, full_goal_complete=False)
(root / 'RESUME_CORRECTION_R2.json').write_text(json.dumps(receipt, indent=2) + '\n', encoding='utf-8')
print(json.dumps(dict(status=receipt['status'], changed_source_files=changed,
                     source_files_parsed=len(current), neural_calls=0, remote_queries=0)))
