"""Read-only publication preconditions and Git clean-filter byte convention."""
import ast
import datetime
import hashlib
import json
from pathlib import Path
import subprocess

root = Path(__file__).resolve().parent.parent
workspace = Path('C:/Users/gb')
previous = json.loads((root.parent / 'pvground_mask_geometry_responsibility_20261005/publication.json').read_bytes())
repos = [workspace / name for name in ['.codex_mcln_g0_20260905', '.codex_pvground_cs_20261002', '.codex_mcln_v99_internal_20260928']]
doc = 'docs/MCLN_CURRENT_COMPLETE_HANDOFF_2026-08-15.md'
copies = [repo / doc for repo in repos] + [workspace / 'Desktop/document' / Path(doc).name]
old = copies[0].read_bytes()
assert hashlib.sha256(old).hexdigest() == previous['handoff_sha256']
assert all(path.read_bytes() == old for path in copies)
assert not (root / 'publication.json').exists()
tree = ast.parse((root / 'publish_branch.py').read_text(encoding='utf-8'))
name_assignment = next(node for node in tree.body if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == 'names' for target in node.targets))
names = [node.value for node in name_assignment.value.elts if isinstance(node, ast.Constant)] + ['publish_branch.py']
for folder in ['complete', 'analysis', 'observations']:
    names += [path.relative_to(root).as_posix() for path in sorted((root / folder).rglob('*')) if path.is_file()]
prefix = 'refine-logs/pvground_mask_branch_responsibility_20261005/'
checks = []
for index, repo in enumerate(repos):
    head = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD']).decode().strip()
    assert head == previous['heads'][index]
    assert not subprocess.check_output(['git', '-C', str(repo), 'status', '--porcelain'])
    checked = 0
    converted = []
    if index < 2:
        for name in names:
            raw = (root / name).read_bytes()
            expected = raw if name.endswith('.npz') else raw.replace(b'\r\n', b'\n')
            filtered = subprocess.check_output(['git', '-C', str(repo), 'hash-object', '--path=' + prefix + name, '--stdin'], input=raw, stderr=subprocess.DEVNULL).decode().strip()
            assert filtered == hashlib.sha1(b'blob ' + str(len(expected)).encode() + b'\0' + expected).hexdigest(), name
            checked += 1
            if raw != expected:
                converted.append(name)
    doc_expected = old.replace(b'\r\n', b'\n')
    doc_filtered = subprocess.check_output(['git', '-C', str(repo), 'hash-object', '--path=' + doc, '--stdin'], input=old, stderr=subprocess.DEVNULL).decode().strip()
    assert doc_filtered == hashlib.sha1(b'blob ' + str(len(doc_expected)).encode() + b'\0' + doc_expected).hexdigest()
    assert subprocess.check_output(['git', '-C', str(repo), 'show', 'HEAD:' + doc]) == doc_expected
    checks.append(dict(repo=str(repo), head=head, clean=True, payloads_checked=checked,
                       text_payloads_with_crlf_normalization=len(converted), doc_git_lf_exact=True))

result = dict(status='PASS', generated_at=datetime.datetime.now().astimezone().isoformat(),
              publisher_executed=False, remote_calls=0, git_mutations=0,
              prepublication_heads_and_four_raw_handoff_copies_exact=True,
              existing_payload_count=len(names), repos=checks,
              byte_convention='Local and remote intake payloads are raw bytes. Git text blobs normalize CRLF to LF under the observed core.autocrlf=true configuration; eight NPZ blobs stay byte-exact.',
              scope='Read-only current preconditions and Git filters for existing payloads. No assertion of future remote connectivity, publication, or training.')
(root / 'analysis/AUDIT_PUBLISHER.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
print(json.dumps(result))
