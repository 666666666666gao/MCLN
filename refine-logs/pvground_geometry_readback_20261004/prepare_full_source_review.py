"""List actual current caller/factory/source dependencies for a fresh source gate."""
import ast
import datetime
import hashlib
import json
from pathlib import Path

local = Path(__file__).resolve().parent
assert not (local / 'FULL_SOURCE_REVIEW_BINDINGS.json').exists()
face = local.parent / 'pvground_face_conditioned_20261004'
prior_review = json.loads((face / 'EXPERIMENT_CODE_REVIEW.json').read_bytes())
paths = sorted((local / 'runtime_bundle').iterdir())
paths += [local / name for name in ('readback_preflight_controller.py', 'create_remote_readback_source.py',
    'launch_readback_preflight_authorized.py', 'EXPERIMENT_PLAN_READBACK.md',
    'evidence_hidden_preflight_template.json', 'evidence_visible_preflight_template.json',
    'PREFLIGHT_BUNDLE_SOURCE_CHECK.json')]
paths += [local / 'source_preview/PV-Ground' / name for name in ('models/pv_ground.py', 'models/modules.py')]
for name in ('evaluator.py', 'main_utils.py', 'models.losses.py', 'prepare_data.py', 'src.joint_det_dataset.py'):
    item = next(item for item in prior_review['reviewed_files'] if Path(item['path']).name == name)
    path = Path(item['path'])
    assert hashlib.sha256(path.read_bytes()).hexdigest() == item['sha256']
    paths.append(path)
boundary = local.parent / 'pvground_boundary_distribution_20261004'
paths += [boundary / 'complete/distribution/load.json', boundary / 'distribution_spec.json']
bindings = []
for path in paths:
    raw = path.read_bytes()
    if path.suffix == '.py':
        ast.parse(raw.decode('utf-8'), filename=str(path), feature_version=(3, 7))
    bindings.append(dict(path=str(path.resolve()), bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest()))
(local / 'FULL_SOURCE_REVIEW_BINDINGS.json').write_text(json.dumps(bindings, indent=2) + '\n', encoding='utf-8')
request = '''Review this exact isolated experiment implementation before deployment. Read the plan and ALL listed caller/factory/helper/model/data/loss/scorer sources directly. This is the full CPU/two-update-preflight source milestone only; there is no formal runner or accuracy claim. The old partial source PASS does not cover these bytes.

Check the real parent reconstruction, semantic-head defer and once-only ordering, frozen official/G/flat geometry states and eval behavior, all256 candidates, same-Query Box/Mask, exact44-D evidence schema and face order, original full encoded text, native token-softmax score reductions, actual final CE+G gradient isolation, hidden/visible capacity and initialization equality, batch/seed/source/import interfaces, two updates and strict memory model/optimizer restoration, and dependency on actual runtime fields. Check source overlay/upload, exact own paths and GPU lifecycle. The source_port_sha256 field is intentionally added by the reviewed remote sealer after it creates the new full source manifest; templates are not executable until sealed. No fallback, extra compatibility or speculative machinery.

Report actual correctness defects as blocking_findings, grounded nonblocking issues and exact proposed fixes. Say plainly if correct. No SSH, CUDA, model/weight imports or runs, network, credentials/private memory/raw goal/auth wrapper access, other agents, or primary file changes. Stdlib source inspection/AST arithmetic is allowed. Write only READBACK_FULL_SOURCE_REVIEW.md and READBACK_FULL_SOURCE_REVIEW.json in this request directory, with verdict PASS|WARN|FAIL, execution_scope=SOURCE_ONLY, blocking_findings[], nonblocking_findings[], reviewed_files[] exact path/bytes/sha256 identities, review_independence=same-family, acceptance_status=provisional. Requested Astra/max backend/effort is not independently attested. Scope CPU/two-update deployment only; do not approve future formal training, actual runtime or accuracy. Provide final response.

Scope limits: cooperating operator, no hypothetical local malicious user; do not propose new hashes/pins/fallbacks/frameworks or unrelated refactors. Existing source/weight identity invariants may be checked. Remote commands use the operator's compute and credentials: actual trust-boundary defects are in scope. This is not a request for more experiments or a redesign.

Exact existing files:
'''
request += '\n'.join(item['path'] for item in bindings) + '\n'
(local / 'FULL_SOURCE_REVIEW_REQUEST.txt').write_text(request, encoding='utf-8')
(local / 'FULL_SOURCE_AST_CHECK.json').write_text(json.dumps(dict(
    status='PYTHON37_AST_ONLY_FULL_PREFLIGHT_SOURCE_PREPARED', time_cst=datetime.datetime.now().astimezone().isoformat(),
    files=len(bindings), python_files=sum(Path(item['path']).suffix == '.py' for item in bindings),
    native_factory_constructed=False, gpu_updates=0, accuracy_result=False), indent=2) + '\n', encoding='utf-8')
print(json.dumps(dict(files=len(bindings), status='FULL_SOURCE_GATE_READY_NOT_CALLED', runtime=False)), flush=True)
