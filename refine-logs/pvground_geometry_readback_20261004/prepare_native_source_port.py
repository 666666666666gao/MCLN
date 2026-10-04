"""Prepare a separate delayed-semantic port; no factory or runtime execution."""
import ast
import datetime
import difflib
import hashlib
import json
from pathlib import Path

local = Path(__file__).resolve().parent
face = local.parent / 'pvground_face_conditioned_20261004'
witness = json.loads((face / 'READBACK_INTERFACE_WITNESS.json').read_bytes())
inputs = {}
for role in ('model', 'head'):
    item = witness['files'][role]
    path = Path(item['path'])
    assert hashlib.sha256(path.read_bytes()).hexdigest() == item['sha256']
    inputs[role] = path.read_text(encoding='utf-8')

model = inputs['model']
anchor = '        self.candidate_box_refiner = None\n'
assert model.count(anchor) == 1
model = model.replace(anchor, anchor + '        self.boundary_evidence_readback = None\n')
anchor = '                geometry_features=geometry_query.transpose(1, 2).contiguous()\n'
assert model.count(anchor) == 1
model = model.replace(anchor,
    '                geometry_features=geometry_query.transpose(1, 2).contiguous(),\n'
    '                defer_semantic=(i == self.num_decoder_layers - 1\n'
    '                    and self.boundary_evidence_readback is not None)\n')
anchor = "            end_points['last_pred_size'] = refined_size\n\n        return end_points\n"
assert model.count(anchor) == 1
model = model.replace(anchor, """            end_points['last_pred_size'] = refined_size

        if self.boundary_evidence_readback is not None:
            assert self.candidate_box_refiner is not None
            end_points['last_semantic_query_before_readback'] = query
            final_query = self.boundary_evidence_readback(
                query, text_feats, text_padding_mask, raw_points, end_points)
            end_points['last_semantic_query_after_readback'] = final_query
            end_points['last_sem_cls_scores'] = self.prediction_heads[-1].sem_cls_scores_head(
                final_query.transpose(1, 2).contiguous()).transpose(2, 1)

        return end_points
""")

head = inputs['head']
anchor = "    def forward(self, features, base_xyz, end_points, prefix='', geometry_features=None):"
assert head.count(anchor) == 1
head = head.replace(anchor, anchor[:-2] + ', defer_semantic=False):')
start = head.index("    def forward(self, features, base_xyz, end_points, prefix='', geometry_features=None,")
constructor, forward = head[:start], head[start:]
assert forward.count('if self.compute_sem_scores:') == 2
head = constructor + forward.replace('if self.compute_sem_scores:', 'if self.compute_sem_scores and not defer_semantic:')

preview = local / 'source_preview/PV-Ground/models'
assert not preview.exists()
preview.mkdir(parents=True)
outputs = {'source_preview/PV-Ground/models/pv_ground.py': model,
           'source_preview/PV-Ground/models/modules.py': head}
for name, source in outputs.items():
    ast.parse(source, feature_version=(3, 7))
    (local / name).write_text(source, encoding='utf-8')
diffs = []
for role, name in (('model', 'source_preview/PV-Ground/models/pv_ground.py'),
                   ('head', 'source_preview/PV-Ground/models/modules.py')):
    diffs.extend(difflib.unified_diff(inputs[role].splitlines(True), outputs[name].splitlines(True),
        fromfile=witness['files'][role]['path'], tofile=name))
(local / 'NATIVE_SOURCE_PORT.diff').write_text(''.join(diffs), encoding='utf-8')
names = list(outputs) + ['install_boundary_evidence_readback.py', 'native_root_bbs.py',
    'prepare_native_source_port.py', 'NATIVE_SOURCE_PORT.diff']
for name in names:
    if name.endswith('.py'):
        ast.parse((local / name).read_text(encoding='utf-8'), feature_version=(3, 7))
receipt = dict(time_cst=datetime.datetime.now().astimezone().isoformat(),
    status='SOURCE_PORT_DRAFT_AST_ONLY', source_inputs={role:witness['files'][role] for role in ('model', 'head')},
    files={name:dict(bytes=(local/name).stat().st_size,
        sha256=hashlib.sha256((local/name).read_bytes()).hexdigest()) for name in names},
    final_semantic_deferred_only_when_readback_installed=True,
    native_semantic_subhead_called_once_by_source_definition=True,
    existing_geometry_and_mask_order_preserved=True,
    native_factory_constructed=False, source_review_pass=False, gpu_forwards=0,
    optimizer_updates=0, active_face_source_changed=False, accuracy_result=False)
(local / 'NATIVE_SOURCE_PORT_CHECK.json').write_text(json.dumps(receipt, indent=2)+'\n', encoding='utf-8')
print(json.dumps(dict(status=receipt['status'], files=len(names), native_factory=False,
    active_face_source_changed=False, accuracy=False)))
