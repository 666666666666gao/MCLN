"""Create an isolated D source; leave the sealed C implementation unchanged."""
from pathlib import Path
import datetime, hashlib, json, shutil

def sha(raw): return hashlib.sha256(raw).hexdigest()

def replace_once(text, old, new):
    assert text.count(old) == 1, old
    return text.replace(old, new)

parent = Path('/root/autodl-tmp/mcln_pvground_observation_source_20260909_v1/PV-Ground')
root = Path('/root/autodl-tmp/mcln_pvground_task_observation_source_20260917_v1')
expected = {
    'models/pv_ground.py': '9e60c9a7cd142073faff6331ab7d7e0e669f04342d4c6916d30eb3a9ebaf3e2b',
    'models/encoder_decoder_layers.py': 'dbdab84c896443efd8537b3f0088277021ac00aaf5164e7ea982354c53df383a',
    'models/modules.py': '4b1bfa7c76d2e9b9b7f071a78680f3412d7d2a3c787fb3a917a585c5bd97328e',
}
assert not root.exists()
texts = {}
for name, digest in expected.items():
    raw = (parent / name).read_bytes()
    assert sha(raw) == digest, name
    texts[name] = raw.decode()
name = 'models/encoder_decoder_layers.py'
texts[name] = 'from pvground_task_observation_query import finish_task_queries\n' + texts[name]
texts[name] = replace_once(texts[name], '        self.source_query_read = None',
                           '        self.source_query_read = None\n        self.task_read = False')
texts[name] = replace_once(texts[name], '        if self.source_query_read is not None:\n',
    '        if self.task_read:\n'
    '            residuals = self.source_query_read(\n'
    '                query + query_pos, source_features, source_position, source_observations)\n'
    '            return finish_task_queries(self, query, query2, residuals)\n'
    '        if self.source_query_read is not None:\n')
name = 'models/pv_ground.py'
texts[name] = replace_once(texts[name], '            )  # (B, V, F)\n            # step project',
    '            )  # (B, V, F), or semantic/geometry pair on the final D layer\n'
    '            if self.decoder[i].task_read:\n'
    '                query, geometry_query = query\n'
    '            else:\n'
    '                geometry_query = query\n'
    '            # step project')
texts[name] = replace_once(texts[name],
    '                prefix=prefix\n            )\n            base_xyz = base_xyz.detach().clone()',
    '                prefix=prefix,\n'
    '                geometry_features=geometry_query.transpose(1, 2).contiguous()\n'
    '            )\n            base_xyz = base_xyz.detach().clone()')
texts[name] = replace_once(texts[name], '            query_last = query\n',
                           '            query_last = geometry_query\n')
name = 'models/modules.py'
texts[name] = replace_once(texts[name], "    def forward(self, features, base_xyz, end_points, prefix=''):",
    "    def forward(self, features, base_xyz, end_points, prefix='', geometry_features=None):")
texts[name] = replace_once(texts[name], '        net = features  # ([B, C=288, num_proposal=256])',
    '        net = features if geometry_features is None else geometry_features\n'
    '        # Semantic scores use features; center/size use the geometric read.')
for name, text in texts.items(): compile(text, name, 'exec')
root.mkdir()
shutil.copytree(parent, root/'PV-Ground', ignore=shutil.ignore_patterns('__pycache__'))
for name, text in texts.items(): (root/'PV-Ground'/name).write_text(text)
files = {}
for path in parent.rglob('*'):
    if path.is_file() and '__pycache__' not in path.parts:
        name = path.relative_to(parent).as_posix()
        assert name in texts or path.read_bytes() == (root/'PV-Ground'/name).read_bytes(), name
        files[name] = sha((root/'PV-Ground'/name).read_bytes())
for name, digest in expected.items(): assert sha((parent/name).read_bytes()) == digest
record = dict(status='prepared', time_cst=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
    root=str(root), model_source=str(root/'PV-Ground'), parent_source=str(parent), parent_sha256=expected,
    changed_sha256={name:files[name] for name in texts}, files=files,
    parent_unchanged=True, runtime_unchanged=True, model_forwards=0, optimizer_steps=0,
    formal_rows=0, disk_free=shutil.disk_usage(root).free)
(root/'source_port.json').write_text(json.dumps(record, indent=2)+'\n')
print(json.dumps({k:v for k,v in record.items() if k!='files'}))
