"""Create an isolated, explicit source-memory port; leave the corrected parent intact."""
from pathlib import Path
import datetime
import hashlib
import json
import shutil


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def replace_once(text, old, new):
    assert text.count(old) == 1, old
    return text.replace(old, new)


runtime = Path('/root/autodl-tmp/mcln_pvground_runtime_20260908_v1')
parent = Path('/root/autodl-tmp/mcln_pvground_vsa_order_source_20260908_v1/PV-Ground')
root = Path('/root/autodl-tmp/mcln_pvground_source_query_source_20260908_v1')
expected = {
    'models/pv_utils.py': '33279bc4e0302250025eb963f956f164cbe3f9b265ac57d9d4826abf6c370810',
    'models/pv_ground.py': 'fca2dcb6b5dd1dd59b9f4e12f999664f1aa6730374aef6ed09a66fe0a2f0a2f7',
    'models/encoder_decoder_layers.py': '6be03732f8067a6d6138d7a4f33378e7448df61ed9bda641db9c590160edd1d0',
}
assert not root.exists()
texts = {}
for name, digest in expected.items():
    raw = (parent / name).read_bytes()
    assert sha(raw) == digest, name
    texts[name] = raw.decode()
name = 'models/pv_utils.py'
texts[name] = replace_once(texts[name],
    "        # batch_dict['point_features_before_fusion'] = point_features.view(-1, point_features.shape[-1])",
    "        batch_dict['point_features_before_fusion'] = point_features.view(batch_size, self.n_keypoints, -1)")
name = 'models/pv_ground.py'
texts[name] = replace_once(texts[name],
    "        end_points['seed_features'] = voxel_batch['point_features']",
    "        end_points['seed_features'] = voxel_batch['point_features']\n        end_points['source_features'] = voxel_batch['point_features_before_fusion']")
texts[name] = replace_once(texts[name],
    '        spatial_point_xyz=calc_pairwise_locs(points_xyz)',
    '        spatial_point_xyz=calc_pairwise_locs(points_xyz)\n        point_position = self.pos_embed(points_xyz).transpose(1, 2).contiguous()')
texts[name] = replace_once(texts[name],
    '            pos_feats=self.pos_embed(points_xyz).transpose(1, 2).contiguous(),',
    '            pos_feats=point_position,')
texts[name] = replace_once(texts[name],
    '                detected_mask=detected_mask if self.butd else None\n            )',
    "                detected_mask=detected_mask if self.butd else None,\n                source_features=end_points['source_features'],\n                source_position=point_position\n            )")
name = 'models/encoder_decoder_layers.py'
# Restrict replacements to BiDecoderLayer, since Encoder has similar visual attention.
prefix, decoder = texts[name].split('class BiDecoderLayer', 1)
decoder = replace_once(decoder, '        self.cross_v = deepcopy(self.cross_l)',
    '        self.cross_v = deepcopy(self.cross_l)\n        self.source_query_read = None')
decoder = replace_once(decoder, '                detected_feats=None, detected_mask=None):',
    '                detected_feats=None, detected_mask=None,\n                source_features=None, source_position=None):')
decoder = replace_once(decoder, '        query = self.norm_v(query + self.dropout_v(query2))',
    '        if self.source_query_read is not None:\n            query2 = query2 + self.source_query_read(\n                query + query_pos, source_features, source_position)\n        query = self.norm_v(query + self.dropout_v(query2))')
texts[name] = prefix + 'class BiDecoderLayer' + decoder
for name, text in texts.items():
    compile(text, name, 'exec')
root.mkdir()
shutil.copytree(parent, root / 'PV-Ground', ignore=shutil.ignore_patterns('__pycache__'))
for name, text in texts.items():
    (root / 'PV-Ground' / name).write_text(text)
files = {}
for path in parent.rglob('*'):
    if path.is_file() and '__pycache__' not in path.parts:
        name = path.relative_to(parent).as_posix()
        assert name in texts or path.read_bytes() == (root / 'PV-Ground' / name).read_bytes(), name
        files[name] = sha((root / 'PV-Ground' / name).read_bytes())
for name, digest in expected.items():
    assert sha((parent / name).read_bytes()) == digest
record = dict(status='prepared', time_cst=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
    root=str(root), model_source=str(root / 'PV-Ground'), corrected_parent=str(parent),
    parent_sha256=expected, changed_sha256={name:files[name] for name in texts}, files=files,
    parent_unchanged=True, runtime_unchanged=True, model_forwards=0, optimizer_steps=0, formal_rows=0,
    disk_free=shutil.disk_usage(root).free)
(root / 'source_port.json').write_text(json.dumps(record, indent=2) + '\n')
print(json.dumps({k:v for k,v in record.items() if k!='files'}))
