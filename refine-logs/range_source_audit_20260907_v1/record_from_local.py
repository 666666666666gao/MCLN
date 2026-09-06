import datetime
import hashlib
import json
import urllib.request
from pathlib import Path

repo = Path('C:/Users/gb/.codex_mcln_g0_20260905')
destination = repo / 'refine-logs/range_source_audit_20260907_v1'
destination.mkdir(exist_ok=False)
sources = [
    ('open-mmlab/OpenPCDet', '233f849829b6ac19afb8af8837a0246890908755', 'pcdet/models/roi_heads/pvrcnn_head.py', ['def roi_grid_pool', 'def get_dense_grid_points', 'dense_idx + 0.5']),
    ('open-mmlab/OpenPCDet', '233f849829b6ac19afb8af8837a0246890908755', 'pcdet/ops/pointnet2/pointnet2_stack/pointnet2_utils.py', ['empty_ball_mask = (idx[:, 0] == -1)', 'grouped_xyz[empty_ball_mask] = 0', 'grouped_features[empty_ball_mask] = 0']),
    ('open-mmlab/OpenPCDet', '233f849829b6ac19afb8af8837a0246890908755', 'pcdet/ops/pointnet2/pointnet2_stack/src/ball_query_gpu.cu', ['for (int k = 0; k < n; ++k)', 'if (cnt == 0)', 'idx[0] = -1']),
    ('open-mmlab/OpenPCDet', '233f849829b6ac19afb8af8837a0246890908755', 'pcdet/ops/pointnet2/pointnet2_stack/pointnet2_modules.py', ['class StackSAModuleMSG', 'new_features = self.mlps[k](new_features)', 'F.max_pool2d']),
    ('open-mmlab/OpenPCDet', '233f849829b6ac19afb8af8837a0246890908755', 'tools/cfgs/kitti_models/pv_rcnn.yaml', ['GRID_SIZE: 6', 'NSAMPLE: [16, 16]', 'POOL_RADIUS: [0.8, 1.6]']),
    ('open-mmlab/OpenPCDet', '233f849829b6ac19afb8af8837a0246890908755', 'LICENSE', ['Apache License', 'Version 2.0']),
    ('tiny-smart/box-detr', '053bd1f65159e431db7a0ab17a12413db1c7b8ae', 'models/box_detr/transformer.py', ['self.point_offset = MLP', 'point_offset = self.point_offset(output)', 'agent = box[..., :2]', 'self.ca_qpos_sine_proj = nn.Conv1d']),
]
records = []
for repository, commit, path, anchors in sources:
    url = 'https://raw.githubusercontent.com/' + repository + '/' + commit + '/' + path
    with urllib.request.urlopen(url, timeout=30) as response:
        raw = response.read()
    lines = raw.decode().splitlines()
    evidence = []
    for anchor in anchors:
        matches = [index + 1 for index, line in enumerate(lines) if anchor in line]
        assert matches, (path, anchor)
        evidence.append({'anchor': anchor, 'lines_one_based': matches})
    records.append({'repository': repository, 'commit': commit, 'path': path, 'url': url,
        'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest(), 'line_evidence': evidence})

local_path = 'models/candidate_range_visual.py'
local_raw = (repo / local_path).read_bytes()
record = {'schema': 'mcln-range-source-audit-v1',
    'time_cst': datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
    'scope': 'Pinned primary source inspection only; no runtime changes or performance result',
    'sources': records,
    'local_module': {'path': local_path, 'bytes': len(local_raw), 'sha256': hashlib.sha256(local_raw).hexdigest()},
    'source_content_copied_to_repository': False,
    'training_config_changed': False, 'training_started': False, 'new_performance_evidence': False}
(destination / 'source_receipt.json').write_text(json.dumps(record, indent=2, sort_keys=True) + '\n', encoding='utf-8')
(destination / 'record_from_local.py').write_bytes(Path(__file__).read_bytes())
print(json.dumps({'sources': len(records), 'source_receipt_sha256': hashlib.sha256((destination / 'source_receipt.json').read_bytes()).hexdigest(),
    'local_module_sha256': record['local_module']['sha256']}))
