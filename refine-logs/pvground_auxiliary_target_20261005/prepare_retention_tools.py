"""Prepare closed-only cleanup; do not touch any current remote weight."""
import ast
import hashlib
import json
from pathlib import Path


root = Path(__file__).resolve().parent
old = root.parent / 'pvground_query_supported_geometry_20261005'
source = (old / 'retain_metric_best.py').read_text(encoding='utf-8')
changes = {
    "assert str(root)=='/root/autodl-tmp/pvground_query_supported_geometry_20261005'":
        "assert str(root)=='/root/autodl-tmp/pvground_auxiliary_target_20261005'",
    "assert str(parent)=='/root/autodl-tmp/pvground_boundary_fit_20261004/distribution/terminal.pth'":
        "assert str(parent)=='/root/autodl-tmp/pvground_query_supported_geometry_20261005/query_supported/terminal.pth'",
    "query_supported=root/'query_supported/terminal.pth'":
        "member_target=root/'member_target/terminal.pth'",
    "for arm in ('control','query_supported'):": "for arm in ('control','member_target'):",
    "payload['geometry_parent_fit_updates']==3723 and payload['total_geometry_fit_updates']==7446":
        "payload['geometry_parent_fit_updates']==7446 and payload['total_geometry_fit_updates']==11169",
    "assert payload['checkpoint_sha256']==spec['checkpoint_sha256'] and payload['source_port_sha256']==spec['source_port_sha256']":
        "assert payload['checkpoint_sha256']==spec['checkpoint_sha256'] and payload['source_port_sha256']==spec['source_port_sha256']\n    assert payload['geometry_terminal_sha256']==spec['geometry_terminal_sha256'] and payload['extra_geometry_weight']==1.0\n    assert payload['auxiliary_target_mode']==('native_gt' if arm=='control' else 'member_gt')",
    "decision['hits']['protected_geometry_parent']==[5616,4506]":
        "decision['hits']['protected_geometry_parent']==[5614,4509]",
}
for before, after in changes.items():
    assert source.count(before) == 1, before
    source = source.replace(before, after)
assert not (root / 'retain_metric_best.py').exists()
ast.parse(source)
(root / 'retain_metric_best.py').write_text(source, encoding='utf-8')
assert not (root / 'retain_metric_best_authorized.py').exists()
(root / 'retain_metric_best_authorized.py').write_bytes((old / 'retain_metric_best_authorized.py').read_bytes())
ast.parse((root / 'retain_metric_best_authorized.py').read_text(encoding='utf-8'))
record = dict(status='CLOSED_ONLY_RETENTION_TOOLS_PREPARED_NOT_EXECUTED',
              source_review_pending=True, terminal_integrity_review_pending=True,
              weights_deleted=0, inference_or_optimizer_replayed=False,
              files={name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                     for name in ('retain_metric_best.py', 'retain_metric_best_authorized.py')})
(root / 'RETENTION_TOOLS.json').write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
print(json.dumps(record), flush=True)
