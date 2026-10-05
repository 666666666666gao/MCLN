"""Reuse the completed predecessor's read-only intake and CPU recount tools."""
import ast
import hashlib
import json
from pathlib import Path


root = Path(__file__).resolve().parent
old = root.parent / 'pvground_query_supported_geometry_20261005'
for name in ('collect_formal_authorized.py', 'geometry_result_metrics.py'):
    assert not (root / name).exists()
    (root / name).write_bytes((old / name).read_bytes())
source = (old / 'analyze_closed_formal.py').read_text(encoding='utf-8')
source = source.replace("'query_supported'", "'member_target'").replace('query_supported_hits50', 'member_target_hits50')
changes = {
    "assert spec['extra_geometry_weight']==(0. if arm=='control' else 1.)":
        "assert spec['extra_geometry_weight']==1.0\n    assert spec['auxiliary_target_mode']==('native_gt' if arm=='control' else 'member_gt')\n    assert fit['extra_geometry_weight']==1.0",
    "assert all(entry['extra_geometry_weight']==spec['extra_geometry_weight'] for entry in logs)":
        "assert all(entry['extra_geometry_weight']==spec['extra_geometry_weight'] and entry['auxiliary_target_mode']==spec['auxiliary_target_mode'] for entry in logs)",
    "parent=local.parent/'pvground_boundary_distribution_20261004/complete/distribution/formal'":
        "parent=local.parent/'pvground_query_supported_geometry_20261005/complete/query_supported/formal'",
    "==[5616,4506]": "==[5614,4509]",
    "best=max(table,key=lambda row:(row['rec_hits50'],row['rec_hits25'],row['system']=='protected_geometry_parent'))":
        "best=max(table,key=lambda row:(row['rec_hits50'],row['system']=='protected_geometry_parent',row['rec_hits25']))",
    "geometry_parent_updates=3723,geometry_total_updates_at_terminal=7446":
        "geometry_parent_updates=7446,geometry_total_updates_at_terminal=11169",
    "# Query-supported geometry responsibility: closed result":
        "# Native/member auxiliary geometry targets: closed result",
    "Single seed2027; two actual same-start full fits, parents and zeroR frozen; existing boundary head only.":
        "Single seed2027; two actual same-start full fits, parents and zeroR frozen; existing boundary head only. Both extra geometry weights1; native/member coordinates change auxiliary qualification and localization targets together, while native GT matching/losses/evaluation remain unchanged.",
}
for before, after in changes.items():
    assert source.count(before) == 1, before
    source = source.replace(before, after)
assert not (root / 'analyze_closed_formal.py').exists()
ast.parse(source)
(root / 'analyze_closed_formal.py').write_text(source, encoding='utf-8')
for name in ('collect_formal_authorized.py', 'geometry_result_metrics.py', 'analyze_closed_formal.py'):
    ast.parse((root / name).read_text(encoding='utf-8'))
record = dict(status='READ_ONLY_CLOSED_RESULT_TOOLS_PREPARED_NOT_EXECUTED',
              predecessor=str(old), model_or_optimizer_replayed=False, weights_downloaded=0,
              files={name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                     for name in ('collect_formal_authorized.py', 'geometry_result_metrics.py', 'analyze_closed_formal.py')})
(root / 'CLOSED_RESULT_TOOLS.json').write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
print(json.dumps(record), flush=True)
