import datetime
import hashlib
import json
from pathlib import Path

base = Path('/root/autodl-tmp/DATA_ROOT/output')
contracts = {
    'e71': ('preserved_best/mcln_pair_sweep/mcln_pair_default_rankblend010_2ep_best_acc025_epoch71_0.57993.pth', '3e44f4bdad3bd66ad82102032e1cb0241de57d147c0aa1d3eff9736926ef2208'),
    'parent': ('rec_reranker/e71_top16/artifacts/reranker_h256_d010_lr1e3_seed0_final_contract.pth', 'f06f8972fdcfbbdcb799df267864ab2ebc9ca8403ff92576e2bbdb0a8c17269b'),
    'geometry': ('rec_reranker/e71_top16/geometry_artifacts/selected_geometry_reranker.pth', '835c25be4717dfcbb324e0c4c5b9d1d3f3e2b90a4dbcb4d4ebe79f215f263b6f'),
    'v99': ('rec_reranker/e71_top16/v99_artifacts/pareto_contextual_h128_seed0_fullfit.pth', '9752990c393fa6e45173a9dd129c4de4bb740924094dcbbec2f3121cbf39d1f2')}
records = {}
for label, (relative, expected) in contracts.items():
    path = base / relative
    record = {'path': str(path), 'expected_sha256': expected, 'exists': path.is_file()}
    if record['exists']:
        stat = path.stat()
        digest = hashlib.sha256()
        with path.open('rb') as f:
            for chunk in iter(lambda: f.read(8388608), b''):
                digest.update(chunk)
        record.update({'bytes': stat.st_size, 'mode': oct(stat.st_mode & 0o777),
                       'sha256': digest.hexdigest(), 'hash_matches': digest.hexdigest() == expected})
        assert path.stat().st_mtime_ns == stat.st_mtime_ns
    records[label] = record
result = {'time_cst': datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
          'artifacts': records, 'all_hashes_match': all(r.get('hash_matches', False) for r in records.values()),
          'model_forwards': 0, 'optimizer_steps': 0, 'files_modified': False,
          'scope': 'Exact four protected weight artifacts only; not a new inference or environment validation.'}
out = Path('/root/autodl-tmp/mcln_eg3dvg_acceptance_20260920_v1/protected_scanrefer_artifact_check_20260920.json')
with out.open('x') as f:
    json.dump(result, f, indent=2)
print(json.dumps(result, indent=2))
