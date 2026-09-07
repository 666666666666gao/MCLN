"""Collect a finished fixed pair, preserving exact rows and recomputing deployed hits."""
import gzip
import hashlib
import json
import os
from pathlib import Path

import paramiko

repo = Path('C:/Users/gb/.codex_mcln_g0_20260905')
local = repo / 'refine-logs/scanrefer_mask_geometry_pair_20260907_v1'
remote = '/root/autodl-tmp/mcln_scanrefer_mask_geometry_pair_20260907_v1'
queue_remote = '/root/autodl-tmp/mcln_scanrefer_mask_geometry_posttraining_20260907_v1'
queue_local = repo / 'refine-logs/scanrefer_mask_geometry_posttraining_20260907_v1'
c = paramiko.SSHClient()
c.load_system_host_keys()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect('region-9.autodl.pro', port=33476, username='root', password=os.environ['MCLN_SSH_PASSWORD'], timeout=30)
s = c.open_sftp()
names = ['controller.exit', 'training.exit', 'receipt.json', 'independent_audit.json',
    'fit_complete.json', 'fit_point_batches.json', 'terminal_rows.json',
    'terminal_metrics.json', 'terminal_native_metrics.json', 'terminal_geometry_metrics.json']
hashes = {}
compressed = {}
for name in names:
    with s.open(remote + '/' + name, 'rb') as stream:
        stream.prefetch(file_size=stream.stat().st_size)
        raw = stream.read()
    path = local / name
    if path.exists():
        assert path.read_bytes() == raw, name
    else:
        path.write_bytes(raw)
    hashes[name] = hashlib.sha256(raw).hexdigest()
    if name in ['controller.exit', 'training.exit']:
        assert raw.strip() == b'0', (name, raw)
    if name in ['fit_point_batches.json', 'terminal_rows.json']:
        packed = gzip.compress(raw, compresslevel=6, mtime=0)
        assert gzip.decompress(packed) == raw
        (local / (name + '.gz')).write_bytes(packed)
        compressed[name] = {'bytes': len(raw), 'gzip_bytes': len(packed),
                            'sha256': hashes[name], 'gzip_sha256': hashlib.sha256(packed).hexdigest()}
receipt = json.loads((local / 'receipt.json').read_bytes())
audit = json.loads((local / 'independent_audit.json').read_bytes())
assert receipt['status'] == 'complete' and receipt['steps_per_arm'] == 2482
assert receipt['holdout_rows'] == 6887 and receipt['formal_rows'] == 0
assert receipt['schema'] == 'mcln-scanrefer-mask-geometry-gt-pair-v1'
assert audit['status'] == 'pass' and audit['receipt_sha256'] == hashes['receipt.json']
assert receipt['fit_batches_sha256'] == hashes['fit_point_batches.json']
assert receipt['terminal_rows_sha256'] == hashes['terminal_rows.json']
assert receipt['manifest_sha256'] == audit['manifest_sha256'] == '15f46411069a7172a55373c5c13075b22bcb4146d39251e9fca2c37ed5867eb3'
terminal = json.loads((local / 'terminal_rows.json').read_bytes())
baseline = json.loads((local / 'baseline_rows.json').read_bytes())
native = json.loads((local / 'terminal_native_metrics.json').read_bytes())
recounts = {}
for arm, rows in terminal.items():
    assert len(rows) == 6887
    assert [(r['row_id'], r['scan_id'], r['point_sha256']) for r in rows] == [
        (r['row_id'], r['scan_id'], r['point_sha256']) for r in baseline[arm]]
    result = {}
    for field, key, record in [('rec_iou', 'rec_hits', receipt['terminal_metrics'][arm]),
                               ('mask_iou', 'mask_hits', receipt['terminal_metrics'][arm]),
                               ('native_rec_iou', 'rec_hits', native[arm])]:
        hits = [sum(r[field] > threshold for r in rows) for threshold in [.25, .5]]
        assert hits == [record[key + suffix] for suffix in ['025', '050']]
        result[field] = hits
    result['mask_miou'] = sum(r['mask_iou'] for r in rows) / len(rows) * 100
    assert abs(result['mask_miou'] - receipt['terminal_metrics'][arm]['mask_miou']) < 1e-10
    recounts[arm] = result
effects = {}
for field, record_key in [('rec_iou', 'system_rec_effects'), ('native_rec_iou', 'native_rec_effects')]:
    effects[field] = {}
    for reference in ['baseline', 'native_gt']:
        old = baseline['native_gt_mask_geometry'] if reference == 'baseline' else terminal['native_gt']
        new = terminal['native_gt_mask_geometry']
        effects[field][reference] = {}
        for threshold in [.25, .5]:
            repair = sum(a[field] <= threshold < b[field] for a, b in zip(old, new))
            damage = sum(b[field] <= threshold < a[field] for a, b in zip(old, new))
            value = {'repair': repair, 'damage': damage, 'net': repair - damage}
            assert value == receipt[record_key][reference][str(threshold)]
            effects[field][reference][str(threshold)] = value
eligible = all(item['net'] >= 0 for row in effects['rec_iou'].values() for item in row.values())
assert eligible == receipt['eligible_for_fixed_terminal_formal_evaluation'] == audit['eligible_for_fixed_terminal_formal_evaluation']
queue_decision = None
if 'decision.json' in s.listdir(queue_remote):
    for name in ['decision.json', 'controller.exit']:
        with s.open(queue_remote + '/' + name, 'rb') as stream:
            raw = stream.read()
        path = queue_local / name
        if path.exists():
            assert path.read_bytes() == raw
        else:
            path.write_bytes(raw)
    assert (queue_local / 'controller.exit').read_text().strip() == '0'
    queue_decision = json.loads((queue_local / 'decision.json').read_bytes())
result = {'schema': 'mcln-mask-geometry-terminal-local-collection-v1',
    'training_receipt_sha256': hashes['receipt.json'], 'training_audit_sha256': hashes['independent_audit.json'],
    'file_hashes': hashes, 'lossless_archives': compressed,
    'recomputed_deployed_metrics': recounts, 'recomputed_effects': effects,
    'fixed_candidate_module_rec_pass': eligible, 'formal_rows_in_this_collection': 0,
    'queue_decision': queue_decision, 'checkpoints_downloaded': False,
    'geometry_diagnostic_metrics': json.loads((local / 'terminal_geometry_metrics.json').read_bytes())}
with (local / 'terminal_collection.json').open('x', encoding='utf-8') as stream:
    json.dump(result, stream, indent=2, sort_keys=True)
print(json.dumps(result, indent=2), flush=True)
s.close()
c.close()
