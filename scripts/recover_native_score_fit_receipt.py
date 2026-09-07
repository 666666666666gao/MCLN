"""Finalize the existing v2 rows after its final relative-path receipt error.

This one-shot recovery preserves the failed controller and original source.
It performs no model construction, new inference or parameter update.
"""
import argparse
import datetime
import gzip
import hashlib
import json
from pathlib import Path
import sys


def sha(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', required=True, type=Path)
    args = parser.parse_args()
    root = args.directory.resolve()
    manifest_path = root / 'input_manifest.json'
    manifest = json.loads(manifest_path.read_text())
    assert int((root / 'controller.exit').read_text()) == 1
    log = (root / 'run.log').read_text()
    assert "FileNotFoundError: [Errno 2] No such file or directory: 'input_manifest.json'" in log
    assert "'manifest_sha256': sha(option.manifest), 'shards': shards" in log
    progress = [json.loads(line.split('NATIVE SCORE FIT AUDIT ', 1)[1]) for line in log.splitlines()
                if line.startswith('NATIVE SCORE FIT AUDIT ')]
    assert [r['rows'] for r in progress] == list(range(12, 505, 12)) + [512]
    original_script = root / 'run_scanrefer_native_score_fit_audit.py'
    for name, expected in manifest['files'].items():
        assert sha(root / name) == expected, name
    # The exact source and traceback establish that these assertions were reached
    # before writing all four shards and failing in the following receipt literal.
    source_text = original_script.read_text()
    state_check = source_text.index('assert all(torch.equal(value.cpu(), initial[name])')
    final_data_check = source_text.index("verify_scanrefer_superpoints(manifest['data_root'], 'train', manifest['train_superpoint_files'])", state_check)
    shard_write = source_text.index('    shards = []')
    receipt_write = source_text.index("    receipt = {'status': 'complete'")
    assert state_check < final_data_check < shard_write < receipt_write
    assert 'except' not in source_text and 'try:' not in source_text
    for name, item in manifest['artifacts'].items():
        assert sha(Path(item['path'])) == item['sha256'], name
    source_manifest = Path(manifest['source_manifest'])
    assert sha(source_manifest) == manifest['source_manifest_sha256']
    for name, expected in json.loads(source_manifest.read_text())['files'].items():
        assert sha(Path(manifest['model_source']) / name) == expected, name
    sys.path.insert(0, str(root))
    from scanrefer_data_contract import verify_scanrefer_superpoints
    data = verify_scanrefer_superpoints(manifest['data_root'], 'train', manifest['train_superpoint_files'])
    rows, shards = [], []
    for start in range(0, 512, 128):
        filename = 'rows_{:03d}_{:03d}.json.gz'.format(start, start + 127)
        decoded = gzip.decompress((root / filename).read_bytes())
        part = json.loads(decoded)
        assert len(part) == 128
        rows.extend(part)
        shards.append({'file': filename, 'rows': 128, 'sha256': sha(root / filename),
                       'uncompressed_sha256': hashlib.sha256(decoded).hexdigest()})
    assert [r['row_id'] for r in rows] == manifest['selected_row_ids']
    assert all(len(r['boxes']) == len(r['query_ious']) == len(r['native_scores']) == 256 for r in rows)
    receipt = {'status': 'complete', 'schema': 'mcln-native-score-fit-result-v1',
               'time_cst': datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
               'rows': 512, 'scope': 'Previously seen fit rows; diagnostic counterfactual not deployed result',
               'manifest_sha256': sha(manifest_path), 'shards': shards, 'superpoint_inputs': data,
               'model_and_source_unchanged': True, 'model_tensors': 1144,
               'optimizer_steps': 0, 'checkpoint_writes': 0, 'formal_rows': 0,
               'forward_elapsed_seconds_from_log': progress[-1]['elapsed_seconds'],
               'max_gpu_mib': None,
               'map_count_histogram': {str(n): sum(len(r['main_map_values']) == n for r in rows)
                                       for n in sorted({len(r['main_map_values']) for r in rows})},
               'top1_disagreements': sum(r['native_query'] != r['adapter_query'] for r in rows),
               'native_top1_excluded_original': sum(r['native_query'] not in r['original_candidates'] for r in rows),
               'native_score_counterfactual_membership_changes': sum(set(r['original_candidates']) != set(r['native_score_counterfactual_candidates']) for r in rows),
               'historical_point_mismatches': sum(not r['point_sha_matches_historical'] for r in rows),
               'recovery': {'original_controller_exit': 1, 'new_model_forwards': 0,
                            'original_script_sha256': sha(original_script), 'original_log_sha256': sha(root / 'run.log'),
                            'finalizer_sha256': sha(Path(__file__).resolve()),
                            'state_check_evidence': 'Original exact-source linear execution passed state equality, source/data and evaluator checks before four shard writes; traceback is in following receipt expression.',
                            'missing_runtime_fields': ['max_gpu_mib', 'total_elapsed_seconds'],
                            'reason': 'Final receipt sha(option.manifest) used a relative CLI path after os.chdir(source); original rows are complete.'}}
    with (root / 'receipt.json').open('x') as stream:
        json.dump(receipt, stream, sort_keys=True, allow_nan=False)
        stream.write('\n')
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()
