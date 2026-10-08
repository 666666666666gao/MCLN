"""Independent, CPU-only, standard-library audit of saved PV-Ground results.

Does not import or execute experiment modules. Float32 NPY inputs are decoded
directly; geometric IoUs use Python double arithmetic, not the executor's NumPy
implementation. Native float32 values are compared with an explicit tolerance,
and every reported threshold decision must agree exactly.
"""
import array
import ast
import collections
import datetime
import hashlib
import json
import math
from pathlib import Path
import re
import struct
import sys
import time


ROOT = Path(r'C:\Users\gb\.codex\tmp\pvground_pretrained_protocol_20261009')
OUT = ROOT / 'formal_analysis'
ACTUAL = ROOT / 'preflight_r3/formal_actual'
HIST = Path(r'C:\Users\gb\.codex_mcln_g0_20260905\refine-logs\pvground_scanrefer_formal_20260918_semantic_assignment_v1')
AUTHOR = Path(r'C:\Users\gb\.codex\tmp\pvground_compressed_geometry_support_20261008\pretrained_reproduction_question')
SOURCE = Path(r'C:\Users\gb\.codex\tmp\pvground_mask_support_correction_20261008_v2\complete_fit\PV-Ground')
READS = {}
CHECKS = []
START = time.perf_counter()


def read(path):
    path = Path(path).resolve()
    data = path.read_bytes()
    meta = {'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data)}
    if path.suffix != '.npy':
        meta['lines'] = len(data.splitlines())
    key = str(path)
    if key in READS:
        assert READS[key] == meta, 'Input changed during audit: ' + key
    READS[key] = meta
    return data


def js(path):
    return json.loads(read(path))


def lines(path):
    return [json.loads(line) for line in read(path).decode('utf-8').splitlines()]


def check(name, passed, detail=None):
    CHECKS.append({'name': name, 'pass': bool(passed), 'detail': detail})


def digest_matches(path, digest, size=None):
    data = read(path)
    ok = hashlib.sha256(data).hexdigest() == digest
    if size is not None:
        ok = ok and len(data) == size
    check('file_hash:' + str(path), ok)


def npy_f32(path, shape):
    data = read(path)
    assert data[:8] == b'\x93NUMPY\x01\x00', 'Only actual NPY v1 input is expected'
    nhead = struct.unpack('<H', data[8:10])[0]
    header = ast.literal_eval(data[10:10+nhead].decode('latin1').strip())
    assert header == {'descr': '<f4', 'fortran_order': False, 'shape': shape}, header
    assert sys.byteorder == 'little'
    values = array.array('f')
    values.frombytes(data[10+nhead:])
    assert len(values) == math.prod(shape)
    check('finite:' + str(path), all(math.isfinite(v) for v in values))
    return values


F32_MIN = struct.unpack('<f', struct.pack('<f', 1e-6))[0]


def box_iou(pred, truth):
    """AABB overlap via interval intersection, evaluated with double precision."""
    sizes = [max(F32_MIN, pred[d+3]) for d in range(3)]
    overlap = []
    for d in range(3):
        lo = max(pred[d] - sizes[d] / 2, truth[d] - truth[d+3] / 2)
        hi = min(pred[d] + sizes[d] / 2, truth[d] + truth[d+3] / 2)
        overlap.append(max(0.0, hi - lo))
    intersection = math.prod(overlap)
    union = math.prod(sizes) + math.prod(truth[3:]) - intersection
    assert union > 0 and math.isfinite(union)
    answer = intersection / union
    assert math.isfinite(answer) and 0 <= answer <= 1 + 1e-12
    return answer


def paired(a, b, key):
    result = {}
    for th in (.25, .5):
        table = collections.Counter((x[key] > th, y[key] > th) for x, y in zip(a, b))
        result[str(th)] = {
            'both_hit': table[(True, True)], 'neither_hit': table[(False, False)],
            'gain': table[(False, True)], 'loss': table[(True, False)],
            'net': table[(False, True)] - table[(True, False)],
        }
    return result


def method_tree(path, class_name, method):
    tree = ast.parse(read(path).decode('utf-8'))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name)
    node = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == method)
    return ast.dump(node, include_attributes=False)


metadata_names = [
    'FORMAL_ACTUAL_AUDIT_REQUEST.txt', 'preflight_r3/evaluate_parent_batches.py',
    'preflight_r3/spec.json', 'preflight_r3/actual_env_spec.json',
    'analyze_parent_batch_results.py', 'input_manifest.json',
    'actual_joint_det_dataset.py', 'actual_visual_data_handlers.py',
    'actual_pv_utils.py', 'actual_prepare_data.py', 'parent_pv_ground.py',
    'parent_source_port.json', 'FINDINGS.json', 'INFERENCE_PROTOCOL_FACTS.json',
    'PREFLIGHT_AND_FORMAL_NOTES.md', 'preflight_and_formal_publication.json',
    'parser_mappings_actual/INTAKE.json', 'official_entry_sources/INTAKE.json',
    'official_entry_sources/train_dist_mod.py', 'official_entry_sources/main_utils.py',
    'official_entry_sources/models/pv_ground.py', 'official_entry_sources/scripts/test_scanrefer.sh',
]
for name in metadata_names:
    read(ROOT / name)
for name in ['author_grounding_evaluator.py', 'PV-Ground_ScanRefer.txt']:
    read(AUTHOR / name)
for name in ['models/losses.py', 'models/encoder_decoder_layers.py']:
    read(SOURCE / name)
for name in ['pvground_source_query.py', 'pvground_observation_query.py', 'pvground_task_observation_query.py']:
    read(HIST / name)

spec = js(ROOT / 'preflight_r3/spec.json')
manifest = js(ROOT / 'input_manifest.json')
port = js(ROOT / 'parent_source_port.json')
env = js(ROOT / 'preflight_r3/actual_env_spec.json')
summary = js(ROOT / 'formal_analysis/SUMMARY.json')
intake = js(ACTUAL / 'INTAKE.json')
formal = js(ACTUAL / 'formal_receipt.json')
imports = js(ACTUAL / 'formal_imports.json')
findings = js(ROOT / 'FINDINGS.json')
historic_receipt = js(HIST / 'published_parent/receipt.json')
digest_matches(ROOT / 'preflight_r3/evaluate_parent_batches.py', spec['runner_sha256'])
digest_matches(ROOT / 'input_manifest.json', spec['input_manifest_sha256'])
digest_matches(ROOT / 'actual_joint_det_dataset.py', spec['dataset_source_sha256'])
digest_matches(ROOT / 'actual_visual_data_handlers.py', spec['scan_helper_sha256'])
digest_matches(ROOT / 'parent_source_port.json', spec['source_port_sha256'])
digest_matches(AUTHOR / 'author_grounding_evaluator.py', spec['evaluator_sha256'])
digest_matches(AUTHOR / 'PV-Ground_ScanRefer.txt', findings['author_log_sha256'])
digest_matches(HIST / 'published_parent/receipt.json', findings['local_receipt_sha256'])
digest_matches(ROOT / 'parent_pv_ground.py', port['files']['models/pv_ground.py'])
digest_matches(ROOT / 'actual_pv_utils.py', port['files']['models/pv_utils.py'])
digest_matches(ROOT / 'actual_prepare_data.py', port['files']['prepare_data.py'])
for name in ['models/losses.py', 'models/encoder_decoder_layers.py']:
    digest_matches(SOURCE / name, port['files'][name])
for name, value in spec['reader_module_hashes'].items():
    digest_matches(HIST / name, value)
check('canonical_environment_spec_hash', hashlib.sha256(json.dumps(env, sort_keys=True, separators=(',', ':')).encode()).hexdigest() == spec['env_spec_sha256'])
check('declared_checkpoint_hash_cross_reference', env['weight_dirs']['scanrefer']['sha256'] == spec['checkpoint_sha256'], 'The checkpoint bytes were not available/read in this local audit.')
for entry in intake['files']:
    digest_matches(ACTUAL / entry['name'], entry['sha256'], entry['bytes'])
check('formal_completion', intake['exitcode'] == 0 and not intake['process_live'] and read(ACTUAL / 'formal.exit').strip() == b'0' and formal['status'] == 'COMPLETED')
check('formal_not_training', formal['formal_accuracy_result'] and formal['training_runs'] == 0 and formal['new_weights'] == 0)
for module_name, local_name in [
    ('src.joint_det_dataset', 'actual_joint_det_dataset.py'),
    ('src.visual_data_handlers', 'actual_visual_data_handlers.py'),
    ('sng_parser', 'actual_parser/__init__.py'),
    ('models.pv_ground', 'parent_pv_ground.py'),
    ('main_utils', 'official_entry_sources/main_utils.py'),
    ('prepare_data', 'actual_prepare_data.py'),
]:
    digest_matches(ROOT / local_name, imports[module_name]['sha256'])
mapping_intake = js(ROOT / 'parser_mappings_actual/INTAKE.json')
mapping_values = []
for entry in mapping_intake['files']:
    path = ROOT / 'parser_mappings_actual' / (entry['name'] + '.json')
    digest_matches(path, entry['sha256'], entry['bytes'])
    value = js(path)
    check('mapping_entries:' + entry['name'], len(value) == entry['entries'])
    mapping_values.append(read(path))
check('mapping_bytes_identical', len(set(mapping_values)) == 1)
check('mapping_port_pin', hashlib.sha256(mapping_values[0]).hexdigest() == port['files']['mapping_full2rio27.json'])
for entry in js(ROOT / 'official_entry_sources/INTAKE.json'):
    digest_matches(ROOT / 'official_entry_sources' / entry['name'], entry['sha256'], entry['bytes'])
for class_name, method in [('PVGround', '_generate_queries'), ('GumbelSampling', 'forward')]:
    check('author_actual_AST:' + class_name + '.' + method,
          method_tree(ROOT / 'parent_pv_ground.py', class_name, method) == method_tree(ROOT / 'official_entry_sources/models/pv_ground.py', class_name, method))

all_rows = {}
all_arrays = {}
all_results = {}
selected_csv = OUT / 'AUDIT_selected_box_recount.jsonl'
with selected_csv.open('w', encoding='utf-8', newline='\n') as selected_out:
    for batch in (8, 24):
        folder = ACTUAL / 'formal' / ('batch' + str(batch))
        rows = lines(folder / 'rows.jsonl')
        receipt = js(folder / 'receipt.json')
        check('contiguous_rows:' + str(batch), len(rows) == 9508 and [r['row_id'] for r in rows] == list(range(9508)))
        check('receipt_embedded:' + str(batch), receipt == formal['receipts'][(8, 24).index(batch)])
        check('spec_hash_receipt:' + str(batch), receipt['spec_sha256'] == READS[str((ROOT/'preflight_r3/spec.json').resolve())]['sha256'])
        check('receipt_state_claims:' + str(batch), receipt['weight_state_unchanged'] and receipt['model_state_tensors'] == 1234 and not receipt['observation_reader_installed'] and receipt['state_free_observation_groupers'] == 10 and receipt['optimizer_updates'] == 0 and receipt['new_weights'] == 0)
        for name, entry in receipt['files'].items():
            digest_matches(folder / name, entry['sha256'], entry['bytes'])
        boxes = npy_f32(folder / 'boxes.npy', (9508, 256, 6))
        scores = npy_f32(folder / 'scores.npy', (9508, 2, 256))
        candidate_hash = hashlib.sha256()
        oracle = [0, 0]
        candidate_hits = [0, 0]
        candidate_min, candidate_max = 1.0, 0.0
        clamped_sizes = 0
        box_size_valid = True
        modes = {mode: {'rec_hits25': 0, 'rec_hits50': 0, 'mask_hits25': 0, 'mask_hits50': 0,
                        'max_saved_iou_abs_difference': 0.0, 'max_selected_box_abs_difference': 0.0,
                        'threshold_disagreements': 0, 'chosen_not_maximum': 0,
                        'score_max_tie_rows': 0, 'mask_outside_unit_interval': 0,
                        'minimum_selected_threshold_distance': 1.0}
                 for mode in ('bbs', 'bbf')}
        for i, row in enumerate(rows):
            truth = row['root_box']
            box_size_valid = box_size_valid and len(truth) == 6 and all(math.isfinite(v) for v in truth) and all(v > 0 for v in truth[3:])
            ious = []
            for q in range(256):
                p = (i * 256 + q) * 6
                pred = boxes[p:p+6]
                clamped_sizes += sum(v < F32_MIN for v in pred[3:])
                val = box_iou(pred, truth)
                ious.append(val)
            candidate_hash.update(struct.pack('<256d', *ious))
            candidate_min = min(candidate_min, min(ious))
            candidate_max = max(candidate_max, max(ious))
            for t, threshold in enumerate((.25, .5)):
                candidate_hits[t] += sum(v > threshold for v in ious)
                oracle[t] += int(max(ious) > threshold)
            evidence = {'row_id': i, 'source_line': i + 1, 'batch': batch}
            for m, mode in enumerate(('bbs', 'bbf')):
                saved = row[mode]
                q = saved['query']
                assert isinstance(q, int) and 0 <= q < 256
                val = ious[q]
                start = (i * 256 + q) * 6
                pred = list(boxes[start:start+6])
                pred[3:] = [max(v, F32_MIN) for v in pred[3:]]
                score_vector = scores[(i * 2 + m) * 256:(i * 2 + m + 1) * 256]
                maximum = max(score_vector)
                result = modes[mode]
                result['max_saved_iou_abs_difference'] = max(result['max_saved_iou_abs_difference'], abs(val - saved['iou']))
                result['max_selected_box_abs_difference'] = max(result['max_selected_box_abs_difference'], max(abs(a-b) for a, b in zip(pred, saved['box'])))
                result['chosen_not_maximum'] += int(score_vector[q] != maximum)
                result['score_max_tie_rows'] += int(sum(v == maximum for v in score_vector) > 1)
                result['mask_outside_unit_interval'] += int(not (math.isfinite(saved['mask_iou']) and 0 <= saved['mask_iou'] <= 1))
                for threshold, suffix in ((.25, '25'), (.5, '50')):
                    result['rec_hits'+suffix] += int(val > threshold)
                    result['mask_hits'+suffix] += int(saved['mask_iou'] > threshold)
                    result['threshold_disagreements'] += int((val > threshold) != (saved['iou'] > threshold))
                    result['minimum_selected_threshold_distance'] = min(result['minimum_selected_threshold_distance'], abs(val-threshold))
                evidence[mode] = {'query': q, 'recomputed_iou_f64': val, 'saved_iou_f32': saved['iou'], 'chosen_score': score_vector[q], 'maximum_score': maximum}
            selected_out.write(json.dumps(evidence, allow_nan=False) + '\n')
        check('root_box_finite_positive_sizes:' + str(batch), box_size_valid)
        for mode in ('bbs', 'bbf'):
            result = modes[mode]
            counts = {k: result[k] for k in receipt['metrics'][mode]}
            check('native_counts:' + str(batch) + ':' + mode, counts == receipt['metrics'][mode], counts)
            check('selected_geometry:' + str(batch) + ':' + mode, result['max_saved_iou_abs_difference'] < 1e-5 and result['max_selected_box_abs_difference'] == 0 and result['threshold_disagreements'] == 0)
            check('native_top1:' + str(batch) + ':' + mode, result['chosen_not_maximum'] == 0)
            check('saved_mask_iou_finite_unit_interval:' + str(batch) + ':' + mode, result['mask_outside_unit_interval'] == 0)
            claimed = summary['batches'][str(batch)]['modes'][mode]
            hits = [result['rec_hits25'], result['rec_hits50']]
            mask_hits = [result['mask_hits25'], result['mask_hits50']]
            result['percent'] = [100 * x / len(rows) for x in hits]
            result['mask_percent_from_saved_iou'] = [100 * x / len(rows) for x in mask_hits]
            check('summary_counts_percent:' + str(batch) + ':' + mode,
                  claimed['hits'] == hits and claimed['mask_hits_from_saved_iou'] == mask_hits and all(abs(a-b) < 1e-12 for a,b in zip(claimed['percent'], result['percent'])))
        check('oracle_summary:' + str(batch), oracle == summary['batches'][str(batch)]['full256_oracle_hits'])
        all_results[str(batch)] = {
            'rows': len(rows), 'unique_scenes': len({r['scan_id'] for r in rows}),
            'unique_scene_target_pairs': len({(r['scan_id'], r['target_id']) for r in rows}),
            'candidate_box_iou_computations': len(rows) * 256,
            'candidate_iou_f64_little_endian_sha256': candidate_hash.hexdigest(),
            'candidate_iou_range': [candidate_min, candidate_max],
            'candidate_hits': candidate_hits, 'oracle_hits': oracle,
            'clamped_size_components': clamped_sizes, 'modes': modes,
            'scope': 'IoUs recomputed from saved predicted boxes and saved dataset root boxes; Mask hits use saved Mask IoUs.'
        }
        all_rows[batch] = rows
        all_arrays[batch] = (boxes, scores)
        print(json.dumps({'completed_batch': batch, 'rows': len(rows), 'candidate_ious': len(rows)*256, 'seconds': time.perf_counter()-START}), flush=True)

identity_keys = ('row_id', 'scan_id', 'target_id', 'utterance', 'root_box', 'point_sha256', 'detector_sha256', 'native_maps_sha256')
identity = {k: sum(a[k] != b[k] for a,b in zip(all_rows[8], all_rows[24])) for k in identity_keys}
check('batch_input_record_equality', not any(identity.values()) and identity == summary['input_identity_differences'], identity)
batch_pairs = {}
for mode in ('bbs', 'bbf'):
    a = [r[mode] for r in all_rows[8]]
    b = [r[mode] for r in all_rows[24]]
    batch_pairs[mode] = {'box': paired(a,b,'iou'), 'mask_from_saved_iou': paired(a,b,'mask_iou'),
                         'same_query_slot_count': sum(x['query'] == y['query'] for x,y in zip(a,b)),
                         'identical_selected_box_rows': sum(x['box'] == y['box'] for x,y in zip(a,b))}
    for th, name in ((.25, '025'), (.5, '050')):
        p = batch_pairs[mode]['box'][str(th)]
        claimed = summary['batch24_vs_batch8'][mode][name]
        check('paired_summary:' + mode + ':' + name, claimed == {'batch24_gain': p['gain'], 'batch24_loss': p['loss'], 'net': p['net']})
    check('query_slot_summary:' + mode, batch_pairs[mode]['same_query_slot_count'] == summary['same_query_slot_counts'][mode])
array_difference = {}
for ix, name in enumerate(('boxes', 'scores')):
    a,b = all_arrays[8][ix],all_arrays[24][ix]
    array_difference[name] = {'max_absolute_difference_f64': max(abs(x-y) for x,y in zip(a,b)),
                              'different_elements': sum(x != y for x,y in zip(a,b))}
    claimed_key = 'all256_box_max_absolute_batch_difference' if name == 'boxes' else 'all_native_score_max_absolute_batch_difference'
    check('array_difference_summary:' + name, abs(array_difference[name]['max_absolute_difference_f64'] - summary[claimed_key]) < 1e-6)

history = lines(ROOT / 'historical_parent_rows.jsonl')
check('history_rows_order', len(history) == 9508 and [r['row_id'] for r in history] == list(range(9508)))
digest_matches(ROOT / 'historical_parent_rows.jsonl', historic_receipt['rows_sha256'])
historic_identity = {k: sum(a[k] != b[k] for a,b in zip(history, all_rows[8])) for k in ('row_id','scan_id','target_id','root_box','point_sha256')}
check('history_input_record_equality', not any(historic_identity.values()) and historic_identity == summary['historical_batch8_inputs'])
historic_comparison = {}
for mode in ('bbs', 'bbf'):
    a = [r[mode] for r in history]
    b = [r[mode] for r in all_rows[8]]
    counts = {'rec_hits25': sum(r['iou']>.25 for r in a), 'rec_hits50': sum(r['iou']>.5 for r in a),
              'mask_hits25': sum(r['mask_iou']>.25 for r in a), 'mask_hits50': sum(r['mask_iou']>.5 for r in a)}
    check('historical_receipt_counts:' + mode, all(historic_receipt['metrics'][mode][k] == v for k,v in counts.items()))
    historic_comparison[mode] = {'counts': counts, 'box_pairs': paired(a,b,'iou'),
                                'mask_pairs_from_saved_iou': paired(a,b,'mask_iou'),
                                'different_mode_records': sum(x != y for x,y in zip(a,b))}
    check('historical_summary:' + mode, summary['historical_batch8_comparison'][mode]['old_hits'] == [counts['rec_hits25'], counts['rec_hits50']])
for name in ('boxes', 'scores'):
    current_sha = READS[str((ACTUAL/'formal/batch8'/(name+'.npy')).resolve())]['sha256']
    check('historic_receipt_digest_matches_current_batch8:' + name, current_sha == historic_receipt[name+'_sha256'], 'Historical array bytes themselves were not read; this compares current bytes to the historical receipt.')

author_text = read(AUTHOR / 'PV-Ground_ScanRefer.txt').decode('utf-8')
author_rates = {}
for label, mode in [('position', 'bbs'), ('semantic','bbf')]:
    values = [float(re.search(r'last_ '+label+r' alignment Acc'+re.escape(t)+r': Top-1: ([0-9.]+)', author_text).group(1))*100 for t in ('0.25','0.50')]
    author_rates[mode] = values
author_gaps = {str(batch): {mode: [a-b for a,b in zip(author_rates[mode], all_results[str(batch)]['modes'][mode]['percent'])] for mode in ('bbs','bbf')} for batch in (8,24)}
check('author_log_rates_in_findings', author_rates['bbs'] == findings['author_log_position_percent'] and author_rates['bbf'] == findings['author_log_semantic_percent'])
check('total_candidate_count_summary', sum(r['candidate_box_iou_computations'] for r in all_results.values()) == summary['candidate_box_iou_recomputations'])

unchanged = True
for path, before in READS.items():
    data = Path(path).read_bytes()
    unchanged &= hashlib.sha256(data).hexdigest() == before['sha256'] and len(data) == before['bytes']
check('all_audited_inputs_unchanged_during_verification', unchanged)
evidence = {
    'status': 'PASS' if all(c['pass'] for c in CHECKS) else 'FAIL',
    'generated_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'runtime': {'executable': sys.executable, 'python': sys.version, 'dependencies': 'standard library only', 'new_model_forwards': 0, 'new_gpu_jobs': 0, 'remote_calls': 0},
    'arithmetic': 'Independent scalar double-precision AABB IoU from float32 saved boxes; size floor equals float32(1e-6); tolerance only for saved numerical IoU, exact threshold agreement required.',
    'checks_executed': len(CHECKS), 'checks': CHECKS,
    'audited_input_hashes': READS, 'batches': all_results,
    'batch_input_record_differences': identity, 'batch24_vs_batch8': batch_pairs,
    'array_differences': array_difference, 'historical_input_record_differences': historic_identity,
    'historical_batch8_comparison': historic_comparison,
    'author_log_percent': author_rates, 'author_log_minus_current_percentage_points': author_gaps,
    'superpoint_manifest_val_files': len(manifest['superpoint_files']['val']),
    'scope_limits': [
        'No reconstruction of raw GT or scan caches, detector inputs, native maps, or superpoints; stored input digest values are compared, not independently recomputed from absent underlying tensors.',
        'No raw masks/logits/pointwise GT masks are saved here; Mask validation is a recount of recorded IoUs.',
        'No independent checkpoint byte hash or 1234-tensor comparison; state claims are supported by inspected source assertions and completion receipts.',
        'The historical selected rows were directly read and hashed. Historical array comparison uses their published receipt digests.',
        'Author log is a rounded aggregate record without sampled inputs/RNG snapshot; this audit does not establish author-environment equivalence or isolated causality.'
    ],
    'seconds': time.perf_counter()-START,
}
(OUT / 'AUDIT_evidence.json').write_text(json.dumps(evidence, indent=2, ensure_ascii=False, allow_nan=False)+'\n', encoding='utf-8')
(OUT / 'AUDIT_input_hashes.json').write_text(json.dumps(READS, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')
print(json.dumps({'status': evidence['status'], 'checks': len(CHECKS), 'failed_checks': [c for c in CHECKS if not c['pass']], 'files_hashed': len(READS), 'seconds': evidence['seconds']}, ensure_ascii=False), flush=True)
