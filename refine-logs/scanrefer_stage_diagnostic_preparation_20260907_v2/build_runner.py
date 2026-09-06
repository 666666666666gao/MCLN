import hashlib
from pathlib import Path

repo = Path('C:/Users/gb/.codex_mcln_g0_20260905')
base = (repo / 'scripts/evaluate_scanrefer_local_visual_official.py').read_bytes()
assert hashlib.sha256(base).hexdigest() == '1e5e1e30e8efba15a2696e612d677c4f18e4c587c9076d59b3f0be3bc96cc935'
text = base.decode()
header = text[text.index('import argparse'):text.index('def file_sha')]
header = '"""Isolated stage diagnostic; never replaces the fixed formal evaluation.\n\nDerived from the existing formal evaluator with the same forward and selection paths.\nA completed, independently audited formal run must be supplied as its reference.\n"""\n\n' + header
header += ('sys.path.insert(0, str(Path(__file__).resolve().parents[1]))\n\n'
           'from scripts.evaluate_scanrefer_local_visual_official import (\n'
           '    file_sha, write_json, row_metrics, verify_training_endpoint,\n)\n\n\n')
text = header + text[text.index('def main():'):]

def replace(old, new):
    global text
    assert text.count(old) == 1, old[:120]
    text = text.replace(old, new)

replace("assert manifest['schema'] == 'mcln-scanrefer-local-visual-official-input-v2'", """assert manifest['schema'] == 'mcln-scanrefer-stage-diagnostic-input-v1'
    reference_directory = Path(manifest['reference_formal_directory'])
    reference_names = {'input_manifest.json', 'controller.exit', 'result/receipt.json',
                       'result/rows.json', 'result/native_rows.json',
                       'result/protocol.json', 'result/independent_audit.json'}
    assert set(manifest['reference_files']) == reference_names
    for name, digest in manifest['reference_files'].items():
        assert file_sha(reference_directory / name) == digest, name
    assert (reference_directory / 'controller.exit').read_text().strip() == '0'
    reference_receipt = json.loads((reference_directory / 'result/receipt.json').read_text())
    reference_audit = json.loads((reference_directory / 'result/independent_audit.json').read_text())
    reference_input = json.loads((reference_directory / 'input_manifest.json').read_text())
    assert reference_receipt['status'] == 'complete' and reference_receipt['formal_rows'] == 9508
    assert reference_audit['integrity_pass']
    assert reference_audit['receipt_sha256'] == manifest['reference_files']['result/receipt.json']
    for name in ('training_directory', 'training_receipt_sha256', 'data_root', 'val_superpoint_files'):
        assert manifest[name] == reference_input[name], name
    reference_native = json.loads((reference_directory / 'result/native_rows.json').read_text())
    reference_final = json.loads((reference_directory / 'result/rows.json').read_text())
    reference_protocol = json.loads((reference_directory / 'result/protocol.json').read_text())""")
replace("result_directory = directory / 'result'", "result_directory = directory / 'diagnostic_result'")
replace("scripts.__path__ = [str(directory / 'scripts')] + list(scripts.__path__)",
        "scripts.__path__ = [str(directory / 'scripts'), str(source / 'scripts')]")
replace('    from scripts.scanrefer_rec_evaluation import rec_evaluation_view', '''    from scripts.scanrefer_rec_evaluation import rec_evaluation_view
    from scripts.trace_scanrefer_readout_stages import STAGES, trace_readout_stages
    from scripts.scanrefer_stage_diagnostics import FeatureMoments, summarize_stages''')
replace("    write_json(result_directory / 'protocol.json',", "    assert [list(value) for value in identity] == reference_protocol['identities']\n    write_json(result_directory / 'protocol.json',")
replace("'formal_checkpoint_arm': 'local'", "'checkpoint_arm': 'local', 'formal_rows': 0, 'purpose': 'stage_diagnostic'")
replace("    states = {'protected_v99': protected_state, 'local_v99': trained['model']}", '''    states = {'protected_v99': protected_state, 'local_v99': trained['model']}
    trace_records = {arm: [] for arm in models}
    feature_moments = {arm: {'parent': FeatureMoments(152), 'geometry': FeatureMoments(179)}
                       for arm in models}
    feature_hooks = [readouts[arm].scorers[name].register_forward_pre_hook(moment)
                     for arm, moments in feature_moments.items() for name, moment in moments.items()]''')
replace('                del native_outputs, native_boxes, native_ious', '                del native_outputs, native_ious')
replace("                runtime = readout['runtime']", '''                trace = trace_readout_stages(native_boxes,
                    torch.tensor(native_selected, dtype=torch.long, device=native_boxes.device),
                    readout, readouts[arm].metadata)
                # Root GT is used only after the GT-free stage selections are fixed.
                stage_ious = compute_query_ious(trace['boxes'], roots, root_valid)
                candidate = readout['parent']['candidate_batch']
                top16_ious = compute_query_ious(candidate['boxes'], roots, root_valid)
                top16_oracle = top16_ious.masked_fill(~candidate['valid_mask'], -1.).max(dim=1).values
                for offset in range(len(roots)):
                    row_id = len(trace_records[arm])
                    stages = {name: {'query_index': int(trace['query_indices'][offset, stage]),
                                    'variant_index': int(trace['variant_indices'][offset, stage]),
                                    'box': trace['boxes'][offset, stage].cpu().tolist(),
                                    'rec_iou': float(stage_ious[offset, stage])}
                              for stage, name in enumerate(STAGES)}
                    trace_records[arm].append({
                        'row_id': row_id, 'scan_id': raw['scan_ids'][offset],
                        'target_id': identity[row_id][1], 'utterance': identity[row_id][2],
                        'point_sha256': points[offset], 'root_box': roots[offset, 0].cpu().tolist(),
                        'stages': stages, 'top16_oracle_iou': float(top16_oracle[offset]),
                        'top16_query_indices': trace['top16_query_indices'][offset].cpu().tolist(),
                        'top16_valid': trace['top16_valid'][offset].cpu().tolist(),
                        'top16_boxes': candidate['boxes'][offset].detach().cpu().tolist(),
                        'effective_variant_valid': trace['effective_variant_valid'][offset].cpu().tolist(),
                        'deployed_variant_valid': trace['deployed_variant_valid'][offset].cpu().tolist(),
                        'geometry_flat_index': int(trace['geometry_flat_indices'][offset]),
                        'proposal_flat_index': int(trace['proposal_flat_indices'][offset]),
                        'final_flat_index': int(trace['final_flat_indices'][offset]),
                        'pareto_pass': bool(trace['pareto_pass'][offset]),
                        'predicted_head_gain': trace['predicted_head_gain'][offset].cpu().tolist(),
                        'predicted_aggregate_gain': float(trace['predicted_aggregate_gain'][offset])})
                del native_boxes, trace, stage_ious, top16_ious, top16_oracle
                runtime = readout['runtime']''')
replace("    metrics = {arm: row_metrics(rows) for arm, rows in records.items()}", '''    for handle in feature_hooks:
        handle.remove()
    metrics = {arm: row_metrics(rows) for arm, rows in records.items()}''')
replace("    result = {'schema': 'mcln-scanrefer-local-visual-official-v2', 'status': 'complete',", '''    stage_summary = summarize_stages(trace_records, reference_native, reference_final)
    for arm in models:
        for suffix in ('025', '050'):
            assert stage_summary['arms'][arm]['metrics']['native']['hits' + suffix] == native_metrics[arm]['rec_hits' + suffix]
            assert stage_summary['arms'][arm]['metrics']['v99_final']['hits' + suffix] == metrics[arm]['rec_hits' + suffix]
    write_json(result_directory / 'stage_rows.json', trace_records)
    write_json(result_directory / 'stage_summary.json', stage_summary)
    write_json(result_directory / 'normalized_features.json', {
        arm: {name: moment.export() for name, moment in moments.items()}
        for arm, moments in feature_moments.items()})
    result = {'schema': 'mcln-scanrefer-stage-diagnostic-result-v1', 'status': 'complete',''')
replace("'formal_rows': 9508, 'optimizer_steps': 0, 'checkpoint_writes': 0,", "'formal_rows': 0, 'diagnostic_rows': 9508, 'optimizer_steps': 0, 'checkpoint_writes': 0,")
replace("        'promotion': promotion_check(metrics['protected_v99'], metrics['local_v99']),", '''        'used_for_promotion': False, 'selection_uses_gt': False,
        'all_forward_final_scores_verified_equal': True,
        'reference_formal_directory': str(reference_directory),
        'reference_files': manifest['reference_files'],
        'stage_rows_sha256': file_sha(result_directory / 'stage_rows.json'),
        'stage_summary_sha256': file_sha(result_directory / 'stage_summary.json'),
        'normalized_features_sha256': file_sha(result_directory / 'normalized_features.json'),''')
text = text.replace('SCANREFER LOCAL VISUAL OFFICIAL', 'SCANREFER STAGE DIAGNOSTIC')
text = text.replace("logging.getLogger('scanrefer-local-visual-official')", "logging.getLogger('scanrefer-stage-diagnostic')")
compile(text, 'diagnose_scanrefer_readout_stages.py', 'exec')
(repo / 'scripts/diagnose_scanrefer_readout_stages.py').write_text(text, encoding='utf-8', newline='\n')
print('Created isolated diagnostic runner from the frozen formal evaluator; no job launched.')
