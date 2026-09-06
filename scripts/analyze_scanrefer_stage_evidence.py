"""Describe saved stage changes without fitting, threshold search, or GPU inference."""

import argparse
import ast
import json
from pathlib import Path

import numpy as np

from audit_scanrefer_stage_diagnostic import ARMS, STAGES, box_iou, changes, hits, read, sha


def feature_names(source):
    tree = ast.parse((source / 'models/rec_candidate_adapter.py').read_bytes())
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == '_feature_names')
    namespace = {}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[])), '<feature names>', 'exec'), namespace)
    parent = namespace['_feature_names'](64)
    tree = ast.parse((source / 'models/rec_mask_geometry.py').read_bytes())
    assignment = next(node for node in tree.body if isinstance(node, ast.Assign)
                      and any(isinstance(target, ast.Name) and target.id == 'REC_MASK_GEOMETRY_FEATURE_NAMES' for target in node.targets))
    geometry = parent + list(ast.literal_eval(assignment.value)) + ['parent_score', 'parent_is_deployed_top1']
    assert len(parent) == 152 and len(geometry) == 179
    return {'parent': parent, 'geometry': geometry}


def analyze(directory, source):
    result = directory / 'diagnostic_result'
    assert read(result / 'independent_audit.json')['integrity_pass']
    rows, moments = read(result / 'stage_rows.json'), read(result / 'normalized_features.json')
    output = {'arms': {}, 'normalized_feature_distribution_changes': {},
              'formal_rows': 0, 'diagnostic_rows': 9508, 'used_for_promotion': False,
              'stage_rows_sha256': sha(result / 'stage_rows.json'),
              'audit_sha256': sha(result / 'independent_audit.json'), 'analyzer_sha256': sha(Path(__file__)),
              'interpretation': 'Post-hoc accounting on each actual selected query. Raw-box substitution is diagnostic, not a proposed deployment. Different valid populations and dynamic slots preclude causal feature attribution.'}
    for arm in ARMS:
        items = rows[arm]
        all_iou = box_iou([row['top16_boxes'] for row in items], [[row['root_box']] for row in items])
        parents = [row['stages']['parent_after_geometry_validity']['rec_iou'] for row in items]
        stages = {}
        for stage in STAGES[3:]:
            selected_raw = []
            for index, row in enumerate(items):
                query = row['stages'][stage]['query_index']
                positions = [slot for slot, value in enumerate(row['top16_query_indices']) if value == query and row['top16_valid'][slot]]
                assert len(positions) == 1
                selected_raw.append(float(all_iou[index, positions[0]]))
            selected_final = [row['stages'][stage]['rec_iou'] for row in items]
            stages[stage] = {
                'raw_box_of_selected_query_hits': hits(selected_raw),
                'actual_selected_variant_hits': hits(selected_final),
                'parent_to_selected_raw_query_box': changes(parents, selected_raw),
                'same_selected_query_raw_to_variant_box': changes(selected_raw, selected_final),
                'selected_variant_counts': {str(variant): sum(row['stages'][stage]['variant_index'] == variant for row in items) for variant in range(7)}}
        output['arms'][arm] = stages
    names = feature_names(source)
    output['feature_schema_source_sha256'] = {name: sha(source / name) for name in
        ['models/rec_candidate_adapter.py', 'models/rec_mask_geometry.py']}
    for scorer in ('parent', 'geometry'):
        before, after = moments[ARMS[0]][scorer], moments[ARMS[1]][scorer]
        mean_delta = np.asarray(after['mean']) - np.asarray(before['mean'])
        rms_delta = np.asarray(after['root_mean_square']) - np.asarray(before['root_mean_square'])
        order = np.argsort(-np.abs(mean_delta), kind='stable')[:10]
        output['normalized_feature_distribution_changes'][scorer] = {
            'valid_candidates': [before['valid_candidates'], after['valid_candidates']],
            'rms_of_mean_difference': float(np.sqrt(np.mean(mean_delta**2))),
            'largest_mean_changes': [{'index': int(index), 'name': names[scorer][index],
                'protected_mean': before['mean'][index], 'local_mean': after['mean'][index],
                'mean_difference': float(mean_delta[index]), 'rms_difference': float(rms_delta[index])} for index in order]}
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--source', type=Path, required=True)
    args = parser.parse_args()
    value = analyze(args.directory, args.source)
    with (args.directory / 'diagnostic_result/evidence_breakdown.json').open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write('\n')
    print(json.dumps(value))
