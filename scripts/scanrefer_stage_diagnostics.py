"""Post-forward measurements for the isolated ScanRefer stage diagnostic."""

import math

import numpy as np

from scripts.trace_scanrefer_readout_stages import STAGES


class FeatureMoments:
    """Measure the actual normalized scorer input on valid candidates only."""

    def __init__(self, width):
        self.width = width
        self.count = 0
        self.total = np.zeros(width, dtype=np.float64)
        self.squared_total = np.zeros(width, dtype=np.float64)
        self.minimum = np.full(width, np.inf)
        self.maximum = np.full(width, -np.inf)

    def __call__(self, module, arguments):
        features, valid = arguments
        assert features.shape[-1] == self.width
        values = features.detach()[valid].cpu().numpy().astype(np.float64)
        assert len(values) > 0 and np.isfinite(values).all()
        self.count += len(values)
        self.total += values.sum(axis=0)
        self.squared_total += np.square(values).sum(axis=0)
        self.minimum = np.minimum(self.minimum, values.min(axis=0))
        self.maximum = np.maximum(self.maximum, values.max(axis=0))

    def export(self):
        assert self.count > 0
        return {'valid_candidates': self.count, 'feature_width': self.width,
                'mean': (self.total / self.count).tolist(),
                'root_mean_square': np.sqrt(self.squared_total / self.count).tolist(),
                'minimum': self.minimum.tolist(), 'maximum': self.maximum.tolist(),
                'population': 'Each arm uses its own valid candidates; not a causal attribution.'}


def transitions(before, after, threshold):
    assert len(before) == len(after)
    repaired = sum(a <= threshold < b for a, b in zip(before, after))
    damaged = sum(b <= threshold < a for a, b in zip(before, after))
    return {'repairs': repaired, 'damage': damaged, 'net_hits': repaired - damaged}


def summarize_stages(records, reference_native, reference_final):
    """Compare stage outcomes; query slots are never treated as instance labels."""
    arms = ('protected_v99', 'local_v99')
    count = len(records[arms[0]])
    assert count > 0
    summary = {'rows_per_arm': count, 'arms': {}, 'paired_stage_changes': {},
               'reference_agreement': {}, 'v99_proposal_is_ungated': True,
               'query_slot_changes_do_not_prove_instance_changes': True}
    for arm in arms:
        rows = records[arm]
        assert len(rows) == len(reference_native[arm]) == len(reference_final[arm]) == count
        for index, (row, native, final) in enumerate(zip(rows, reference_native[arm], reference_final[arm])):
            assert row['row_id'] == native['row_id'] == final['row_id'] == index
            for key in ('scan_id', 'point_sha256'):
                assert row[key] == native[key] == final[key]
            assert set(row['stages']) == set(STAGES)
            assert all(math.isfinite(stage['rec_iou']) and 0 <= stage['rec_iou'] <= 1
                       for stage in row['stages'].values())
        arm_summary = {'metrics': {}, 'consecutive_transitions': {}}
        for name in STAGES:
            values = [row['stages'][name]['rec_iou'] for row in rows]
            arm_summary['metrics'][name] = {
                'hits025': sum(value > .25 for value in values),
                'hits050': sum(value > .50 for value in values)}
        for earlier, later in zip(STAGES[:-1], STAGES[1:]):
            before = [row['stages'][earlier]['rec_iou'] for row in rows]
            after = [row['stages'][later]['rec_iou'] for row in rows]
            change = {suffix: transitions(before, after, threshold)
                      for threshold, suffix in ((.25, '025'), (.50, '050'))}
            change['query_slot_changed'] = sum(
                row['stages'][earlier]['query_index'] != row['stages'][later]['query_index']
                for row in rows)
            change['same_query_slot_different_box'] = sum(
                row['stages'][earlier]['query_index'] == row['stages'][later]['query_index']
                and row['stages'][earlier]['box'] != row['stages'][later]['box'] for row in rows)
            arm_summary['consecutive_transitions'][earlier + '->' + later] = change
        summary['arms'][arm] = arm_summary
        agreement = {}
        for stage, reference in (('native', reference_native[arm]), ('v99_final', reference_final[arm])):
            values = [row['stages'][stage]['rec_iou'] for row in rows]
            old_values = [row['rec_iou'] for row in reference]
            agreement[stage] = {
                'max_absolute_iou_difference': max(abs(a - b) for a, b in zip(old_values, values)),
                '025': transitions(old_values, values, .25),
                '050': transitions(old_values, values, .50)}
        agreement['native_query_mismatches'] = sum(
            row['stages']['native']['query_index'] != old['query_index']
            for row, old in zip(rows, reference_native[arm]))
        agreement['final_variant_position_mismatches'] = sum(
            row['final_flat_index'] != old['selected_variant_position']
            for row, old in zip(rows, reference_final[arm]))
        summary['reference_agreement'][arm] = agreement
    for first, second in zip(records[arms[0]], records[arms[1]]):
        for key in ('row_id', 'scan_id', 'point_sha256', 'target_id', 'utterance', 'root_box'):
            assert first[key] == second[key]
    for name in STAGES:
        before = [row['stages'][name]['rec_iou'] for row in records[arms[0]]]
        after = [row['stages'][name]['rec_iou'] for row in records[arms[1]]]
        summary['paired_stage_changes'][name] = {
            suffix: transitions(before, after, threshold)
            for threshold, suffix in ((.25, '025'), (.50, '050'))}
    return summary
