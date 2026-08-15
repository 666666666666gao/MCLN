# ------------------------------------------------------------------------
# Modification: EDA
# Created: 05/21/2022
# Author: Yanmin Wu
# E-mail: wuyanminmax@gmail.com
# https://github.com/yanmin-wu/EDA 
# ------------------------------------------------------------------------
# BEAUTY DETR
# Copyright (c) 2022 Ayush Jain & Nikolaos Gkanatsios
# Licensed under CC-BY-NC [see LICENSE for details]
# All Rights Reserved
# ------------------------------------------------------------------------
"""A class to collect and evaluate language grounding results."""

import math

import torch

from models.losses import _iou3d_par, box_cxcyczwhd_to_xyzxyz
from models.mask_fusion import as_query_mask_logits, fuse_query_mask_logits
from models.rec_geometry_reranker import (
    stable_flat_descending_indices,
    stable_query_descending_order,
)
import utils.misc as misc
import numpy as np


POSITION_SUBGROUPS = (
    'unique',
    'multiple',
    'easy',
    'hard',
    'view_dependent',
    'view_independent',
)


def _load_wandb():
    import wandb
    return wandb

def box2points(box):
    """Convert box center/hwd coordinates to vertices (8x3)."""
    x_min, y_min, z_min = (box[:, :3] - (box[:, 3:] / 2)).transpose(1, 0)
    x_max, y_max, z_max = (box[:, :3] + (box[:, 3:] / 2)).transpose(1, 0)
    return np.stack((
        np.concatenate((x_min[:, None], y_min[:, None], z_min[:, None]), 1),
        np.concatenate((x_min[:, None], y_max[:, None], z_min[:, None]), 1),
        np.concatenate((x_max[:, None], y_min[:, None], z_min[:, None]), 1),
        np.concatenate((x_max[:, None], y_max[:, None], z_min[:, None]), 1),
        np.concatenate((x_min[:, None], y_min[:, None], z_max[:, None]), 1),
        np.concatenate((x_min[:, None], y_max[:, None], z_max[:, None]), 1),
        np.concatenate((x_max[:, None], y_min[:, None], z_max[:, None]), 1),
        np.concatenate((x_max[:, None], y_max[:, None], z_max[:, None]), 1)
    ), axis=1)

def softmax(x):
    """Numpy function for softmax."""
    shape = x.shape
    probs = np.exp(x - np.max(x, axis=len(shape) - 1, keepdims=True))
    probs /= np.sum(probs, axis=len(shape) - 1, keepdims=True)
    return probs

# BRIEF Evaluator
class GroundingEvaluator:
    """
    Evaluate language grounding.

    Args:
        only_root (bool): detect only the root noun
        thresholds (list): IoU thresholds to check
        topks (list): k to evaluate top--k accuracy
        prefixes (list): names of layers to evaluate
    """

    def __init__(self, only_root=True, thresholds=[0.25, 0.5],
                 topks=[1, 5, 10], prefixes=[], filter_non_gt_boxes=False,
                 logger=None, model=None,
                 eval_use_selector_choice_scores=False,
                 eval_use_rec_reranker_scores=False,
                 eval_use_rec_geometry_reranker_scores=False,
                 eval_use_rec_joint_box_mask=False):
        """Initialize accumulators."""
        self.only_root = only_root
        self.thresholds = thresholds
        self.topks = topks
        self.prefixes = prefixes
        self.filter_non_gt_boxes = filter_non_gt_boxes
        self.reset()
        self.logger = logger
        self.model = model
        self.eval_use_selector_choice_scores = bool(
            eval_use_selector_choice_scores
        )
        self.eval_use_rec_reranker_scores = bool(
            eval_use_rec_reranker_scores
        )
        self.eval_use_rec_geometry_reranker_scores = bool(
            eval_use_rec_geometry_reranker_scores
        )
        self.eval_use_rec_joint_box_mask = bool(
            eval_use_rec_joint_box_mask
        )
        self.visualization_pred = False
        self.visualization_gt = False
        self.bad_case_visualization = False
        self.kps_points_visualization = False
        self.bad_case_threshold = 0.15

    def reset(self):
        """Reset accumulators to empty."""
        self.dets = {
            (prefix, t, k, mode): 0
            for prefix in self.prefixes
            for t in self.thresholds
            for k in self.topks
            for mode in ['bbs', 'bbf']
        }
        self.gts = dict(self.dets)

        for threshold in self.thresholds:
            for group in POSITION_SUBGROUPS:
                key = ('position_subgroup', group, threshold)
                self.dets[key] = 0
                self.gts[key] = 0

        self.dets.update({'vd': 0, 'vid': 0})
        self.dets.update({'hard': 0, 'easy': 0})
        self.dets.update({'multi': 0, 'unique': 0})
        self.gts.update({'vd': 1e-14, 'vid': 1e-14})
        self.gts.update({'hard': 1e-14, 'easy': 1e-14})
        self.gts.update({'multi': 1e-14, 'unique': 1e-14})
        self.dets.update({'vd50': 0, 'vid50': 0})
        self.dets.update({'hard50': 0, 'easy50': 0})
        self.dets.update({'multi50': 0, 'unique50': 0})
        self.gts.update({'vd50': 1e-14, 'vid50': 1e-14})
        self.gts.update({'hard50': 1e-14, 'easy50': 1e-14})
        self.gts.update({'multi50': 1e-14, 'unique50': 1e-14})
        self.dets.update({'mask_pos': 0})
        self.gts.update({'mask_pos': 1e-14})
        self.dets.update({'mask_sem': 0})
        self.gts.update({'mask_sem': 1e-14})
        self.dets.update({'vd_mask': 0})
        self.dets.update({'vid_mask': 0})
        self.dets.update({'hard_mask': 0})
        self.dets.update({'easy_mask': 0})
        self.dets.update({'unique_mask': 0})
        self.dets.update({'multi_mask': 0})
        self.dets.update({'vd50_mask': 0})
        self.dets.update({'vid50_mask': 0})
        self.dets.update({'hard50_mask': 0})
        self.dets.update({'easy50_mask': 0})
        self.dets.update({'unique50_mask': 0})
        self.dets.update({'multi50_mask': 0})
        self.dets.update({'overall_mask': 0})
        self.dets.update({'overall50_mask': 0})
        self.gts.update({'vd_num': 0})
        self.gts.update({'vid_num': 0})
        self.gts.update({'easy_num': 0})
        self.gts.update({'hard_num': 0})
        self.gts.update({'unique_num': 0})
        self.gts.update({'multi_num': 0})

    @staticmethod
    def _retrain_integer_counter(value, name):
        if isinstance(value, (bool, str, bytes)):
            raise ValueError('{} must be an integer counter'.format(name))
        try:
            numeric = float(value)
        except (TypeError, ValueError, OverflowError):
            raise ValueError('{} must be an integer counter'.format(name))
        if not math.isfinite(numeric):
            raise ValueError('{} must be finite'.format(name))
        rounded = int(round(numeric))
        if abs(numeric - rounded) > 1e-9:
            raise ValueError(
                '{} must be within 1e-9 of an integer'.format(name)
            )
        if rounded < 0:
            raise ValueError('{} must be non-negative'.format(name))
        return rounded

    def export_retrain_metrics(self, expected_sample_count=None):
        """Export validated raw counters used by the MCLN retrainer."""
        source_keys = {
            'fixed_default': {
                'hits025': ('source_choice', 'fixed_default', 0.25, 1),
                'hits050': ('source_choice', 'fixed_default', 0.5, 1),
            },
            'learned_selector': {
                'hits025': ('source_choice', 'learned_selector', 0.25, 1),
                'hits050': ('source_choice', 'learned_selector', 0.5, 1),
            },
        }
        flat_source_keys = [
            key
            for metric_keys in source_keys.values()
            for key in metric_keys.values()
        ]
        for key in flat_source_keys:
            if key not in self.dets or key not in self.gts:
                raise ValueError(
                    'missing retrain source counter {}'.format(key)
                )
        for key in ('overall_mask', 'overall50_mask', 'mask_sem'):
            if key not in self.dets:
                raise ValueError(
                    'missing retrain mask counter {}'.format(key)
                )
        if 'mask_sem' not in self.gts:
            raise ValueError('missing retrain mask denominator mask_sem')

        denominators = [
            self._retrain_integer_counter(
                self.gts[key], 'denominator {!r}'.format(key)
            )
            for key in flat_source_keys
        ]
        denominators.append(self._retrain_integer_counter(
            self.gts['mask_sem'], 'denominator mask_sem'
        ))
        if len(set(denominators)) != 1:
            raise ValueError(
                'retrain metrics must share one sample denominator'
            )
        sample_count = denominators[0]
        if sample_count <= 0:
            raise ValueError(
                'retrain metrics require a positive sample denominator'
            )

        if expected_sample_count is not None:
            expected_sample_count = self._retrain_integer_counter(
                expected_sample_count, 'expected_sample_count'
            )
            if expected_sample_count != sample_count:
                raise ValueError(
                    'expected {} samples but evaluator contains {}'.format(
                        expected_sample_count, sample_count
                    )
                )

        position = {}
        all_hits = []
        for source_name, metric_keys in source_keys.items():
            hits = {
                metric_name: self._retrain_integer_counter(
                    self.dets[key], '{} {}'.format(source_name, metric_name)
                )
                for metric_name, key in metric_keys.items()
            }
            if hits['hits050'] > hits['hits025']:
                raise ValueError(
                    '{} hits050 cannot exceed hits025'.format(source_name)
                )
            position[source_name] = hits
            all_hits.extend(hits.values())

        mask_hits = {
            'hits025': self._retrain_integer_counter(
                self.dets['overall_mask'], 'mask hits025'
            ),
            'hits050': self._retrain_integer_counter(
                self.dets['overall50_mask'], 'mask hits050'
            ),
        }
        if mask_hits['hits050'] > mask_hits['hits025']:
            raise ValueError('mask hits050 cannot exceed hits025')
        all_hits.extend(mask_hits.values())
        if any(hit > sample_count for hit in all_hits):
            raise ValueError('retrain metric hits cannot exceed sample_count')

        position_subgroups = {}
        for group in ('unique', 'multiple'):
            subgroup = {}
            totals = []
            for threshold, suffix in ((0.25, '025'), (0.5, '050')):
                key = ('position_subgroup', group, threshold)
                if key not in self.dets or key not in self.gts:
                    raise ValueError(
                        'missing position subgroup counter {}'.format(key)
                    )
                hits = self._retrain_integer_counter(
                    self.dets[key], '{} hits{}'.format(group, suffix)
                )
                total = self._retrain_integer_counter(
                    self.gts[key], '{} total{}'.format(group, suffix)
                )
                if hits > total:
                    raise ValueError(
                        '{} hits{} cannot exceed its total'.format(
                            group, suffix
                        )
                    )
                subgroup['hits' + suffix] = hits
                subgroup['acc' + suffix] = (
                    hits / float(total) if total > 0 else 0.0
                )
                totals.append(total)
            if len(set(totals)) != 1:
                raise ValueError(
                    '{} subgroup thresholds must share one denominator'.format(
                        group
                    )
                )
            subgroup['sample_count'] = totals[0]
            position_subgroups[group] = subgroup
        if sum(
                group['sample_count']
                for group in position_subgroups.values()) != sample_count:
            raise ValueError(
                'unique and multiple subgroup counts must partition samples'
            )
        # The source-choice counters are recorded before the optional parent,
        # geometry and contextual REC rerankers.  Position subgroups instead
        # follow the final deployed ``last_`` ranking, so their hits need not
        # partition ``learned_selector`` once a downstream reranker is active.
        # Their denominators still strictly partition the evaluated samples,
        # and each subgroup must preserve threshold nesting.
        for group, subgroup in position_subgroups.items():
            if subgroup['hits050'] > subgroup['hits025']:
                raise ValueError(
                    '{} position subgroup hits050 cannot exceed hits025'
                    .format(group)
                )

        mask_position_subgroups = {}
        mask_group_keys = {
            'unique': ('unique_num', 'unique_mask', 'unique50_mask'),
            'multiple': ('multi_num', 'multi_mask', 'multi50_mask'),
        }
        for group, keys in mask_group_keys.items():
            total_key, hits025_key, hits050_key = keys
            for key in keys:
                mapping = self.gts if key == total_key else self.dets
                if key not in mapping:
                    raise ValueError(
                        'missing mask position subgroup counter {}'.format(key)
                    )
            total = self._retrain_integer_counter(
                self.gts[total_key], 'mask {} sample_count'.format(group)
            )
            hits025 = self._retrain_integer_counter(
                self.dets[hits025_key], 'mask {} hits025'.format(group)
            )
            hits050 = self._retrain_integer_counter(
                self.dets[hits050_key], 'mask {} hits050'.format(group)
            )
            if hits050 > hits025 or hits025 > total:
                raise ValueError(
                    'mask {} subgroup hits are inconsistent'.format(group)
                )
            mask_position_subgroups[group] = {
                'sample_count': total,
                'hits025': hits025,
                'hits050': hits050,
                'acc025': hits025 / float(total) if total > 0 else 0.0,
                'acc050': hits050 / float(total) if total > 0 else 0.0,
            }
        if sum(
                group['sample_count']
                for group in mask_position_subgroups.values()) != sample_count:
            raise ValueError(
                'mask unique and multiple counts must partition samples'
            )
        for suffix in ('025', '050'):
            if sum(
                    group['hits' + suffix]
                    for group in mask_position_subgroups.values()
            ) != mask_hits['hits' + suffix]:
                raise ValueError(
                    'mask unique and multiple hits{} must partition mask hits'
                    .format(suffix)
                )

        try:
            iou_sum = float(self.dets['mask_sem'])
        except (TypeError, ValueError, OverflowError):
            raise ValueError('mask iou_sum must be finite')
        if not math.isfinite(iou_sum):
            raise ValueError('mask iou_sum must be finite')

        return {
            'schema': 'mcln-retrain-metrics-v1',
            'sample_count': sample_count,
            'position': position,
            'position_subgroups': position_subgroups,
            'mask': {
                'hits025': mask_hits['hits025'],
                'hits050': mask_hits['hits050'],
                'iou_sum': iou_sum,
                'miou': iou_sum / sample_count,
                'position_subgroups': mask_position_subgroups,
            },
        }

    def export_source_choice_diagnostics(self, expected_sample_count=None):
        """Export the gate candidate-set oracle without changing metrics v1."""
        oracle_keys = {
            'hits025': ('source_choice', 'gate_candidate_oracle', 0.25, 1),
            'hits050': ('source_choice', 'gate_candidate_oracle', 0.5, 1),
        }
        headroom_keys = {
            'hits025': (
                'source_choice_effect', 0.25, 'gate_oracle_headroom'
            ),
            'hits050': (
                'source_choice_effect', 0.5, 'gate_oracle_headroom'
            ),
        }
        mean_iou_key = (
            'source_choice_mean_iou', 'gate_candidate_oracle'
        )
        required_keys = (
            list(oracle_keys.values())
            + list(headroom_keys.values())
            + [mean_iou_key]
        )
        present = [
            key in self.dets or key in self.gts for key in required_keys
        ]
        if not any(present):
            return None
        if not all(
                key in self.dets and key in self.gts
                for key in required_keys):
            raise ValueError(
                'gate candidate-set oracle diagnostics are incomplete'
            )

        denominators = [
            self._retrain_integer_counter(
                self.gts[key], 'gate diagnostic denominator {}'.format(key)
            )
            for key in required_keys
        ]
        if len(set(denominators)) != 1 or denominators[0] <= 0:
            raise ValueError(
                'gate candidate-set oracle diagnostics must share one '
                'positive denominator'
            )
        sample_count = denominators[0]
        if expected_sample_count is not None:
            expected_sample_count = self._retrain_integer_counter(
                expected_sample_count, 'expected_sample_count'
            )
            if expected_sample_count != sample_count:
                raise ValueError(
                    'expected {} samples but gate diagnostics contain {}'.format(
                        expected_sample_count, sample_count
                    )
                )

        oracle_hits = {
            name: self._retrain_integer_counter(
                self.dets[key], 'gate candidate oracle {}'.format(name)
            )
            for name, key in oracle_keys.items()
        }
        headroom_hits = {
            name: self._retrain_integer_counter(
                self.dets[key], 'gate oracle headroom {}'.format(name)
            )
            for name, key in headroom_keys.items()
        }
        if any(
                value < 0 or value > sample_count
                for value in list(oracle_hits.values())
                + list(headroom_hits.values())):
            raise ValueError('gate diagnostic hits are out of range')
        if oracle_hits['hits050'] > oracle_hits['hits025']:
            raise ValueError(
                'gate candidate oracle hits050 cannot exceed hits025'
            )
        try:
            iou_sum = float(self.dets[mean_iou_key])
        except (TypeError, ValueError, OverflowError):
            raise ValueError('gate candidate oracle iou_sum must be finite')
        if (
                not math.isfinite(iou_sum)
                or iou_sum < 0.0
                or iou_sum > float(sample_count)):
            raise ValueError(
                'gate candidate oracle iou_sum is out of range'
            )

        return {
            'schema': 'mcln-source-choice-diagnostics-v1',
            'sample_count': sample_count,
            'gate_candidate_oracle': {
                'hits025': oracle_hits['hits025'],
                'hits050': oracle_hits['hits050'],
                'iou_sum': iou_sum,
                'miou': iou_sum / sample_count,
            },
            'gate_oracle_headroom': {
                'hits025': headroom_hits['hits025'],
                'hits050': headroom_hits['hits050'],
                'rate025': headroom_hits['hits025'] / float(sample_count),
                'rate050': headroom_hits['hits050'] / float(sample_count),
            },
        }

    def print_stats(self):
        """Print accumulated accuracies."""
        mode_str = {
            'bbs': 'position alignment',
            'bbf': 'semantic alignment'
        }
        for prefix in self.prefixes:
            for mode in ['bbs', 'bbf']:
                for t in self.thresholds:
                    self.logger.info(
                    # print(
                        prefix + ' ' + mode_str[mode] + ' ' +  'Acc%.2f:' % t + ' ' + 
                        ', '.join([
                            'Top-%d: %.5f' % (
                                k,
                                self.dets[(prefix, t, k, mode)]
                                / max(self.gts[(prefix, t, k, mode)], 1)
                            )
                            for k in self.topks
                        ])
                    )
        self.logger.info('\nPosition subgroup diagnostics')
        for threshold in self.thresholds:
            for group in POSITION_SUBGROUPS:
                key = ('position_subgroup', group, threshold)
                hits = int(self.dets[key])
                total = int(self.gts[key])
                accuracy = hits / float(max(total, 1))
                self.logger.info(
                    'position subgroup {} Acc{:.2f}: '
                    'hits={}, total={}, accuracy={:.12f}'.format(
                        group, threshold, hits, total, accuracy
                    )
                )
        self.logger.info('\nAnalysis')
        self.logger.info('iou@0.25')
        for field in ['easy', 'hard', 'vd', 'vid', 'unique', 'multi']:
            self.logger.info(field + ' ' +  str(self.dets[field] / self.gts[field]))
        self.logger.info('iou@0.50')
        for field in ['easy50', 'hard50', 'vd50', 'vid50', 'unique50', 'multi50']:
            self.logger.info(field + ' ' +  str(self.dets[field] / self.gts[field]))
        self.logger.info('mask@mean iou')
        self.logger.info('mask_pos' + ' ' +  str(self.dets['mask_pos'] / self.gts['mask_sem']))
        self.logger.info('mask_sem' + ' ' +  str(self.dets['mask_sem'] / self.gts['mask_sem']))
        self.logger.info('mask@kiou')
        if self.gts['unique_num'] != 0:
            self.logger.info('unique25' + ' ' +  str(self.dets['unique_mask'] / self.gts['unique_num']))
            self.logger.info('unique50' + ' ' +  str(self.dets['unique50_mask'] / self.gts['unique_num']))
            self.logger.info('multi25' + ' ' +  str(self.dets['multi_mask'] / self.gts['multi_num']))
            self.logger.info('multi50' + ' ' +  str(self.dets['multi50_mask'] / self.gts['multi_num']))
        self.logger.info('overall25' + ' ' +  str(self.dets['overall_mask'] / self.gts['mask_sem']))
        self.logger.info('overall50' + ' ' +  str(self.dets['overall50_mask'] / self.gts['mask_sem']))
        self.logger.info('mask@identity')
        self.logger.info('vd25' + ' ' +  str(self.dets['vd_mask'] / max(self.gts['vd_num'], 1)))
        self.logger.info('vd50' + ' ' +  str(self.dets['vd50_mask'] / max(self.gts['vd_num'], 1)))
        self.logger.info('vid25' + ' ' +  str(self.dets['vid_mask'] / max(self.gts['vid_num'], 1)))
        self.logger.info('vid50' + ' ' +  str(self.dets['vid50_mask'] / max(self.gts['vid_num'], 1)))
        self.logger.info('easy25' + ' ' +  str(self.dets['easy_mask'] / max(self.gts['easy_num'], 1)))
        self.logger.info('easy50' + ' ' +  str(self.dets['easy50_mask'] / max(self.gts['easy_num'], 1)))
        self.logger.info('hard25' + ' ' +  str(self.dets['hard_mask'] / max(self.gts['hard_num'], 1)))
        self.logger.info('hard50' + ' ' +  str(self.dets['hard50_mask'] / max(self.gts['hard_num'], 1)))

        source_choice_keys = [
            key for key in self.gts
            if isinstance(key, tuple) and len(key) == 4
            and key[0] == 'source_choice'
        ]
        if source_choice_keys:
            self.logger.info('\nSource choice diagnostics')
            source_names = sorted({key[1] for key in source_choice_keys})
            for source_name in source_names:
                parts = []
                for threshold in self.thresholds:
                    key = ('source_choice', source_name, threshold, 1)
                    if key not in self.gts:
                        continue
                    parts.append(
                        'Acc%.2f Top-1: %.5f' % (
                            threshold,
                            self.dets[key] / max(self.gts[key], 1)
                        )
                    )
                if parts:
                    self.logger.info(
                        source_name + ' ' + ', '.join(parts)
                    )

        mean_iou_keys = [
            key for key in self.gts
            if isinstance(key, tuple) and len(key) == 2
            and key[0] == 'source_choice_mean_iou'
        ]
        if mean_iou_keys:
            self.logger.info('\nSource choice mean IoU')
            for key in sorted(mean_iou_keys):
                self.logger.info(
                    '{} {:.5f}'.format(
                        key[1],
                        self.dets[key] / max(self.gts[key], 1)
                    )
                )

        selected_source_keys = [
            key for key in self.gts
            if isinstance(key, tuple) and len(key) == 2
            and key[0] == 'source_choice_selected_source'
        ]
        if selected_source_keys:
            self.logger.info('\nSource choice selected-source ratio')
            for key in sorted(selected_source_keys):
                self.logger.info(
                    '{} {:.5f}'.format(
                        key[1],
                        self.dets[key] / max(self.gts[key], 1)
                    )
                )

        effect_keys = [
            key for key in self.gts
            if isinstance(key, tuple) and len(key) == 3
            and key[0] == 'source_choice_effect'
        ]
        if effect_keys:
            self.logger.info('\nSource choice threshold effects')
            for threshold in sorted({key[1] for key in effect_keys}):
                parts = []
                for effect_name in [
                        'selector_fix',
                        'selector_break',
                        'candidate_fix',
                        'candidate_break',
                        'selector_kept_correct',
                        'selector_kept_wrong',
                        'oracle_headroom',
                        'gate_oracle_headroom']:
                    key = ('source_choice_effect', threshold, effect_name)
                    if key not in self.gts:
                        continue
                    parts.append(
                        '{}: {:.5f}'.format(
                            effect_name,
                            self.dets[key] / max(self.gts[key], 1)
                        )
                    )
                if parts:
                    self.logger.info(
                        'Acc{:.2f} {}'.format(
                            threshold,
                            ', '.join(parts)
                        )
                    )



    def synchronize_between_processes(self):
        all_dets = misc.all_gather(self.dets)
        all_gts = misc.all_gather(self.gts)

        if misc.is_main_process():
            merged_predictions = {}
            det_keys = set()
            for p in all_dets:
                det_keys.update(p.keys())
            for key in det_keys:
                merged_predictions[key] = 0
                for p in all_dets:
                    merged_predictions[key] += p.get(key, 0)
            self.dets = merged_predictions

            merged_predictions = {}
            gt_keys = set()
            for p in all_gts:
                gt_keys.update(p.keys())
            for key in gt_keys:
                merged_predictions[key] = 0
                for p in all_gts:
                    merged_predictions[key] += p.get(key, 0)
            self.gts = merged_predictions

    # BRIEF Evaluation
    def evaluate(self, end_points, prefix):
        """
        Evaluate all accuracies.

        Args:
            end_points (dict): contains predictions and gt
            prefix (str): layer name
        """
        # NOTE Two Evaluation Ways: position alignment, semantic alignment
        self.evaluate_bbox_by_pos_align(end_points, prefix)
        self.evaluate_source_choice_diagnostics(end_points, prefix)
        self.evaluate_bbox_by_sem_align(end_points, prefix)
        self.evaluate_masks_by_pos_align(end_points, prefix)
        self.evaluate_masks_by_sem_align(end_points, prefix)

    def _record_source_choice_iou(self, source_name, iou):
        for threshold in self.thresholds:
            key = ('source_choice', source_name, threshold, 1)
            if key not in self.dets:
                self.dets[key] = 0
                self.gts[key] = 0
            self.dets[key] += float((iou > threshold).item())
            self.gts[key] += 1

    def _record_counter(self, namespace, name, value=1.0):
        key = (namespace, name)
        if key not in self.dets:
            self.dets[key] = 0.0
            self.gts[key] = 0.0
        self.dets[key] += float(value)
        self.gts[key] += 1.0

    def _record_position_subgroups(
            self, end_points, bid, threshold, found):
        metadata = ('is_unique', 'is_hard', 'is_view_dep')
        if any(key not in end_points for key in metadata):
            return
        if found.numel() != 1:
            raise ValueError(
                'position subgroup reporting requires exactly one root'
            )
        groups = (
            'unique' if bool(end_points['is_unique'][bid].item())
            else 'multiple',
            'hard' if bool(end_points['is_hard'][bid].item()) else 'easy',
            'view_dependent'
            if bool(end_points['is_view_dep'][bid].item())
            else 'view_independent',
        )
        hit = int(bool(found[0].item()))
        for group in groups:
            key = ('position_subgroup', group, threshold)
            self.dets[key] += hit
            self.gts[key] += 1

    def _record_effect(self, threshold, effect_name, value):
        key = ('source_choice_effect', threshold, effect_name)
        if key not in self.dets:
            self.dets[key] = 0.0
            self.gts[key] = 0.0
        self.dets[key] += float(value)
        self.gts[key] += 1.0

    @staticmethod
    def _top1_iou_for_scores(scores, pred_bbox, gt_bbox):
        top_idx = scores.argmax().item()
        ious, _ = _iou3d_par(
            box_cxcyczwhd_to_xyzxyz(gt_bbox),
            box_cxcyczwhd_to_xyzxyz(pred_bbox[top_idx:top_idx + 1])
        )
        return ious.max()

    @staticmethod
    def _oracle_iou_for_query_mask(query_mask, pred_bbox, gt_bbox):
        if (not isinstance(query_mask, torch.Tensor)
                or query_mask.dtype != torch.bool
                or query_mask.dim() != 1
                or query_mask.shape[0] != pred_bbox.shape[0]
                or query_mask.device != pred_bbox.device
                or not bool(query_mask.any().item())):
            raise ValueError(
                'gate oracle query mask must select at least one query'
            )
        candidate_boxes = pred_bbox[query_mask]
        ious, _ = _iou3d_par(
            box_cxcyczwhd_to_xyzxyz(gt_bbox),
            box_cxcyczwhd_to_xyzxyz(candidate_boxes),
        )
        return ious.max()

    @staticmethod
    def _validate_rec_reranker_scores(scores, pred_bbox):
        if (not isinstance(scores, torch.Tensor)
                or scores.shape != pred_bbox.shape[:2]
                or scores.device != pred_bbox.device
                or not torch.is_floating_point(scores)
                or bool(torch.isnan(scores).any().item())
                or bool(torch.isposinf(scores).any().item())
                or not bool(torch.isfinite(scores).any(dim=1).all().item())):
            raise ValueError(
                'enabled rec_reranker_scores must have shape [B,Q], '
                'share the prediction device, contain no NaN/+inf, and '
                'include a finite score per row'
            )
        return torch.isfinite(scores)

    @staticmethod
    def _validate_active_geometry_axis(boxes, scores, valid, fallback,
                                       pred_bbox):
        batch_size = pred_bbox.shape[0]
        if (not isinstance(boxes, torch.Tensor) or boxes.dim() != 3
                or boxes.shape[0] != batch_size
                or boxes.shape[1] <= 0 or boxes.shape[2] != 6
                or not isinstance(scores, torch.Tensor) or scores.dim() != 2
                or tuple(scores.shape) != tuple(boxes.shape[:2])
                or not isinstance(valid, torch.Tensor) or valid.dim() != 2
                or valid.dtype != torch.bool
                or tuple(valid.shape) != tuple(scores.shape)
                or not isinstance(fallback, torch.Tensor)
                or fallback.dtype != torch.long
                or tuple(fallback.shape) != (batch_size,)):
            raise ValueError('geometry candidate tensor shapes are invalid')
        if (boxes.dtype != torch.float32 or scores.dtype != torch.float32):
            raise ValueError('geometry boxes and scores must use float32')
        if any(value.device != pred_bbox.device for value in (
                boxes, scores, valid, fallback)):
            raise ValueError(
                'geometry candidate tensors must share the prediction device'
            )
        if not bool(valid.any(dim=1).all().item()):
            raise ValueError(
                'every geometry row needs at least one valid candidate'
            )
        if not bool(torch.isfinite(scores[valid]).all().item()):
            raise ValueError('valid geometry scores must be finite')
        valid_boxes = boxes[valid]
        if (not bool(torch.isfinite(valid_boxes).all().item())
                or not bool((valid_boxes[:, 3:] > 0.0).all().item())):
            raise ValueError(
                'valid geometry boxes must be finite with positive size'
            )
        if (bool((fallback < 0).any().item())
                or bool((fallback >= scores.shape[1]).any().item())):
            raise ValueError('geometry fallback index is out of range')
        fallback_valid = torch.gather(
            valid, 1, fallback.unsqueeze(1)
        ).squeeze(1)
        if not bool(fallback_valid.all().item()):
            raise ValueError(
                'geometry fallback index must identify a valid candidate'
            )

    def _resolve_position_candidates(self, end_points, prefix, pred_bbox):
        """Resolve the boxes, optional scores, validity, and ordering policy."""
        default_valid = torch.ones(
            pred_bbox.shape[:2], dtype=torch.bool, device=pred_bbox.device
        )
        if prefix != 'last_':
            return pred_bbox, None, default_valid, 'default_query_axis'

        geometry_keys = (
            'rec_geometry_boxes',
            'rec_geometry_scores',
            'rec_geometry_valid_mask',
            'rec_geometry_fallback_index',
        )
        if self.eval_use_rec_geometry_reranker_scores:
            mode = end_points.get('rec_geometry_runtime_mode')
            present_geometry_keys = tuple(
                key for key in geometry_keys if key in end_points
            )
            if mode == 'flat_geometry_axis':
                if len(present_geometry_keys) != len(geometry_keys):
                    raise ValueError(
                        'flat geometry mode requires a complete geometry '
                        'attachment'
                    )
                boxes = end_points['rec_geometry_boxes']
                scores = end_points['rec_geometry_scores']
                valid = end_points['rec_geometry_valid_mask']
                fallback = end_points['rec_geometry_fallback_index']
                self._validate_active_geometry_axis(
                    boxes, scores, valid, fallback, pred_bbox
                )
                return boxes, scores, valid, mode
            if mode == 'parent_query_axis':
                if present_geometry_keys:
                    raise ValueError(
                        'parent query mode cannot carry geometry tensors'
                    )
                scores = end_points.get('rec_reranker_scores')
                valid = self._validate_rec_reranker_scores(
                    scores, pred_bbox
                )
                return pred_bbox, scores, valid, mode
            raise ValueError(
                'enabled REC geometry requires a valid geometry runtime mode'
            )

        if self.eval_use_rec_reranker_scores:
            scores = end_points.get('rec_reranker_scores')
            valid = self._validate_rec_reranker_scores(scores, pred_bbox)
            return pred_bbox, scores, valid, 'parent_query_axis'
        return pred_bbox, None, default_valid, 'default_query_axis'

    @staticmethod
    def _position_top_indices(scores, valid, axis_mode, max_topk):
        if (scores.dim() != 2 or valid.shape != scores.shape
                or not bool(valid.any(dim=1).all().item())):
            raise ValueError('position ranking has no valid candidate')
        if axis_mode == 'flat_geometry_axis':
            orders = stable_flat_descending_indices(scores, valid)
        elif axis_mode == 'parent_query_axis':
            canonical = stable_query_descending_order(scores)
            orders = tuple(
                tuple(canonical[row][
                    valid[row, canonical[row]]
                ].detach().cpu().tolist())
                for row in range(scores.shape[0])
            )
        else:
            masked_scores = scores.masked_fill(~valid, -float('inf'))
            sorted_indices = masked_scores.argsort(1, True)
            sorted_valid = torch.gather(valid, 1, sorted_indices)
            top_count = min(
                max_topk, int(valid.sum(dim=1).min().item())
            )
            if top_count <= 0:
                raise ValueError('position ranking has no valid candidate')
            selected = sorted_valid & (
                sorted_valid.to(torch.int64).cumsum(dim=1) <= top_count
            )
            return sorted_indices[selected].reshape(
                scores.shape[0], top_count
            )
        top_count = min(max_topk, min(len(order) for order in orders))
        if top_count <= 0:
            raise ValueError('position ranking has no valid candidate')
        return torch.tensor(
            [order[:top_count] for order in orders],
            dtype=torch.long,
            device=scores.device,
        )

    def _resolve_joint_mask_parent_queries(
            self, end_points, prefix, num_queries, device):
        """Return the geometry winner's original detector query per sample."""
        if not self.eval_use_rec_joint_box_mask or prefix != 'last_':
            return None
        if end_points.get('rec_geometry_runtime_mode') != 'flat_geometry_axis':
            raise ValueError(
                'joint box mask requires flat geometry runtime candidates'
            )
        geometry_keys = (
            'rec_geometry_boxes',
            'rec_geometry_scores',
            'rec_geometry_valid_mask',
            'rec_geometry_fallback_index',
            'rec_geometry_parent_query_indices',
        )
        missing = tuple(key for key in geometry_keys if key not in end_points)
        if missing:
            raise ValueError(
                'joint box mask requires a complete flat geometry parent mapping'
            )
        boxes = end_points['rec_geometry_boxes']
        scores = end_points['rec_geometry_scores']
        valid = end_points['rec_geometry_valid_mask']
        fallback = end_points['rec_geometry_fallback_index']
        mapping = end_points['rec_geometry_parent_query_indices']
        reference = torch.empty(
            (scores.shape[0], num_queries, 6),
            dtype=torch.float32,
            device=device,
        ) if isinstance(scores, torch.Tensor) and scores.dim() == 2 else None
        if reference is None:
            raise ValueError('joint box mask geometry scores have invalid shape')
        self._validate_active_geometry_axis(
            boxes, scores, valid, fallback, reference
        )
        if (not isinstance(mapping, torch.Tensor)
                or mapping.dtype != torch.long
                or mapping.dim() != 2
                or tuple(mapping.shape) != tuple(scores.shape)
                or mapping.device != scores.device
                or mapping.device != device):
            raise ValueError(
                'joint box mask parent mapping must be int64 with geometry shape'
            )
        if (bool((mapping < 0).any().item())
                or bool((mapping >= num_queries).any().item())):
            raise ValueError('joint box mask parent mapping index is out of range')
        orders = stable_flat_descending_indices(scores, valid)
        selected_flat = torch.tensor(
            [order[0] for order in orders],
            dtype=torch.long,
            device=scores.device,
        )
        return torch.gather(
            mapping, 1, selected_flat.unsqueeze(1)
        ).squeeze(1)

    def _resolve_joint_mask_policy(
            self, end_points, prefix, num_queries, device):
        """Validate the learned source/threshold bound to the REC winner."""
        parent_queries = self._resolve_joint_mask_parent_queries(
            end_points, prefix, num_queries, device
        )
        if parent_queries is None:
            return None
        from models.rec_joint_box_mask import (
            MASK_LOGIT_THRESHOLDS,
            MASK_POLICY_COUNT,
            MASK_SOURCE_NAMES,
        )
        keys = (
            'rec_joint_selected_flat_index',
            'rec_joint_selected_parent_position',
            'rec_joint_mask_policy_index',
            'rec_joint_mask_source_index',
            'rec_joint_mask_threshold_index',
            'rec_joint_mask_threshold',
        )
        missing = tuple(key for key in keys if key not in end_points)
        if missing:
            raise ValueError(
                'joint box mask requires learned mask policy payload'
            )
        batch_size = parent_queries.shape[0]
        integers = [end_points[key] for key in keys[:-1]]
        threshold = end_points[keys[-1]]
        scores = end_points['rec_geometry_scores']
        valid = end_points['rec_geometry_valid_mask']
        if (any(not isinstance(value, torch.Tensor)
                or value.dtype != torch.long
                or tuple(value.shape) != (batch_size,)
                or value.device != device for value in integers)
                or not isinstance(threshold, torch.Tensor)
                or threshold.dtype != torch.float32
                or tuple(threshold.shape) != (batch_size,)
                or threshold.device != device
                or not bool(torch.isfinite(threshold).all().item())):
            raise ValueError('joint mask policy tensor schema is invalid')
        selected, parent_position, policy, source, threshold_index = integers
        orders = stable_flat_descending_indices(scores, valid)
        expected_selected = torch.tensor(
            [order[0] for order in orders],
            dtype=torch.long,
            device=device,
        )
        if not torch.equal(selected, expected_selected):
            raise ValueError('joint mask policy is not bound to final REC winner')
        if (bool((parent_position < 0).any().item())
                or bool((parent_position >= 16).any().item())
                or bool((policy < 0).any().item())
                or bool((policy >= MASK_POLICY_COUNT).any().item())
                or bool((source < 0).any().item())
                or bool((source >= len(MASK_SOURCE_NAMES)).any().item())
                or bool((threshold_index < 0).any().item())
                or bool((threshold_index >= len(
                    MASK_LOGIT_THRESHOLDS
                )).any().item())):
            raise ValueError('joint mask policy index is out of range')
        if (not torch.equal(
                parent_position,
                torch.div(selected, 7, rounding_mode='floor'))
                or not torch.equal(
                    source,
                    torch.div(
                        policy, len(MASK_LOGIT_THRESHOLDS),
                        rounding_mode='floor',
                    ))
                or not torch.equal(
                    threshold_index,
                    torch.remainder(policy, len(MASK_LOGIT_THRESHOLDS)))):
            raise ValueError('joint mask policy identity is inconsistent')
        expected_threshold = torch.tensor(
            MASK_LOGIT_THRESHOLDS,
            dtype=threshold.dtype,
            device=device,
        )[threshold_index]
        if not torch.equal(threshold, expected_threshold):
            raise ValueError('joint mask threshold is inconsistent')
        mapping = end_points['rec_geometry_parent_query_indices']
        expected_parent = mapping.gather(
            1, selected.unsqueeze(1)
        ).squeeze(1)
        if not torch.equal(parent_queries, expected_parent):
            raise ValueError('joint mask parent query mapping drifted')
        return {
            'parent_queries': parent_queries,
            'source_indices': source,
            'threshold_indices': threshold_index,
            'thresholds': threshold,
        }

    def _build_mask_point_predictions(self, end_points, prefix):
        """Build formal point masks, applying the joint policy when enabled."""
        centers = end_points.get('{}center'.format(prefix))
        if (not isinstance(centers, torch.Tensor) or centers.dim() != 3):
            raise ValueError('mask evaluation needs query centers')
        joint_policy = self._resolve_joint_mask_policy(
            end_points, prefix, centers.shape[1], centers.device
        )
        if self.model == "ThreeDRefTR_SP":
            if joint_policy is not None:
                raise ValueError('joint mask policy requires MCLN mask sources')
            pred_masks = []
            for bs in range(len(end_points['last_pred_masks'])):
                pred_masks_ = end_points['last_pred_masks'][bs].unsqueeze(0)
                pred_masks_ = (pred_masks_.sigmoid() > 0.5).int()
                superpoints = end_points['superpoints'][bs].unsqueeze(0)
                pred_masks_ = torch.gather(
                    pred_masks_, 2,
                    superpoints.unsqueeze(1).expand(
                        -1, centers.shape[1], -1
                    ),
                )
                pred_masks.append(pred_masks_.squeeze(0))
            return torch.stack(pred_masks, dim=0), joint_policy
        if self.model == "ThreeDRefTR_HR":
            if joint_policy is not None:
                raise ValueError('joint mask policy requires MCLN mask sources')
            return (
                (end_points['last_pred_masks'].sigmoid() > 0.5).int(),
                joint_policy,
            )

        pred_masks = []
        for bs in range(len(end_points['last_pred_masks'])):
            text_masks = as_query_mask_logits(
                end_points['last_pred_masks'][bs], 'last_pred_masks'
            )
            query_masks = as_query_mask_logits(
                end_points['sp_last_pred_masks'][bs], 'sp_last_pred_masks'
            )
            fused_masks = fuse_query_mask_logits(
                text_masks,
                query_masks,
                end_points['adaptive_weights'][bs],
            )
            binary_masks = fused_masks > 0.0
            if joint_policy is not None:
                query_index = int(
                    joint_policy['parent_queries'][bs].item()
                )
                source_index = int(
                    joint_policy['source_indices'][bs].item()
                )
                threshold = joint_policy['thresholds'][bs]
                source_logits = (
                    text_masks, query_masks, fused_masks
                )[source_index]
                binary_masks = binary_masks.clone()
                binary_masks[query_index] = (
                    source_logits[query_index] > threshold
                )
            superpoints = end_points['superpoints'][bs].unsqueeze(0)
            point_masks = torch.gather(
                binary_masks.int().unsqueeze(0), 2,
                superpoints.unsqueeze(1).expand(
                    -1, centers.shape[1], -1
                ),
            )
            pred_masks.append(point_masks.squeeze(0))
        return torch.stack(pred_masks, dim=0), joint_policy

    def _resolve_learned_mask_queries(
            self, end_points, prefix, num_queries, device):
        """Use the learned REC ranking for mask selection on the same query."""
        if (not self.eval_use_selector_choice_scores or prefix != 'last_'
                or self.eval_use_rec_joint_box_mask):
            return None
        scores = end_points.get('selected_source_scores')
        if (not isinstance(scores, torch.Tensor)
                or scores.dim() != 2
                or scores.shape != (len(end_points['box_label_mask']), num_queries)
                or scores.device != device
                or not torch.is_floating_point(scores)
                or not bool(torch.isfinite(scores).all().item())):
            raise ValueError(
                'learned mask query scores must be finite and align [B,Q]'
            )
        valid = end_points.get('moe_valid_mask')
        if valid is not None:
            if (not isinstance(valid, torch.Tensor)
                    or valid.dtype != torch.bool
                    or valid.shape != scores.shape
                    or valid.device != device
                    or not bool(valid.any(dim=1).all().item())):
                raise ValueError('learned mask query validity is invalid')
            scores = scores.masked_fill(~valid, -float('inf'))
        return scores.argmax(dim=1)

    def evaluate_source_choice_diagnostics(self, end_points, prefix):
        """Evaluate fixed source, learned selector and oracle top-1 source."""
        if prefix != 'last_' or 'source_choice_source_scores' not in end_points:
            return

        _, _, _, _, _, _, gt_bboxes = self._parse_gt(end_points)
        pred_bbox = torch.cat([
            end_points[f'{prefix}center'],
            end_points[f'{prefix}pred_size'],
        ], dim=-1)
        source_scores = end_points['source_choice_source_scores']
        source_names = end_points.get(
            'selector_choice_source_names',
            list(source_scores.keys())
        )

        gate_candidate_mask = end_points.get('moe_gate_candidate_mask')
        gate_default_query = end_points.get('moe_gate_default_query')
        gate_action_anchor = end_points.get(
            'moe_gate_supervision_fallback_query',
            end_points.get('moe_gate_action_anchor_query', gate_default_query),
        )
        if (gate_candidate_mask is None) != (gate_default_query is None):
            raise ValueError(
                'gate oracle diagnostics require candidate mask and default query'
            )
        gate_valid_mask = end_points.get('moe_valid_mask')
        if gate_candidate_mask is not None:
            expected_shape = pred_bbox.shape[:2]
            batch_size, num_queries = expected_shape
            if (not isinstance(gate_candidate_mask, torch.Tensor)
                    or gate_candidate_mask.dtype != torch.bool
                    or gate_candidate_mask.shape != expected_shape
                    or gate_candidate_mask.device != pred_bbox.device):
                raise ValueError(
                    'gate candidate mask must be bool and align [B,Q]'
                )
            if (not isinstance(gate_default_query, torch.Tensor)
                    or gate_default_query.dtype != torch.long
                    or gate_default_query.shape != (batch_size,)
                    or gate_default_query.device != pred_bbox.device
                    or bool((gate_default_query < 0).any().item())
                    or bool((gate_default_query >= num_queries).any().item())):
                raise ValueError(
                    'gate default query must be int64 with shape [B]'
                )
            if (not isinstance(gate_action_anchor, torch.Tensor)
                    or gate_action_anchor.dtype != torch.long
                    or gate_action_anchor.shape != (batch_size,)
                    or gate_action_anchor.device != pred_bbox.device
                    or bool((gate_action_anchor < 0).any().item())
                    or bool((gate_action_anchor >= num_queries).any().item())):
                raise ValueError(
                    'gate action anchor must be int64 with shape [B]'
                )
            row_index = torch.arange(batch_size, device=pred_bbox.device)
            if gate_valid_mask is not None:
                if (not isinstance(gate_valid_mask, torch.Tensor)
                        or gate_valid_mask.dtype != torch.bool
                        or gate_valid_mask.shape != expected_shape
                        or gate_valid_mask.device != pred_bbox.device
                        or not bool(gate_valid_mask[
                            row_index, gate_default_query
                        ].all().item())
                        or not bool(gate_valid_mask[
                            row_index, gate_action_anchor
                        ].all().item())
                        or bool((gate_candidate_mask & ~gate_valid_mask).any().item())):
                    raise ValueError(
                        'gate oracle candidates must be valid detector queries'
                    )

        for bid in range(len(gt_bboxes)):
            num_obj = int(end_points['box_label_mask'][bid].sum())
            num_obj = max(1, min(num_obj, gt_bboxes.shape[1]))
            gt_bbox = gt_bboxes[bid, :num_obj]
            source_ious = []
            source_iou_by_name = {}
            gate_oracle_iou = None

            for source_name in source_names:
                if source_name not in source_scores:
                    continue
                iou = self._top1_iou_for_scores(
                    source_scores[source_name][bid],
                    pred_bbox[bid],
                    gt_bbox,
                )
                self._record_source_choice_iou(
                    'fixed_' + source_name,
                    iou,
                )
                self._record_counter(
                    'source_choice_mean_iou',
                    'fixed_' + source_name,
                    iou.item(),
                )
                source_ious.append(iou)
                source_iou_by_name[source_name] = iou

            if source_ious:
                default_iou = source_ious[0]
                if 'default' in source_iou_by_name:
                    default_iou = source_iou_by_name['default']
                oracle_iou = torch.stack(source_ious).max()
                self._record_source_choice_iou('oracle', oracle_iou)
                self._record_counter(
                    'source_choice_mean_iou',
                    'oracle',
                    oracle_iou.item(),
                )

            if gate_candidate_mask is not None:
                gate_oracle_mask = gate_candidate_mask[bid].clone()
                gate_oracle_mask[gate_action_anchor[bid]] = True
                gate_oracle_iou = self._oracle_iou_for_query_mask(
                    gate_oracle_mask, pred_bbox[bid], gt_bbox
                )
                self._record_source_choice_iou(
                    'gate_candidate_oracle', gate_oracle_iou
                )
                self._record_counter(
                    'source_choice_mean_iou',
                    'gate_candidate_oracle',
                    gate_oracle_iou.item(),
                )

            if 'selected_source_scores' in end_points:
                iou = self._top1_iou_for_scores(
                    end_points['selected_source_scores'][bid],
                    pred_bbox[bid],
                    gt_bbox,
                )
                self._record_source_choice_iou('learned_selector', iou)
                self._record_counter(
                    'source_choice_mean_iou',
                    'learned_selector',
                    iou.item(),
                )

                candidate_iou = None
                if 'moe_candidate_scores' in end_points:
                    candidate_iou = self._top1_iou_for_scores(
                        end_points['moe_candidate_scores'][bid],
                        pred_bbox[bid],
                        gt_bbox,
                    )
                    self._record_source_choice_iou(
                        'moe_candidate', candidate_iou
                    )
                    self._record_counter(
                        'source_choice_mean_iou',
                        'moe_candidate',
                        candidate_iou.item(),
                    )

                selected_source_name = None
                if 'selected_source_id' in end_points:
                    source_id = int(end_points['selected_source_id'][bid].item())
                    if 0 <= source_id < len(source_names):
                        selected_source_name = source_names[source_id]
                if selected_source_name is None:
                    selected_source_name = 'unknown'
                for source_name in source_names:
                    self._record_counter(
                        'source_choice_selected_source',
                        source_name,
                        selected_source_name == source_name,
                    )
                if selected_source_name == 'unknown':
                    self._record_counter(
                        'source_choice_selected_source',
                        selected_source_name,
                        1.0,
                    )

                if source_ious:
                    for threshold in self.thresholds:
                        default_ok = bool((default_iou > threshold).item())
                        selected_ok = bool((iou > threshold).item())
                        oracle_ok = bool((oracle_iou > threshold).item())
                        self._record_effect(
                            threshold,
                            'selector_fix',
                            (not default_ok) and selected_ok,
                        )
                        self._record_effect(
                            threshold,
                            'selector_break',
                            default_ok and (not selected_ok),
                        )
                        self._record_effect(
                            threshold,
                            'selector_kept_correct',
                            default_ok and selected_ok,
                        )
                        self._record_effect(
                            threshold,
                            'selector_kept_wrong',
                            (not default_ok) and (not selected_ok),
                        )
                        self._record_effect(
                            threshold,
                            'oracle_headroom',
                            (not default_ok) and oracle_ok,
                        )
                        if gate_oracle_iou is not None:
                            gate_oracle_ok = bool(
                                (gate_oracle_iou > threshold).item()
                            )
                            self._record_effect(
                                threshold,
                                'gate_oracle_headroom',
                                (not default_ok) and gate_oracle_ok,
                            )
                        if candidate_iou is not None:
                            candidate_ok = bool(
                                (candidate_iou > threshold).item()
                            )
                            self._record_effect(
                                threshold,
                                'candidate_fix',
                                (not default_ok) and candidate_ok,
                            )
                            self._record_effect(
                                threshold,
                                'candidate_break',
                                default_ok and (not candidate_ok),
                            )
    
    # BRIEF position alignment
    def evaluate_bbox_by_pos_align(self, end_points, prefix):
        """
        Evaluate bounding box IoU by position alignment

        Args:
            end_points (dict): contains predictions and gt
            prefix (str): layer name
        """
        # step get the position label and GT box 
        positive_map, modify_positive_map, pron_positive_map, other_entity_map, \
            auxi_entity_positive_map, rel_positive_map, gt_bboxes = self._parse_gt(end_points)    
        
        # Parse predictions
        sem_scores = end_points[f'{prefix}sem_cls_scores'].softmax(-1)

        if sem_scores.shape[-1] != positive_map.shape[-1]:
            sem_scores_ = torch.zeros(
                sem_scores.shape[0], sem_scores.shape[1],
                positive_map.shape[-1]).to(sem_scores.device)
            sem_scores_[:, :, :sem_scores.shape[-1]] = sem_scores
            sem_scores = sem_scores_

        # Parse predictions
        pred_center = end_points[f'{prefix}center']  # B, Q=256, 3
        pred_size = end_points[f'{prefix}pred_size']  # (B,Q,3) (l,w,h)
        assert (pred_size < 0).sum() == 0
        pred_bbox = torch.cat([pred_center, pred_size], dim=-1) # ([B, 256, 6])
        candidate_boxes, candidate_scores, candidate_valid, axis_mode = (
            self._resolve_position_candidates(
                end_points, prefix, pred_bbox
            )
        )

        # Highest scoring box -> iou
        for bid in range(len(positive_map)):
            # Keep scores for annotated objects only
            num_obj = int(end_points['box_label_mask'][bid].sum())
            pmap = positive_map[bid, :num_obj]
            scores_main = (
                sem_scores[bid].unsqueeze(0)    
                * pmap.unsqueeze(1)             
            ).sum(-1)

            # score
            pmap_modi = modify_positive_map[bid, :1]
            pmap_pron = pron_positive_map[bid, :1]
            pmap_other = other_entity_map[bid, :1]
            pmap_rel = rel_positive_map[bid, :1]    # num_obj
            scores_modi = (
                sem_scores[bid].unsqueeze(0)    
                * pmap_modi.unsqueeze(1)             
            ).sum(-1)
            scores_pron = (
                sem_scores[bid].unsqueeze(0)    
                * pmap_pron.unsqueeze(1)             
            ).sum(-1)
            scores_other = (
                sem_scores[bid].unsqueeze(0)    
                * pmap_other.unsqueeze(1)             
            ).sum(-1)
            scores_rel = (
                sem_scores[bid].unsqueeze(0)    
                * pmap_rel.unsqueeze(1)             
            ).sum(-1)

            scores = scores_main + scores_modi + scores_pron + scores_rel - scores_other
            if candidate_scores is not None:
                scores = candidate_scores[bid].unsqueeze(0)
            elif (
                    self.eval_use_selector_choice_scores
                    and prefix == 'last_'
                    and 'selected_source_scores' in end_points):
                scores = end_points['selected_source_scores'][bid].unsqueeze(0)

            if (not isinstance(scores, torch.Tensor) or scores.dim() != 2
                    or scores.shape[1] != candidate_boxes.shape[1]
                    or scores.device != candidate_boxes.device
                    or not torch.is_floating_point(scores)):
                raise ValueError(
                    'position scores must align with the active candidate axis'
                )
            row_valid = candidate_valid[bid].clone()
            if self.filter_non_gt_boxes:
                detected_boxes = end_points['all_detected_boxes'][bid][
                    end_points['all_detected_bbox_label_mask'][bid].bool()
                ]
                surviving_filter = torch.zeros_like(row_valid)
                valid_indices = row_valid.nonzero(
                    as_tuple=False
                ).reshape(-1)
                if detected_boxes.numel() and valid_indices.numel():
                    detector_ious, _ = _iou3d_par(
                        box_cxcyczwhd_to_xyzxyz(detected_boxes),
                        box_cxcyczwhd_to_xyzxyz(
                            candidate_boxes[bid, valid_indices]
                        ),
                    )
                    surviving_filter[valid_indices] = (
                        detector_ious.max(0)[0] > 0.25
                    )
                row_valid &= surviving_filter
            if not bool(row_valid.any().item()):
                raise ValueError(
                    'position filtering left no valid candidate'
                )
            ranking_valid = row_valid.unsqueeze(0).expand_as(scores)

            top = self._position_top_indices(
                scores, ranking_valid, axis_mode, max(self.topks)
            )
            pbox = candidate_boxes[bid, top.reshape(-1)]

            ious, _ = _iou3d_par(
                box_cxcyczwhd_to_xyzxyz(gt_bboxes[bid][:num_obj]),  # (obj, 6)
                box_cxcyczwhd_to_xyzxyz(pbox)  # (obj*10, 6)
            )  # (obj, obj*10)
            ious = ious.reshape(top.size(0), top.size(0), top.size(1))
            diagonal = torch.arange(len(ious), device=ious.device)
            ious = ious[diagonal, diagonal]   # ([1, 10])

            # step Measure IoU>threshold, ious are (obj, 10)
            topks = self.topks
            for t in self.thresholds:
                thresholded = ious > t
                for k in topks:
                    found = thresholded[:, :k].any(1)
                    self.dets[(prefix, t, k, 'bbs')] += found.sum().item()
                    self.gts[(prefix, t, k, 'bbs')] += len(thresholded)
                    if prefix == 'last_' and k == 1 and self.only_root:
                        self._record_position_subgroups(
                            end_points, bid, t, found
                        )

    # BRIEF semantic alignment
    def evaluate_bbox_by_sem_align(self, end_points, prefix):
        """
        Evaluate bounding box IoU by semantic alignment.

        Args:
            end_points (dict): contains predictions and gt
            prefix (str): layer name
        """
        # step get the position label and GT box 
        positive_map, modify_positive_map, pron_positive_map, other_entity_map, \
            auxi_entity_positive_map, rel_positive_map, gt_bboxes = self._parse_gt(end_points)    
        
        # Parse predictions
        pred_center = end_points[f'{prefix}center']  # B, Q, 3
        pred_size = end_points[f'{prefix}pred_size']  # (B,Q,3) (l,w,h)

        assert (pred_size < 0).sum() == 0
        pred_bbox = torch.cat([pred_center, pred_size], dim=-1)
        
        # step compute similarity between vision and text
        proj_tokens = end_points['proj_tokens']             # text feature   (B, 256, 64)
        proj_queries = end_points[f'{prefix}proj_queries']  # vision feature (B, 256, 64)
        sem_scores = torch.matmul(proj_queries, proj_tokens.transpose(-1, -2))  # similarity ([B, 256, L]) 
        sem_scores_ = (sem_scores / 0.07).softmax(-1)                           # softmax ([B, 256, L])
        sem_scores = torch.zeros(sem_scores_.size(0), sem_scores_.size(1), 256) # ([B, 256, 256])
        sem_scores = sem_scores.to(sem_scores_.device)
        sem_scores[:, :sem_scores_.size(1), :sem_scores_.size(2)] = sem_scores_ # ([B, P=256, L=256])

        # Highest scoring box -> iou
        for bid in range(len(positive_map)):
            is_correct = None
            if self.filter_non_gt_boxes:  # this works only for the target box
                ious, _ = _iou3d_par(
                    box_cxcyczwhd_to_xyzxyz(
                        end_points['all_detected_boxes'][bid][
                            end_points['all_detected_bbox_label_mask'][bid]
                        ]
                    ),  # (gt, 6)
                    box_cxcyczwhd_to_xyzxyz(pred_bbox[bid])  # (Q, 6)
                )  # (gt, Q)
                is_correct = (ious.max(0)[0] > 0.25) * 1.0
            
            # Keep scores for annotated objects only
            num_obj = int(end_points['box_label_mask'][bid].sum())
            pmap = positive_map[bid, :num_obj]
            scores_main = (
                sem_scores[bid].unsqueeze(0)  # (1, Q, 256)
                * pmap.unsqueeze(1)  # (obj, 1, 256)
            ).sum(-1)  # (obj, Q)
            
            # score
            pmap_modi = modify_positive_map[bid, :1]
            pmap_pron = pron_positive_map[bid, :1]
            pmap_other = other_entity_map[bid, :1]
            pmap_auxi = auxi_entity_positive_map[bid, :1]
            pmap_rel = rel_positive_map[bid, :1]
            scores_modi = (
                sem_scores[bid].unsqueeze(0)    
                * pmap_modi.unsqueeze(1)             
            ).sum(-1)
            scores_pron = (
                sem_scores[bid].unsqueeze(0)    
                * pmap_pron.unsqueeze(1)             
            ).sum(-1)
            scores_other = (
                sem_scores[bid].unsqueeze(0)    
                * pmap_other.unsqueeze(1)             
            ).sum(-1)
            scores_auxi = (
                sem_scores[bid].unsqueeze(0)    
                * pmap_auxi.unsqueeze(1)             
            ).sum(-1)
            scores_rel = (
                sem_scores[bid].unsqueeze(0)    
                * pmap_rel.unsqueeze(1)             
            ).sum(-1)

            # total score
            scores = scores_main + scores_modi + scores_pron + scores_rel - scores_other

            if is_correct is not None:
                scores = scores * is_correct[None]

            # 10 predictions per gt box
            top = scores.argsort(1, True)[:, :10]  # (obj, 10)
            pbox = pred_bbox[bid, top.reshape(-1)]

            # IoU
            ious, _ = _iou3d_par(
                box_cxcyczwhd_to_xyzxyz(gt_bboxes[bid][:num_obj]),  # (obj, 6)
                box_cxcyczwhd_to_xyzxyz(pbox)  # (obj*10, 6)
            )  # (obj, obj*10)
            ious = ious.reshape(top.size(0), top.size(0), top.size(1))
            ious = ious[torch.arange(len(ious)), torch.arange(len(ious))]

            # Check for bad cases
            if self.bad_case_visualization:
                wandb = _load_wandb()
                wandb.init(project="vis", name="badcase")
                bad_cases = ious < self.bad_case_threshold  # Here you set your bad_case_threshold
                if bad_cases.any():
                    # Get point cloud and original color
                    point_cloud = end_points['point_clouds'][bid]
                    og_color = end_points['og_color'][bid]
                    point_cloud[:, 3:] = (og_color + torch.tensor([109.8, 97.2, 83.8]).cuda() / 256) * 256
                    target_name = end_points['target_name'][bid]
                    utterances = end_points['utterances'][bid]

                    # Get all boxes and predicted boxes
                    topk_boxes = 0
                    all_bboxes = end_points['all_bboxes'][bid].cpu()
                    pbox_bad_cases = pbox[bad_cases[0]].cpu()[topk_boxes].unsqueeze(0)  # top 1
                    gt_box = gt_bboxes[bid].cpu()

                    # Convert boxes to points for visualization
                    all_boxes_points = box2points(all_bboxes[..., :6])  # all boxes
                    gt_box = box2points(gt_box[..., :6])  # gt boxes
                    pbox_bad_cases_points = box2points(pbox_bad_cases[..., :6])

                    # Log bad case visualization to wandb
                    wandb.log({
                        "bad_case_point_scene": wandb.Object3D({
                            "type": "lidar/beta",
                            "points": point_cloud,
                            "boxes": np.array(
                                [  # actual boxes
                                    {
                                        "corners": c.tolist(),
                                        "label": "actual",
                                        "color": [0, 255, 0]
                                    }
                                    for c in gt_box
                                ]
                                + [  # predicted boxes
                                    {
                                        "corners": c.tolist(),
                                        "label": "predicted",
                                        "color": [255, 0, 0]
                                    }
                                    for c in pbox_bad_cases_points
                                ]
                            )
                        }),
                        "target_name": wandb.Html(target_name),
                        "utterance": wandb.Html(utterances),
                    })

            # Check for kps points
            if self.kps_points_visualization:
                wandb = _load_wandb()
                wandb.init(project="vis", name="kps_points")
                point_cloud = end_points['point_clouds'][bid]
                og_color = end_points['og_color'][bid]
                point_cloud[:, 3:] = (og_color + torch.tensor([109.8, 97.2, 83.8]).cuda() / 256) * 256
                kps_points = end_points['query_points_xyz'][bid]
                red = torch.zeros((256, 3)).cuda()
                red[:, 0] = 255.0
                kps_points = torch.cat([kps_points, red], dim=1)
                total_point = torch.cat([point_cloud, kps_points], dim=0)
                utterances = end_points['utterances'][bid]
                gt_box = gt_bboxes[bid].cpu()
                gt_box = box2points(gt_box[..., :6])

                wandb.log({
                        "kps_point_scene": wandb.Object3D({
                            "type": "lidar/beta",
                            "points": total_point,
                            "boxes": np.array(
                                [
                                    {
                                        "corners": c.tolist(),
                                        "label": "target",
                                        "color": [0, 255, 0]
                                    }
                                    for c in gt_box
                                ]
                            )
                        }),
                        "utterance": wandb.Html(utterances),
                    })

            # step Measure IoU>threshold, ious are (obj, 10)
            for t in self.thresholds:
                thresholded = ious > t
                for k in self.topks:
                    found = thresholded[:, :k].any(1)
                    self.dets[(prefix, t, k, 'bbf')] += found.sum().item()
                    self.gts[(prefix, t, k, 'bbf')] += len(thresholded)
                    if prefix == 'last_':
                        found = found[0].item()
                        if k == 1 and t == self.thresholds[0]:
                            if end_points['is_view_dep'][bid]:
                                self.gts['vd'] += 1
                                self.dets['vd'] += found
                            else:
                                self.gts['vid'] += 1
                                self.dets['vid'] += found
                            if end_points['is_hard'][bid]:
                                self.gts['hard'] += 1
                                self.dets['hard'] += found
                            else:
                                self.gts['easy'] += 1
                                self.dets['easy'] += found
                            if end_points['is_unique'][bid]:
                                self.gts['unique'] += 1
                                self.dets['unique'] += found
                            else:
                                self.gts['multi'] += 1
                                self.dets['multi'] += found
                        if k == 1 and t == self.thresholds[1]:
                            if end_points['is_view_dep'][bid]:
                                self.gts['vd50'] += 1
                                self.dets['vd50'] += found
                            else:
                                self.gts['vid50'] += 1
                                self.dets['vid50'] += found
                            if end_points['is_hard'][bid]:
                                self.gts['hard50'] += 1
                                self.dets['hard50'] += found
                            else:
                                self.gts['easy50'] += 1
                                self.dets['easy50'] += found
                            if end_points['is_unique'][bid]:
                                self.gts['unique50'] += 1
                                self.dets['unique50'] += found
                            else:
                                self.gts['multi50'] += 1
                                self.dets['multi50'] += found


    # BRIEF Get the postion label of the decoupled text component.
    def _parse_gt(self, end_points):
        positive_map = torch.clone(end_points['positive_map'])                  # main
        modify_positive_map = torch.clone(end_points['modify_positive_map'])    # attribute
        pron_positive_map = torch.clone(end_points['pron_positive_map'])        # pron
        other_entity_map = torch.clone(end_points['other_entity_map'])          # other(including auxi)
        auxi_entity_positive_map = torch.clone(end_points['auxi_entity_positive_map'])  # auxi
        rel_positive_map = torch.clone(end_points['rel_positive_map'])

        positive_map[positive_map > 0] = 1                      
        gt_center = end_points['center_label'][:, :, 0:3]       
        gt_size = end_points['size_gts']                        
        gt_bboxes = torch.cat([gt_center, gt_size], dim=-1)     # GT box cxcyczwhd
        
        if self.only_root:
            positive_map = positive_map[:, :1]  # (B, 1, 256)
            gt_bboxes = gt_bboxes[:, :1]        # (B, 1, 6)
        
        return positive_map, modify_positive_map, pron_positive_map, other_entity_map, auxi_entity_positive_map, \
            rel_positive_map, gt_bboxes
    

    # BRIEF position alignment
    def evaluate_masks_by_pos_align(self, end_points, prefix):
        """
        Evaluate masks IoU by position alignment

        Args:
            end_points (dict): contains predictions and gt
            prefix (str): layer name
        """
        # step get the position label and GT box 
        positive_map, modify_positive_map, pron_positive_map, other_entity_map, \
            auxi_entity_positive_map, rel_positive_map, gt_masks = self._parse_gt_mask(end_points)    
        
        # Parse predictions
        sem_scores = end_points[f'{prefix}sem_cls_scores'].softmax(-1)  # [B, 256 256]

        if sem_scores.shape[-1] != positive_map.shape[-1]:
            sem_scores_ = torch.zeros(
                sem_scores.shape[0], sem_scores.shape[1],
                positive_map.shape[-1]).to(sem_scores.device)
            sem_scores_[:, :, :sem_scores.shape[-1]] = sem_scores
            sem_scores = sem_scores_

        pred_masks, joint_mask_policy = self._build_mask_point_predictions(
            end_points, prefix
        )
        joint_parent_queries = (
            joint_mask_policy['parent_queries']
            if joint_mask_policy is not None else None
        )
        learned_mask_queries = self._resolve_learned_mask_queries(
            end_points,
            prefix,
            pred_masks.shape[1],
            pred_masks.device,
        )

        # Highest scoring box -> iou
        for bid in range(len(positive_map)):
            is_correct = None
            
            # Keep scores for annotated objects only
            num_obj = int(end_points['box_label_mask'][bid].sum())
            pmap = positive_map[bid, :num_obj]
            scores_main = (
                sem_scores[bid].unsqueeze(0)    
                * pmap.unsqueeze(1)             
            ).sum(-1)

            # score
            pmap_modi = modify_positive_map[bid, :1]
            pmap_pron = pron_positive_map[bid, :1]
            pmap_other = other_entity_map[bid, :1]
            pmap_rel = rel_positive_map[bid, :1]    # num_obj
            scores_modi = (
                sem_scores[bid].unsqueeze(0)    
                * pmap_modi.unsqueeze(1)             
            ).sum(-1)
            scores_pron = (
                sem_scores[bid].unsqueeze(0)    
                * pmap_pron.unsqueeze(1)             
            ).sum(-1)
            scores_other = (
                sem_scores[bid].unsqueeze(0)    
                * pmap_other.unsqueeze(1)             
            ).sum(-1)
            scores_rel = (
                sem_scores[bid].unsqueeze(0)    
                * pmap_rel.unsqueeze(1)             
            ).sum(-1)

            scores = scores_main + scores_modi + scores_pron + scores_rel - scores_other  # [1, 256]

            if is_correct is not None:
                scores = scores * is_correct[None]

            if joint_parent_queries is not None:
                top = joint_parent_queries[bid].reshape(1, 1)
            elif learned_mask_queries is not None:
                top = learned_mask_queries[bid].reshape(1, 1)
            else:
                top = scores.argsort(1, True)[:, :1]  # top-1 mask
            pmasks = pred_masks[bid, top.reshape(-1)]

            # comupte mask iou
            iou_score_pos = self.calculate_masks_iou(pmasks, gt_masks[bid])
            # print("{:.14f}".format(iou_score_pos))
            self.gts['mask_pos'] += 1
            self.dets['mask_pos'] += iou_score_pos


    # BRIEF semantic alignment
    def evaluate_masks_by_sem_align(self, end_points, prefix):
        """
        Evaluate masks IoU by semantic alignment.

        Args:
            end_points (dict): contains predictions and gt
            prefix (str): layer name
        """
        # step get the position label and GT box 
        positive_map, modify_positive_map, pron_positive_map, other_entity_map, \
            auxi_entity_positive_map, rel_positive_map, gt_masks = self._parse_gt_mask(end_points)    
        
        pred_masks, joint_mask_policy = self._build_mask_point_predictions(
            end_points, prefix
        )
        joint_parent_queries = (
            joint_mask_policy['parent_queries']
            if joint_mask_policy is not None else None
        )
        learned_mask_queries = self._resolve_learned_mask_queries(
            end_points,
            prefix,
            pred_masks.shape[1],
            pred_masks.device,
        )

        
        # step compute similarity between vision and text
        proj_tokens = end_points['proj_tokens']             # text feature   (B, 256, 64)
        proj_queries = end_points[f'{prefix}proj_queries']  # vision feature (B, 256, 64)
        sem_scores = torch.matmul(proj_queries, proj_tokens.transpose(-1, -2))  # similarity ([B, 256, L]) 
        sem_scores_ = (sem_scores / 0.07).softmax(-1)                           # softmax ([B, 256, L])
        sem_scores = torch.zeros(sem_scores_.size(0), sem_scores_.size(1), 256) # ([B, 256, 256])
        sem_scores = sem_scores.to(sem_scores_.device)
        sem_scores[:, :sem_scores_.size(1), :sem_scores_.size(2)] = sem_scores_ # ([B, P=256, L=256])

        # Highest scoring box -> iou
        for bid in range(len(positive_map)):
            is_correct = None
            
            # Keep scores for annotated objects only
            num_obj = int(end_points['box_label_mask'][bid].sum())
            pmap = positive_map[bid, :num_obj]
            scores_main = (
                sem_scores[bid].unsqueeze(0)  # (1, Q, 256)
                * pmap.unsqueeze(1)  # (obj, 1, 256)
            ).sum(-1)  # (obj, Q)
            
            # score
            pmap_modi = modify_positive_map[bid, :1]
            pmap_pron = pron_positive_map[bid, :1]
            pmap_other = other_entity_map[bid, :1]
            pmap_auxi = auxi_entity_positive_map[bid, :1]
            pmap_rel = rel_positive_map[bid, :1]
            scores_modi = (
                sem_scores[bid].unsqueeze(0)    
                * pmap_modi.unsqueeze(1)             
            ).sum(-1)
            scores_pron = (
                sem_scores[bid].unsqueeze(0)    
                * pmap_pron.unsqueeze(1)             
            ).sum(-1)
            scores_other = (
                sem_scores[bid].unsqueeze(0)    
                * pmap_other.unsqueeze(1)             
            ).sum(-1)
            scores_auxi = (
                sem_scores[bid].unsqueeze(0)    
                * pmap_auxi.unsqueeze(1)             
            ).sum(-1)
            scores_rel = (
                sem_scores[bid].unsqueeze(0)    
                * pmap_rel.unsqueeze(1)             
            ).sum(-1)

            # total score
            scores = scores_main + scores_modi + scores_pron + scores_rel - scores_other

            if is_correct is not None:
                scores = scores * is_correct[None]

            if joint_parent_queries is not None:
                top = joint_parent_queries[bid].reshape(1, 1)
            elif learned_mask_queries is not None:
                top = learned_mask_queries[bid].reshape(1, 1)
            else:
                top = scores.argsort(1, True)[:, :1]
            pmasks = pred_masks[bid, top.reshape(-1)]

            # compute IoU
            iou_score_sem = self.calculate_masks_iou(pmasks, gt_masks[bid])
            # print("{:.14f}".format(iou_score_sem))
            self.gts['mask_sem'] += 1
            self.dets['mask_sem'] += iou_score_sem

            if end_points['is_view_dep'][bid]:
                self.gts['vd_num'] += 1
            else:
                self.gts['vid_num'] += 1
            if end_points['is_unique'][bid]:
                self.gts['unique_num'] += 1
            else:
                self.gts['multi_num'] += 1
            if end_points['is_hard'][bid]:
                self.gts['hard_num'] += 1
            else:
                self.gts['easy_num'] += 1

            if iou_score_sem > 0.25:
                self.dets['overall_mask'] += 1
                if end_points['is_view_dep'][bid]:
                    self.dets['vd_mask'] += 1
                else:
                    self.dets['vid_mask'] += 1
                if end_points['is_hard'][bid]:
                    self.dets['hard_mask'] += 1
                else:
                    self.dets['easy_mask'] += 1
                if end_points['is_unique'][bid]:
                    self.dets['unique_mask'] += 1
                else:
                    self.dets['multi_mask'] += 1
            if iou_score_sem > 0.5:
                self.dets['overall50_mask'] += 1
                if end_points['is_view_dep'][bid]:
                    self.dets['vd50_mask'] += 1
                else:
                    self.dets['vid50_mask'] += 1
                if end_points['is_hard'][bid]:
                    self.dets['hard50_mask'] += 1
                else:
                    self.dets['easy50_mask'] += 1
                if end_points['is_unique'][bid]:
                    self.dets['unique50_mask'] += 1
                else:
                    self.dets['multi50_mask'] += 1

            # visualization for pres mask and box
            if self.visualization_pred:
                wandb = _load_wandb()
                wandb.init(project="vis", name="pred")
                point_cloud = end_points['point_clouds'][bid]
                og_color = end_points['og_color'][bid]
                point_cloud[:, 3:] = (og_color + torch.tensor([109.8, 97.2, 83.8]).cuda() / 256) * 256
                red = torch.tensor([255.0, 0.0, 0.0]).cuda()

                pred_center = end_points[f'{prefix}center'][bid]
                pred_size = end_points[f'{prefix}pred_size'][bid]
                pred_bbox = torch.cat([pred_center, pred_size], dim=-1).cpu()[top.reshape(-1)]

                utterances = end_points['utterances'][bid]
                pred_bbox = box2points(pred_bbox[..., :6])

                mask_idx = pmasks[0] == 1
                pred_cloud = point_cloud
                pred_cloud[mask_idx, 3:] = red

                wandb.log({
                        "point_scene": wandb.Object3D({
                            "type": "lidar/beta",
                            "points": pred_cloud,
                            "boxes": np.array(
                                [ 
                                    {
                                        "corners": c.tolist(),
                                        "label": "predicted",
                                        "color": [0, 0, 255]
                                    }
                                    for c in pred_bbox
                                ]
                            )
                        }),
                        "utterance": wandb.Html(utterances),
                    })
                

            # visualization for gt mask and box
            if self.visualization_gt:
                wandb = _load_wandb()
                wandb.init(project="vis", name="gt")
                point_cloud = end_points['point_clouds'][bid]
                og_color = end_points['og_color'][bid]
                point_cloud[:, 3:] = (og_color + torch.tensor([109.8, 97.2, 83.8]).cuda() / 256) * 256
                blue = torch.tensor([0.0, 0.0, 255.0]).cuda()

                gt_center = end_points['center_label'][bid, :, 0:3]       
                gt_size = end_points['size_gts'][bid]                        
                gt_box = torch.cat([gt_center, gt_size], dim=-1).cpu()

                utterances = end_points['utterances'][bid]
                gt_box = box2points(gt_box[..., :6])

                gt_cloud = point_cloud
                gt_mask_idx = gt_masks[bid][0] == 1
                gt_cloud[gt_mask_idx, 3:] = blue

                wandb.log({
                        "point_scene": wandb.Object3D({
                            "type": "lidar/beta",
                            "points": gt_cloud,
                            "boxes": np.array(
                                [
                                    {
                                        "corners": c.tolist(),
                                        "label": "target",
                                        "color": [0, 255, 0]
                                    }
                                    for c in gt_box
                                ]
                            )
                        }),
                        "utterance": wandb.Html(utterances),
                    })
            

    # BRIEF Get the postion label of the decoupled text component.
    def _parse_gt_mask(self, end_points):
        positive_map = torch.clone(end_points['positive_map'])                  # main
        modify_positive_map = torch.clone(end_points['modify_positive_map'])    # attribute
        pron_positive_map = torch.clone(end_points['pron_positive_map'])        # pron
        other_entity_map = torch.clone(end_points['other_entity_map'])          # other(including auxi)
        auxi_entity_positive_map = torch.clone(end_points['auxi_entity_positive_map'])  # auxi
        rel_positive_map = torch.clone(end_points['rel_positive_map'])

        positive_map[positive_map > 0] = 1                      
        gt_masks = end_points['gt_masks']   
        
        if self.only_root:
            positive_map = positive_map[:, :1]  # (B, 1, 256)
            gt_masks = gt_masks[:, :1]        # (B, 1, 50000)
        
        return positive_map, modify_positive_map, pron_positive_map, other_entity_map, auxi_entity_positive_map, \
            rel_positive_map, gt_masks
    
    def calculate_masks_iou(self, mask1, mask2):
        mask1, mask2 = mask1.cpu().numpy(), mask2.cpu().numpy()
        intersection = np.logical_and(mask1, mask2)
        union = np.logical_or(mask1, mask2)
        iou_score = np.sum(intersection) / np.sum(union)
        return iou_score
