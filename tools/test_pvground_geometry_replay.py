"""CPU scientific-contract checks against the actual deployed evaluator.

Run from a deployed code directory with this tools directory on PYTHONPATH.
The cases intentionally separate box quality from geometry-based filtering.
"""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from pvground_geometry_replay import GeometryReplay, check_recorded_metrics, main
from src.grounding_evaluator import GroundingEvaluator


class GeometryReplayTests(unittest.TestCase):
    def setUp(self):
        positive = torch.zeros(1, 1, 256)
        positive[0, 0, 0] = 1
        self.points = dict(
            last_center=torch.tensor([[[10., 0., 0.], [20., 0., 0.]]]),
            last_pred_size=torch.full((1, 2, 3), 2.),
            native_coarse_center=torch.tensor([[[0., 0., 0.], [20., 0., 0.]]]),
            native_coarse_size=torch.full((1, 2, 3), 2.),
            span_mask_center=torch.tensor([[[10., 0., 0.], [0., 0., 0.]]]),
            span_mask_size=torch.full((1, 2, 3), 2.),
            last_proj_queries=torch.eye(2).unsqueeze(0),
            proj_tokens=torch.eye(2).unsqueeze(0),
            positive_map=positive,
            center_label=torch.zeros(1, 1, 3),
            size_gts=torch.full((1, 1, 3), 2.),
            box_label_mask=torch.ones(1, 1, dtype=torch.bool),
            all_detected_boxes=torch.tensor([[[0., 0., 0., 2., 2., 2.]]]),
            all_detected_bbox_label_mask=torch.ones(1, 1, dtype=torch.bool),
            is_view_dep=torch.tensor([False]),
            is_hard=torch.tensor([False]),
            is_unique=torch.tensor([True]),
            span_axis_gate=torch.tensor([[[0., .5, 1.], [0., .5, 1.]]]),
            span_raw_axis_gate=torch.tensor([[[-1., .5, 2.], [-1., .5, 2.]]]),
        )
        for name in ("modify_positive_map", "pron_positive_map", "other_entity_map",
                     "auxi_entity_positive_map", "rel_positive_map"):
            self.points[name] = torch.zeros_like(positive)
        self.evaluator = GroundingEvaluator(
            only_root=True, thresholds=[.25, .5], topks=[1, 5, 10],
            prefixes=["last_"], filter_non_gt_boxes=True, model="PVGround")
        self.evaluator.evaluate_bbox_by_sem_align(self.points, "last_")

    def test_geometry_filter_changes_candidate_and_input_is_preserved(self):
        before = {name: value.clone() for name, value in self.points.items()}
        replay = GeometryReplay(self.evaluator)
        replay.observe(self.points)
        report = replay.report(expected_rows=1)
        hits = {(row["protocol"], row["geometry"]): (row["hits025"], row["hits050"])
                for row in report["rows"]}
        self.assertEqual(hits["official_filter", "final"], (0, 0))
        self.assertEqual(hits["official_filter", "native"], (1, 1))
        self.assertEqual(hits["official_filter", "mask"], (1, 1))
        self.assertEqual(hits["unfiltered_fixed_queries", "native"], (1, 1))
        self.assertEqual(hits["unfiltered_fixed_queries", "mask"], (0, 0))
        self.assertEqual(report["gates"]["mean"], .5)
        self.assertAlmostEqual(report["gates"]["raw_below_zero_fraction"], 1 / 3)
        self.assertAlmostEqual(report["gates"]["raw_above_one_fraction"], 1 / 3)
        for name, value in self.points.items():
            self.assertTrue(torch.equal(value, before[name]), name)

    def test_incomplete_split_and_formal_metric_disagreement_are_rejected(self):
        replay = GeometryReplay(self.evaluator)
        replay.observe(self.points)
        with self.assertRaises(AssertionError):
            replay.report(expected_rows=2)
        self.evaluator.dets[("last_", .25, 1, "bbf")] += 1
        with self.assertRaises(AssertionError):
            replay.report(expected_rows=1)

    def test_official_filter_disabled_matches_unfiltered_protocol(self):
        evaluator = GroundingEvaluator(
            only_root=True, thresholds=[.25, .5], topks=[1, 5, 10],
            prefixes=["last_"], filter_non_gt_boxes=False, model="PVGround")
        evaluator.evaluate_bbox_by_sem_align(self.points, "last_")
        replay = GeometryReplay(evaluator)
        replay.observe(self.points)
        report = replay.report(expected_rows=1)
        self.assertFalse(report["official_filter_enabled"])
        hits = {(row["protocol"], row["geometry"]): (row["hits025"], row["hits050"])
                for row in report["rows"]}
        for geometry in ("final", "native", "mask"):
            self.assertEqual(hits["official_filter", geometry],
                             hits["unfiltered_fixed_queries", geometry])

    def test_checkpoint_epoch_wins_over_retained_best_and_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "latest.pth"
            metrics = root / "native_metrics.jsonl"
            current = dict(epoch=8, rows=7899, hits025=4768, hits050=3925,
                           primary_score="last/bbf", same_complete_model=True)
            retained = dict(current, epoch=0, hits025=4739, hits050=4132)
            torch.save(dict(epoch=8, retained_metrics=retained), checkpoint)
            metrics.write_text(json.dumps(retained) + "\n" + json.dumps(current) + "\n")
            reference = check_recorded_metrics(metrics, checkpoint, 7899, current)
            self.assertEqual(reference["epoch"], 8)
            self.assertEqual(reference["hits050"], 3925)
            with self.assertRaises(AssertionError):
                check_recorded_metrics(metrics, checkpoint, 7899, retained)

    def test_training_mode_rejected_before_creating_output(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "result.json"
            argv = ["replay", "--replay-output", str(output),
                    "--reference-metrics", str(Path(directory) / "native_metrics.jsonl")]
            with patch("sys.argv", argv), self.assertRaisesRegex(AssertionError, "--eval"):
                main()
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
