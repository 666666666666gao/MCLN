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
            last_sem_cls_scores=torch.zeros(1, 2, 256),
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
        self.points["last_sem_cls_scores"][0, 1, 0] = 5
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
        native = next(row for row in report["paired_transitions"]
                      if row["protocol"] == "official_filter"
                      and row["geometry"] == "native" and row["threshold"] == .25)
        self.assertEqual((native["repaired"], native["damaged"]), (1, 0))
        self.assertAlmostEqual(report["gates"]["raw_below_zero_fraction"], 1 / 3)
        self.assertAlmostEqual(report["gates"]["raw_above_one_fraction"], 1 / 3)
        for name, value in self.points.items():
            self.assertTrue(torch.equal(value, before[name]), name)

    def test_actual_query_trace_distinguishes_bbs_semantic_and_geometry(self):
        replay = GeometryReplay(self.evaluator)
        replay.observe(self.points)
        report = replay.report(expected_rows=1)
        self.assertEqual(replay.bbs_queries, [1])
        self.assertEqual(
            replay.evaluators["unfiltered_fixed_queries", "final"]._replay_root_queries,
            [0])
        self.assertEqual(
            replay.evaluators["official_filter", "mask"]._replay_root_queries, [1])
        agreement = next(row for row in report["query_selection_agreement"]
                         if row["protocol"] == "unfiltered_fixed_queries"
                         and row["geometry"] == "final"
                         and row["reference"] == "extra_mask_bbs")
        self.assertEqual((agreement["agreements"], agreement["disagreements"]), (0, 1))
        replay.evaluators["official_filter", "mask"]._replay_root_queries.pop()
        with self.assertRaises(AssertionError):
            replay.report(expected_rows=1)

    def test_tied_sort_result_is_observed_without_changing_metrics(self):
        points = dict(self.points)
        points["last_proj_queries"] = torch.zeros_like(points["last_proj_queries"])
        evaluator = GroundingEvaluator(
            only_root=True, thresholds=[.25, .5], topks=[1, 5, 10],
            prefixes=["last_"], filter_non_gt_boxes=False, model="PVGround")
        evaluator.evaluate_bbox_by_sem_align(points, "last_")
        replay = GeometryReplay(evaluator)
        replay.observe(points)
        replay.report(expected_rows=1)
        expected = torch.ones(1, 2).argsort(1, True)[0, 0].item()
        for (protocol, geometry), observed in replay.evaluators.items():
            self.assertEqual(observed._replay_root_queries, [expected])
            if geometry == "final":
                self.assertEqual(dict(observed.dets), dict(evaluator.dets))

    def test_selected_gate_population_uses_actual_formal_query(self):
        points = dict(self.points)
        points["span_raw_axis_gate"] = torch.tensor([[[-1., .25, .5], [2., 2., 2.]]])
        points["span_axis_gate"] = points["span_raw_axis_gate"].clamp(0, 1)
        replay = GeometryReplay(self.evaluator)
        replay.observe(points)
        report = replay.report(expected_rows=1)
        self.assertEqual(report["selected_gates"]["count"], 3)
        self.assertEqual(report["selected_gates"]["mean"], .25)
        self.assertEqual(report["gates"]["mean"], .625)
        self.assertAlmostEqual(report["selected_gates"]["raw_below_zero_fraction"], 1 / 3)
        self.assertEqual(report["selected_gates"]["raw_above_one_fraction"], 0)
        self.assertEqual(report["selected_gates"]["population"],
                         "official_final_root_query_axes")

    def test_equal_net_hits_preserve_both_repairs_and_damage_across_batches(self):
        replay = GeometryReplay(self.evaluator)
        replay.observe(self.points)
        second = dict(self.points)
        second["last_center"] = self.points["native_coarse_center"]
        second["native_coarse_center"] = self.points["last_center"]
        second["last_sem_cls_scores"] = torch.zeros_like(self.points["last_sem_cls_scores"])
        second["last_sem_cls_scores"][0, 0, 0] = 5
        self.evaluator.evaluate_bbox_by_sem_align(second, "last_")
        replay.observe(second)
        report = replay.report(expected_rows=2)
        self.assertEqual(replay.bbs_queries, [1, 0])
        for threshold in (.25, .5):
            native = next(row for row in report["paired_transitions"]
                          if row["protocol"] == "official_filter"
                          and row["geometry"] == "native"
                          and row["threshold"] == threshold)
            self.assertEqual(native["repaired"], 1)
            self.assertEqual(native["damaged"], 1)
            self.assertEqual(native["net_hits"], 0)
            self.assertEqual(native["both_correct"], 0)
            self.assertEqual(native["both_wrong"], 0)

    def test_paired_rows_within_batch_use_the_same_sample_order(self):
        batch = {name: torch.cat([value, value], dim=0)
                 for name, value in self.points.items()}
        batch["last_center"][1] = self.points["native_coarse_center"][0]
        batch["native_coarse_center"][1] = self.points["last_center"][0]
        batch["last_sem_cls_scores"][1] = batch["last_sem_cls_scores"][1].flip(0)
        evaluator = GroundingEvaluator(
            only_root=True, thresholds=[.25, .5], topks=[1, 5, 10],
            prefixes=["last_"], filter_non_gt_boxes=True, model="PVGround")
        evaluator.evaluate_bbox_by_sem_align(batch, "last_")
        replay = GeometryReplay(evaluator)
        replay.observe(batch)
        report = replay.report(expected_rows=2)
        self.assertEqual(replay.bbs_queries, [1, 0])
        native = next(row for row in report["paired_transitions"]
                      if row["protocol"] == "official_filter"
                      and row["geometry"] == "native" and row["threshold"] == .5)
        self.assertEqual((native["repaired"], native["damaged"]), (1, 1))
        self.assertEqual(native["net_hits"], 0)
        replay.evaluators["official_filter", "native"].dets.samples[.5].pop()
        with self.assertRaises(AssertionError):
            replay.report(expected_rows=2)

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
