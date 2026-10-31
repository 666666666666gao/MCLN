"""Checks for separation and reconciliation of scientific metric sources."""

import json
from pathlib import Path
import tempfile
import unittest

from pvground_log_diagnostics import collect


class SemanticLogTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.log = Path(self.directory.name) / "train.log"
        self.metrics = Path(self.directory.name) / "native_metrics.jsonl"
        self.native = {
            "epoch": 1,
            "rows": 9508,
            "hits025": 5626,
            "hits050": 4773,
            "primary_score": "last/bbf",
            "same_complete_model": True,
        }
        lines = [
            "[date logs]: epoch 1, total time 3800",
            "[date logs]: last_ position alignment Acc0.25: "
            "Top-1: 0.99000, Top-5: 0.99000, Top-10: 0.99000",
            "[date logs]: last_ semantic alignment Acc0.25: "
            "Top-1: 0.59171, Top-5: 0.70593, Top-10: 0.73549",
            "[date logs]: last_ semantic alignment Acc0.50: "
            "Top-1: 0.50200, Top-5: 0.63189, Top-10: 0.66649",
        ]
        for suffix, threshold in (("", "0.25"), ("50", "0.50")):
            lines.append(f"[date logs]: iou@{threshold}")
            for group in ("easy", "hard", "vd", "vid", "unique", "multi"):
                lines.append(f"[date logs]: {group}{suffix} 0.5")
        lines.extend(
            [
                "[date logs]: mask@kiou",
                "[date logs]: unique50 0.99",
                "[date logs]: multi50 0.99",
                "[date logs]: epoch 2, total time 3800",
                "50%|#####| 400/951",
            ]
        )
        self.log.write_text("\n".join(lines), encoding="utf-8")
        self.write_metrics()

    def write_metrics(self):
        self.metrics.write_text(json.dumps(self.native) + "\n", encoding="utf-8")

    def test_box_source_separation_and_incomplete_epoch(self):
        result = collect(self.log, self.metrics, "scanrefer")
        self.assertEqual(len(result["rows"]), 1)
        row = result["rows"][0]
        self.assertEqual(row["epoch"], 1)
        self.assertEqual(row["subgroups"]["unique50"], 0.5)
        self.assertEqual(row["acc25"]["top1_exact"], 5626 / 9508)
        self.assertAlmostEqual(row["acc50"]["top10_minus_top1_pp"], 16.4491682793)

    def test_native_hit_mismatch_is_rejected(self):
        self.native["hits050"] = 4843
        self.write_metrics()
        with self.assertRaises(AssertionError):
            collect(self.log, self.metrics, "scanrefer")

    def test_incomplete_dataset_is_rejected(self):
        self.native["rows"] = 9507
        self.write_metrics()
        with self.assertRaises(AssertionError):
            collect(self.log, self.metrics, "scanrefer")


if __name__ == "__main__":
    unittest.main()
