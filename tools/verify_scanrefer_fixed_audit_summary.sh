#!/usr/bin/env bash
set -euo pipefail
/root/autodl-tmp/PV-Ground/env/bin/python - <<'PY'
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

audit = Path("/root/autodl-tmp/mcln_pvground_ft_20261101/offline_audits/b2_fixed_audit_20261103_lrpositive")
spec = importlib.util.spec_from_file_location("summary", audit / "summarize_pvground_fixed_audit.py")
summary = importlib.util.module_from_spec(spec)
spec.loader.exec_module(summary)
metadata, forward, semantic = None, [], []
with (audit / "none.jsonl").open() as stream:
    for line in stream:
        row = json.loads(line)
        if row["kind"] == "metadata":
            metadata = row
        elif row["kind"] == "forward_sample" and len(forward) < 8:
            forward.append(row)
        elif row["kind"] == "semantic_sample" and len(semantic) < 8:
            semantic.append(row)
        if len(forward) == len(semantic) == 8:
            break
assert metadata is not None and len(forward) == len(semantic) == 8
footer = {
    "kind": "forward_complete", "samples": 8, "semantic_samples": 8,
    "semantic_hits025": sum(row["selected_iou"] > 0.25 for row in semantic),
    "semantic_hits050": sum(row["selected_iou"] > 0.5 for row in semantic),
    "verified_against_evaluator": True,
}


class SummaryChecks(unittest.TestCase):
    def read_fixture(self, footer_value):
        with tempfile.TemporaryDirectory(dir=audit) as directory:
            path = Path(directory) / "fixture.jsonl"
            rows = [metadata] + forward + semantic + footer_value
            path.write_text("".join(json.dumps(row) + "\n" for row in rows))
            with patch.object(summary, "EXPECTED_SAMPLES", 8):
                return summary.read_complete(path, "none", metadata["checkpoint_sha256"])

    def test_valid_fixture_and_identical_pair(self):
        result, samples = self.read_fixture([footer])
        self.assertEqual(result["hits025"], footer["semantic_hits025"])
        paired = summary.compare(samples, copy.deepcopy(samples))
        self.assertEqual(paired["bypass_net_hits050"], 0)
        self.assertEqual(paired["selected_query_changed"], 0)

    def test_missing_footer_rejected(self):
        with self.assertRaisesRegex(ValueError, "completion footer"):
            self.read_fixture([])

    def test_evaluator_count_disagreement_rejected(self):
        altered = dict(footer, semantic_hits050=footer["semantic_hits050"] + 1)
        with self.assertRaisesRegex(ValueError, "footer hit mismatch"):
            self.read_fixture([altered])

    def test_changed_points_rejected(self):
        changed = copy.deepcopy(semantic)
        changed[0]["point_sha256"] = "changed"
        with self.assertRaisesRegex(ValueError, "input/identity mismatch"):
            summary.compare(semantic, changed)

    def test_strict_threshold_flips(self):
        left, right = copy.deepcopy(semantic[:3]), copy.deepcopy(semantic[:3])
        for row, value in zip(left, (0.5, 0.51, 0.49)):
            row["selected_iou"] = value
        for row, value in zip(right, (0.51, 0.5, 0.5)):
            row["selected_iou"] = value
        result = summary.compare(left, right)
        self.assertEqual(result["bypass_gained_hits050"], 1)
        self.assertEqual(result["bypass_lost_hits050"], 1)
        self.assertEqual(result["bypass_net_hits050"], 0)


suite = unittest.defaultTestLoader.loadTestsFromTestCase(SummaryChecks)
result = unittest.TextTestRunner(verbosity=2).run(suite)
receipt = {
    "tests_run": result.testsRun, "successful": result.wasSuccessful(),
    "scope": "Remote stdlib-only result-parser fixtures derived from eight real forward records. Not full validation or trained ablations.",
}
(audit / "summary_tests.json").write_text(json.dumps(receipt, indent=2) + "\n")
if not result.wasSuccessful():
    raise SystemExit(1)
print(json.dumps(receipt))
PY
