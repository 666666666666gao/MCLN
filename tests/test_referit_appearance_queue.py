"""Rejection paths must never allocate the follow-up appearance cache."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

SOURCE = Path(__file__).resolve().parents[1] / 'scripts/queue_referit3d_openshape_cache.py'
spec = importlib.util.spec_from_file_location('referit_queue_under_test', SOURCE)
queue = importlib.util.module_from_spec(spec)
spec.loader.exec_module(queue)


class RejectionTests(unittest.TestCase):
    def run_case(self, exit_code, decision, expected_code, expected_status):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            formal = root / 'scan'
            formal.mkdir()
            (formal / 'plan.json').write_text('{}')
            (formal / 'queue.exit').write_text(str(exit_code))
            if decision['status'] == 'formal_evaluated_and_audited':
                (formal / 'formal/result').mkdir(parents=True)
                receipt = formal / 'formal/result/receipt.json'
                audit = formal / 'formal/independent_audit.json'
                receipt.write_text(json.dumps(dict(status='complete', formal_rows=9508, promotion=decision['promotion'])))
                audit.write_text(json.dumps(dict(promotion=decision['promotion'])))
                decision = dict(decision, formal_receipt_sha256=queue.sha(receipt), audit_sha256=queue.sha(audit))
            (formal / 'decision.json').write_text(json.dumps(decision))
            plan = root / 'plan.json'
            plan.write_text(json.dumps(dict(files={}, scan_formal_root=str(formal),
                scan_formal_plan_sha256=queue.sha(formal / 'plan.json'))))
            with patch('sys.argv', ['queue', '--plan', str(plan)]), patch.object(queue.subprocess, 'run') as run:
                self.assertEqual(queue.main(), expected_code)
                run.assert_not_called()
            self.assertFalse((root / 'cache').exists())
            self.assertEqual(json.loads((root / 'decision.json').read_text())['status'], expected_status)

    def test_prior_failure_stops(self):
        self.run_case(1, dict(status='native_followup_failed'), 1, 'scan_formal_queue_failed')

    def test_module_rejection_stops(self):
        self.run_case(0, dict(status='module_screen_rejected'), 0, 'scan_not_qualified')

    def test_formal_nonqualification_stops(self):
        self.run_case(0, dict(status='formal_evaluated_and_audited',
            promotion=dict(advance_to_nr3d_sr3d_rec=False)), 0, 'scan_not_qualified')


if __name__ == '__main__':
    unittest.main()
