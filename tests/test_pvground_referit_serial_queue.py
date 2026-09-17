import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

source = Path(__file__).resolve().parents[1] / 'scripts/run_pvground_referit_serial_queue.py'
spec = importlib.util.spec_from_file_location('serial_queue', source)
queue = importlib.util.module_from_spec(spec)
spec.loader.exec_module(queue)


class SerialQueueTests(unittest.TestCase):
    def test_scan_module_regression_does_not_require_nonexistent_formal(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            queue.record(root / 'decision.json', {'status': 'skipped_primary_rec_regression'})
            self.assertFalse(queue.scan_pass(root))

    def test_formal_acceptance_uses_both_rec_checks_and_binds_receipt(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            queue.record(root / 'decision.json', {'status': 'launching_fixed_formal'})
            queue.record(root / 'receipt.json', {'mask_miou': 0})
            audit = dict(integrity_pass=True, formal_rows=9508, scanrefer_mask_gate=False,
                         receipt_sha256=queue.sha(root / 'receipt.json'),
                         advance_to_nr3d_sr3d_rec=True, checks={'rec25': True, 'rec50': True})
            queue.record(root / 'audit.json', audit)
            self.assertTrue(queue.scan_pass(root))
            audit.update(advance_to_nr3d_sr3d_rec=False, checks={'rec25': True, 'rec50': False})
            (root / 'audit.json').write_text(json.dumps(audit))
            self.assertFalse(queue.scan_pass(root))
            (root / 'receipt.json').write_text('{}')
            with self.assertRaises(AssertionError):
                queue.scan_pass(root)

    def test_actual_subprocess_failure_is_recorded_and_not_retried(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaises(AssertionError):
                queue.run_stage([sys.executable, '-c', 'raise SystemExit(7)'], root,
                                'controller.exit', 'controller.log', dict(os.environ))
            self.assertEqual((root / 'controller.exit').read_text().strip(), '7')
            with self.assertRaises(AssertionError):
                queue.run_stage([sys.executable, '-c', 'print("rerun")'], root,
                                'controller.exit', 'controller.log', dict(os.environ))
            self.assertEqual((root / 'controller.log').read_text(), '')

    def test_missing_original_handle_does_not_launch_replacement(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(AssertionError):
                queue.wait_dependency(directory, 999999999)
            self.assertEqual(list(Path(directory).iterdir()), [])


if __name__ == '__main__':
    unittest.main()
