"""Unqualified or corrupt cache must never launch the GPU model probe."""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/queue_referit3d_pretrained_appearance_probe.py'
spec = importlib.util.spec_from_file_location('probe_queue', str(SCRIPT))
queue = importlib.util.module_from_spec(spec)
spec.loader.exec_module(queue)


class ProbeQueueTests(unittest.TestCase):
    def prepare(self, root, code, decision):
        old = root / 'cache_queue'
        old.mkdir()
        (old / 'plan.json').write_text('{}')
        (old / 'queue.exit').write_text(str(code))
        (old / 'decision.json').write_text(json.dumps(decision))
        (root / 'plan.json').write_text(json.dumps(dict(files={}, cache_queue_root=str(old),
            cache_queue_plan_sha256=hashlib.sha256(b'{}').hexdigest())))
        return old

    def test_failed_previous_queue(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.prepare(root, 7, {})
            with patch('sys.argv', ['queue', '--plan', str(root / 'plan.json')]), patch.object(queue.subprocess, 'run') as run:
                self.assertEqual(queue.main(), 7)
                run.assert_not_called()
            self.assertFalse((root / 'probe_manifest.json').exists())

    def test_unqualified_scan(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.prepare(root, 0, dict(status='scan_not_qualified'))
            with patch('sys.argv', ['queue', '--plan', str(root / 'plan.json')]), patch.object(queue.subprocess, 'run') as run:
                self.assertEqual(queue.main(), 0)
                run.assert_not_called()
            self.assertEqual(json.loads((root / 'decision.json').read_text())['status'], 'scan_not_qualified')

    def test_corrupt_cache_receipt(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            old = self.prepare(root, 0, dict(status='train_cache_complete', cache_receipt_sha256='0' * 64))
            (old / 'cache').mkdir()
            (old / 'cache/receipt.json').write_text('{}')
            with patch('sys.argv', ['queue', '--plan', str(root / 'plan.json')]), patch.object(queue.subprocess, 'run') as run:
                with self.assertRaises(AssertionError):
                    queue.main()
                run.assert_not_called()
            self.assertFalse((root / 'probe_manifest.json').exists())


if __name__ == '__main__':
    unittest.main()
