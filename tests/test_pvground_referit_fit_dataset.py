"""Exercise repeated native annotations and pre-parser text identity."""
import hashlib
import json
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from pvground_referit_fit_dataset import make_fit_dataset_class


class NativeFixture:
    def __init__(self, rows):
        self.annos = rows
        self.parsed = False

    def _scene_graph_parse(self, rows):
        self.parsed = True
        for row in rows:
            row['utterance'] += ' normalized'

    def __getitem__(self, index):
        return dict(sample_dataset=self.annos[index]['dataset'],
                    gt_masks=np.array([[0., 1., 1.]], dtype=np.float32))


def fixture():
    row = dict(scan_id='scene0001_00', target_id=1, utterance='the chair', dataset='nr3d')
    digest = hashlib.sha256(json.dumps(row, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    return row, dict(dataset='nr3d', original_language_rows=1, language_row_keys_sha256=[digest])


class DatasetContractTests(unittest.TestCase):
    def test_repeated_detection_occurrences_keep_their_actual_indices(self):
        row, partition = fixture()
        detection = dict(scan_id='scene0002_00', dataset='scannet')
        rows = [row] + [detection] * 10
        dataset = make_fit_dataset_class(NativeFixture, partition, 'nr3d')(rows)
        outputs = [dataset[i] for i in (1, 10, 3, 1)]
        self.assertEqual([x['local_training_id'] for x in outputs], [1, 10, 3, 1])
        self.assertNotIn('local_training_id', detection)
        self.assertEqual(outputs[0]['gt_masks'].dtype, np.bool_)
        np.testing.assert_array_equal(outputs[0]['gt_masks'], [[False, True, True]])

    def test_checks_raw_text_before_native_normalization(self):
        row, partition = fixture()
        dataset = make_fit_dataset_class(NativeFixture, partition, 'nr3d')([row])
        dataset._scene_graph_parse([row])
        self.assertTrue(dataset.parsed)
        self.assertEqual(row['utterance'], 'the chair normalized')

    def test_changed_target_or_text_is_rejected_before_parser(self):
        for key, value in [('target_id', 2), ('utterance', 'another chair')]:
            row, partition = fixture()
            row[key] = value
            dataset = make_fit_dataset_class(NativeFixture, partition, 'nr3d')([row])
            with self.assertRaises(AssertionError):
                dataset._scene_graph_parse([row])
            self.assertFalse(dataset.parsed)


if __name__ == '__main__':
    unittest.main()
