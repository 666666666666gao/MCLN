"""Bind native mixed-dataset loading to the previously saved physical-room split."""
import hashlib
import json


def make_fit_dataset_class(base_class, partition, language_dataset):
    assert language_dataset in ('nr3d', 'sr3d')
    assert partition['dataset'] == language_dataset

    class FitDataset(base_class):
        def _scene_graph_parse(self, annos):
            # Check raw text identity before the native parser normalizes it.
            assert len(annos) == partition['original_language_rows']
            actual = [hashlib.sha256(json.dumps(
                {key: row[key] for key in ('scan_id', 'target_id', 'utterance', 'dataset')},
                sort_keys=True, separators=(',', ':')).encode()).hexdigest() for row in annos]
            assert actual == partition['language_row_keys_sha256']
            assert all(row['dataset'] == language_dataset for row in annos)
            super()._scene_graph_parse(annos)

        def __getitem__(self, index):
            result = super().__getitem__(index)
            # Native tenfold detection expansion aliases annotation dictionaries.
            # The sampled dataset index, not a field attached to that dictionary,
            # identifies this occurrence of a repeated detection example.
            result['local_training_id'] = index
            assert result['sample_dataset'] == self.annos[index]['dataset']
            masks = result['gt_masks']
            assert ((masks == 0) | (masks == 1)).all()
            result['gt_masks'] = masks.astype(bool)
            return result

    return FitDataset
