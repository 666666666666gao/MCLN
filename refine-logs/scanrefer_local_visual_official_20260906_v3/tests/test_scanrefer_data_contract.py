import hashlib

import pytest

from scripts.scanrefer_data_contract import set_scanrefer_data_root, verify_scanrefer_superpoints


def test_current_run_replaces_only_historical_data_root():
    historical = ['train_dist_mod.py', '--data_root', '/old', '--eval', '--batch_size', '12']
    corrected = set_scanrefer_data_root(historical, '/mesh/')
    assert corrected == ['train_dist_mod.py', '--data_root', '/mesh/', '--eval', '--batch_size', '12']
    assert historical[2] == '/old'


def test_native_dataset_requires_directory_separator_for_tokenizer_path():
    with pytest.raises(AssertionError):
        set_scanrefer_data_root(['--data_root', '/old/'], '/mesh')


def make_superpoints(directory, split, count):
    folder = directory / 'superpoints' / split
    folder.mkdir(parents=True)
    expected = {}
    for i in range(count):
        name = 'scene%04d_00.npy' % i
        raw = ('synthetic unit-test superpoint file %d' % i).encode()
        (folder / name).write_bytes(raw)
        expected[name] = hashlib.sha256(raw).hexdigest()
    return expected


@pytest.mark.parametrize('split,count', [('val', 312), ('train', 1201)])
def test_complete_pinned_inputs_are_accepted(tmp_path, split, count):
    expected = make_superpoints(tmp_path, split, count)
    result = verify_scanrefer_superpoints(tmp_path, split, expected)
    assert result['files_verified'] == count
    assert result['resolved_directory'] == str((tmp_path / 'superpoints' / split).resolve())


def test_old_superpoint_with_same_filename_is_rejected_before_model_run(tmp_path):
    expected = make_superpoints(tmp_path, 'val', 312)
    (tmp_path / 'superpoints/val/scene0000_00.npy').write_bytes(b'old mixed superpoint content')
    with pytest.raises(AssertionError):
        verify_scanrefer_superpoints(tmp_path, 'val', expected)


def test_partial_manifest_cannot_certify_full_validation_inputs(tmp_path):
    expected = make_superpoints(tmp_path, 'val', 311)
    with pytest.raises(AssertionError):
        verify_scanrefer_superpoints(tmp_path, 'val', expected)
