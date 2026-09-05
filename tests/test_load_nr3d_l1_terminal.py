import hashlib

import pytest
import torch

from scripts.load_nr3d_l1_terminal import load_terminal_position_key
from scripts.nr3d_text_position_key import TextPositionKey


def saved_fixture(tmp_path, mode='position', steps=6687, parent='a'*64):
    torch.manual_seed(0)
    addon = TextPositionKey(288, 8, mode)
    with torch.no_grad():
        addon.weight.normal_(0, .01)
    path = tmp_path/'synthetic_key_state.pt'
    torch.save({'addon_state':addon.state_dict(), 'mode':mode, 'steps':steps,
                'parent_checkpoint_sha256':parent, 'optimizer':{}}, path)
    return addon, path, hashlib.sha256(path.read_bytes()).hexdigest()


def test_round_trip_preserves_position_bias_and_is_frozen(tmp_path):
    original, path, digest = saved_fixture(tmp_path)
    loaded = load_terminal_position_key(path, digest, 'a'*64)
    query, text, points, positions = [torch.randn(2, length, 288) for length in [4, 5, 7, 7]]
    assert torch.equal(original(query, text, points, positions), loaded(query, text, points, positions))
    assert not loaded.training and not loaded.weight.requires_grad


@pytest.mark.parametrize('mode,steps,parent', [
    ('text',6687,'a'*64), ('position',6686,'a'*64), ('position',6687,'b'*64),
])
def test_rejects_other_arm_incomplete_endpoint_or_other_parent(tmp_path, mode, steps, parent):
    _, path, digest = saved_fixture(tmp_path, mode, steps, parent)
    with pytest.raises(AssertionError):
        load_terminal_position_key(path, digest, 'a'*64)


def test_rejects_artifact_outside_the_recorded_digest(tmp_path):
    _, path, _ = saved_fixture(tmp_path)
    with pytest.raises(AssertionError):
        load_terminal_position_key(path, '0'*64, 'a'*64)
