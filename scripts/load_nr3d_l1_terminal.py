"""Read the registered position-key endpoint for an isolated evaluation.

Both L1 arms have the same tensor shape but different input meanings. Loading
checks the saved arm and parent identity; it does not decide scientific promotion.
The terminal paired screen must separately pass before formal evaluation.
"""

import hashlib
import io
from pathlib import Path

import torch

from scripts.nr3d_text_position_key import TextPositionKey


def load_terminal_position_key(path, artifact_sha256, parent_checkpoint_sha256):
    raw = Path(path).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == artifact_sha256
    checkpoint = torch.load(io.BytesIO(raw), map_location='cpu')
    assert checkpoint['mode'] == 'position'
    assert checkpoint['steps'] == 6687
    assert checkpoint['parent_checkpoint_sha256'] == parent_checkpoint_sha256
    addon = TextPositionKey(288, 8, 'position')
    addon.load_state_dict(checkpoint['addon_state'], strict=True)
    addon.requires_grad_(False).eval()
    return addon
