import pytest

from src.joint_det_dataset import Joint3DDataset


@pytest.mark.parametrize(
    "utterance",
    (
        "Facing the whiteboard, choose the lamp on the right.",
        "Looking at the books, choose the book on the left.",
        "The chair is on the left.",
        "Choose the rightmost pillow.",
    ),
)
def test_nr3d_view_cues_block_large_rotation_across_case_and_punctuation(
        utterance):
    assert Joint3DDataset._is_view_dep(utterance) is True
    assert Joint3DDataset._augment_nr3d(utterance) is False


def test_nr3d_non_view_description_keeps_large_rotation_augmentation():
    utterance = "choose the wooden chair beside the table"
    assert Joint3DDataset._is_view_dep(utterance) is False
    assert Joint3DDataset._augment_nr3d(utterance) is True
