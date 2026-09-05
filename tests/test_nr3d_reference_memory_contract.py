from copy import deepcopy

from scripts.nr3d_reference_memory_contract import decide


def example():
    return {"raw_token_count": 15, "distractor_count": 3,
            "protected_mask_iou": .4,
            "scores": {name: {"box_iou": iou, "mask_iou": .6}
                       for name, iou in [("query_global", .1), ("query_pair", .1),
                                         ("object_global", .1), ("object_pair", .8),
                                         ("protected", .1)]}}


def test_primary_requires_memory_readout_and_practical_improvement():
    row = example()
    result = decide([row])
    assert result["eligible_for_decoder_experiment"]
    assert not result["formal_promotion"]
    same_readout = deepcopy(row)
    same_readout["scores"]["object_global"]["box_iou"] = .8
    result = decide([same_readout])
    assert result["memory_screen_pass"] and not result["readout_screen_pass"]
    assert not result["eligible_for_decoder_experiment"]
    assert not result["control_substituted_for_failed_primary_candidate"]


def test_mask_regression_blocks_primary_even_when_both_controls_lose():
    row = example()
    row["scores"]["object_pair"]["mask_iou"] = .3
    result = decide([row])
    assert result["memory_screen_pass"] and result["readout_screen_pass"]
    assert not result["practical_screen_pass"]
    assert not result["eligible_for_decoder_experiment"]
