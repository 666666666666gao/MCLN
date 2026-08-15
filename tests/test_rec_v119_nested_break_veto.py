from scripts.run_v119_meshsp_nested_break_veto_oof import (
    V118_BREAK_COST,
    V119_BREAK_VETO_THRESHOLD,
)


def test_v119_veto_is_the_frozen_break_neutral_midpoint():
    assert V118_BREAK_COST == 4.0
    assert V119_BREAK_VETO_THRESHOLD == -2.0
    assert V119_BREAK_VETO_THRESHOLD == (-V118_BREAK_COST + 0.0) / 2.0

