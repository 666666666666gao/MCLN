import pytest

from scripts.nr3d_view_pair_contract import compare_rows, digest_ids, split_rows


def row(index, view, h25, h50):
    return dict(id=index, view_dependent=view, hit025=h25, hit050=h50)


def test_rows_from_same_scene_always_share_partition():
    rows = [dict(scan_id="scene0000_00"), dict(scan_id="scene0001_00"),
            dict(scan_id="scene0000_00")]
    partitions = split_rows(rows)
    assert any(0 in ids and 2 in ids for ids in partitions.values())
    assert sorted(partitions["fit"] + partitions["holdout"]) == [0, 1, 2]


def test_identity_and_order_are_distinct_and_duplicates_are_rejected():
    assert digest_ids([1, 2]) == digest_ids([2, 1])
    assert digest_ids([1, 2], True) != digest_ids([2, 1], True)
    with pytest.raises(ValueError, match="repeated"):
        digest_ids([1, 1])


def test_view_gain_alone_cannot_pass_overall_gate():
    old = [row(0, True, False, False), row(1, False, True, False)]
    fixed = [row(0, True, True, False), row(1, False, False, False)]
    result = compare_rows(old, fixed)
    assert result["metrics"]["overall"]["025"]["delta_hits"] == 0
    assert not result["scientific_gate_pass"]


def test_050_regression_vetoes_025_gain():
    old = [row(0, True, False, False), row(1, False, True, True)]
    fixed = [row(0, True, True, False), row(1, False, True, False)]
    assert not compare_rows(old, fixed)["scientific_gate_pass"]


def test_fixed_common_view_groups_and_row_identity_are_required():
    old = [row(0, True, False, False)]
    with pytest.raises(ValueError, match="groups"):
        compare_rows(old, [row(0, False, True, False)])
    with pytest.raises(ValueError, match="order"):
        compare_rows(old, [row(1, True, True, False)])


def test_positive_primary_and_view_without_050_regression_pass():
    result = compare_rows([row(0, True, False, False)],
                          [row(0, True, True, False)])
    assert result["scientific_gate_pass"]
    assert result["metrics"]["overall"]["025"]["fixes"] == 1
