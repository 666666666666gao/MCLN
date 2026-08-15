import scripts.materialize_v101_oof_sidecar as sidecar


def test_sidecar_is_bound_to_frozen_v101_oof():
    assert sidecar.V101_RESULT_SHA256 == (
        "2cb453b130306449901bed9c337984aad7f8b7048d05bd4240b5077f0de9ac1e"
    )
    assert sidecar.EXPECTED_PREDICTION_SHA256 == (
        "b81664e65d64dad7058f8f252d990d4ab11dd8c00746c64a918bb120b6434c99"
    )
    assert sidecar.EXPECTED_ROW_COUNT == 36665
    assert sidecar.EXPECTED_SCENE_COUNT == 562
