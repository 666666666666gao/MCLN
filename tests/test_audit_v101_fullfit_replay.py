import scripts.audit_v101_fullfit_replay as audit


def test_v101_replay_is_bound_to_frozen_evidence():
    assert audit.V101_ARTIFACT_SHA256 == (
        "2c969a6c28a0c9315b53f0f847567345e47da8c912091344b23612680643a2ae"
    )
    assert audit.V101_OOF_SHA256 == (
        "2cb453b130306449901bed9c337984aad7f8b7048d05bd4240b5077f0de9ac1e"
    )
    assert audit.EXPECTED_ROW_COUNT == 36665
    assert audit.EXPECTED_SCENE_COUNT == 562
