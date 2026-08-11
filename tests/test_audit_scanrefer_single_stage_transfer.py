import re

from scripts.audit_scanrefer_single_stage_transfer import (
    DETECTED_STREAM_PATTERNS,
    is_detected_stream_tensor,
)


def test_detected_stream_allowlist_covers_only_removed_modules():
    allowed = (
        "butd_class_embeddings.weight",
        "class_embeddings.bias",
        "box_embeddings.position_embedding_head.0.weight",
        "cross_encoder.layers.2.cross_layer.cross_d.in_proj_weight",
        "cross_encoder.layers.0.cross_layer.norm_d.bias",
        "decoder.5.cross_d.out_proj.weight",
        "decoder.1.norm_d.bias",
    )
    rejected = (
        "backbone_net.sa1.mlp_module.0.weight",
        "decoder.1.cross_t.in_proj_weight",
        "source_choice_selector.head.weight",
        "cross_encoder.layers.0.cross_layer.norm_t.bias",
    )
    assert all(is_detected_stream_tensor(name) for name in allowed)
    assert not any(is_detected_stream_tensor(name) for name in rejected)
    assert all(pattern.pattern.startswith("^") for pattern in DETECTED_STREAM_PATTERNS)


def test_detected_stream_patterns_are_end_anchored_to_module_prefixes():
    assert not is_detected_stream_tensor("prefix.decoder.1.norm_d.bias")
    assert not is_detected_stream_tensor("decoder.bad.norm_d.bias")
    assert not is_detected_stream_tensor("decoder.1.norm_danger.bias")
