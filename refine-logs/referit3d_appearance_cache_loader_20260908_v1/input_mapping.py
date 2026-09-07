def _get_inputs(batch_data):
    inputs = {
        'point_clouds': batch_data['point_clouds'].float(), # ([B, 50000, 6]) xyz + colour
        'text': batch_data['utterances'],                   # list[B]  text
        "det_boxes": batch_data['all_detected_boxes'],      # ([B, 132, 6]) groupfree detection boxes
        "det_bbox_label_mask": batch_data['all_detected_bbox_label_mask'],  # ([B, 132]) mask
        "det_class_ids": batch_data['all_detected_class_ids'],   # ([B, 132])  box id
        "superpoint": batch_data['superpoint'],  # ([B, 50000]) superpoint map
    }
    for key in [
            "positive_map",
            "modify_positive_map",
            "pron_positive_map",
            "other_entity_map",
            "rel_positive_map",
            "target_spans",
            "entity_spans",
            "attr_spans",
            "rel_spans",
            "structured_anchor_ids",
            "coverage_stats",
            "parse_confidence",
            "decomposition_status",
            "decomp_global_only_mask",
            "decomp_weak_generic_mask",
            "structured_annotation_available",
            "det_visual_features",
            "det_visual_available",
    ]:
        if key in batch_data:
            inputs[key] = batch_data[key]
    return inputs
