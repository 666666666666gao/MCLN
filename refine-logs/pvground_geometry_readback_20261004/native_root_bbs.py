"""Actual ScanRefer root bbs evidence; no additional deployed ranking."""


def native_root_bbs(semantic_logits, batch):
    probability = semantic_logits.softmax(-1)
    evidence = (batch['positive_map'][:, 0] > 0).to(probability.dtype)
    for name in ('modify_positive_map', 'pron_positive_map', 'rel_positive_map'):
        evidence = evidence + batch[name][:, 0]
    evidence = evidence - batch['other_entity_map'][:, 0]
    return (probability * evidence[:, None]).sum(-1)
