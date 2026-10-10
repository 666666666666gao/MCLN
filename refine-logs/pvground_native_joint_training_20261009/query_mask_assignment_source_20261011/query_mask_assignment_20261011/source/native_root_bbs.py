"""Actual ScanRefer root bbs evidence; no additional deployed ranking."""


def native_root_bbs(semantic_logits, batch):
    probability = semantic_logits.softmax(-1)
    score = (probability * (batch['positive_map'][:, 0] > 0)[:, None]).sum(-1)
    for name in ('modify_positive_map', 'pron_positive_map', 'rel_positive_map'):
        score = score + (probability * batch[name][:, 0, None]).sum(-1)
    return score - (probability * batch['other_entity_map'][:, 0, None]).sum(-1)
