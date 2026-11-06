"""Checkpoint identity for the fixed last-layer candidate-local reader."""


LOCAL_VISUAL_PREFIX = 'decoder.5.local_visual.'


def local_visual_state_keys(state):
    return {name for name in state
            if (name[7:] if name.startswith('module.') else name).startswith(LOCAL_VISUAL_PREFIX)}


def validate_local_visual_checkpoint(current, saved, model_only_initialization,
                                     current_variant='local', saved_variant='local',
                                     additional_new_prefixes=(),
                                     require_local_visual=True):
    def canonical_state(state):
        return {
            name[7:] if name.startswith('module.') else name: value
            for name, value in state.items()
        }

    current_state = canonical_state(current)
    saved_state = canonical_state(saved)
    local_keys = local_visual_state_keys(current_state)
    if require_local_visual and not local_keys:
        raise ValueError('candidate-local checkpoint requires --use_candidate_local_visual')
    saved_local_keys = local_visual_state_keys(saved_state)
    if saved_local_keys and current_variant != saved_variant:
        raise ValueError('candidate-local reader variant mismatch: current={}, saved={}'.format(
            current_variant, saved_variant))
    expected_missing = set()
    if model_only_initialization:
        expected_missing.update(local_keys - saved_local_keys)
        for prefix in additional_new_prefixes:
            expected_missing.update(
                name for name in current_state
                if name.startswith(prefix) and name not in saved_state
            )
    missing = set(current_state) - set(saved_state)
    unexpected = set(saved_state) - set(current_state)
    if model_only_initialization:
        # Older RoBERTa checkpoints persisted this generated, non-trainable buffer.
        unexpected.discard('text_encoder.embeddings.position_ids')
    if missing != expected_missing or unexpected:
        raise ValueError('candidate-local checkpoint key mismatch: missing={}, unexpected={}'.format(
            sorted(missing), sorted(unexpected)))
    incompatible = [name for name, value in saved_state.items()
                    if name in current_state and (
                        value.shape != current_state[name].shape
                        or value.dtype != current_state[name].dtype)]
    if incompatible:
        raise ValueError('candidate-local checkpoint shape/dtype mismatch: ' + ', '.join(incompatible))
