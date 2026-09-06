"""Checkpoint identity for the fixed last-layer candidate-local reader."""


LOCAL_VISUAL_PREFIX = 'decoder.5.local_visual.'


def local_visual_state_keys(state):
    return {name for name in state
            if (name[7:] if name.startswith('module.') else name).startswith(LOCAL_VISUAL_PREFIX)}


def validate_local_visual_checkpoint(current, saved, model_only_initialization):
    local_keys = local_visual_state_keys(current)
    if not local_keys:
        raise ValueError('candidate-local checkpoint requires --use_candidate_local_visual')
    expected_missing = (local_keys if model_only_initialization
                        and not local_visual_state_keys(saved) else set())
    missing = set(current) - set(saved)
    unexpected = set(saved) - set(current)
    if missing != expected_missing or unexpected:
        raise ValueError('candidate-local checkpoint key mismatch: missing={}, unexpected={}'.format(
            sorted(missing), sorted(unexpected)))
    incompatible = [name for name, value in saved.items()
                    if value.shape != current[name].shape or value.dtype != current[name].dtype]
    if incompatible:
        raise ValueError('candidate-local checkpoint shape/dtype mismatch: ' + ', '.join(incompatible))
