"""Bound current save peaks from the actual CPU-loaded state and fixed optimizer."""
import json
from pathlib import Path

root = Path(__file__).resolve().parent
measured = json.loads((root.parent / 'SAVED_STATE_CAPACITY.json').read_bytes())
assert measured['amsgrad'] == [False] * 4 and not measured['unexpected_state_ids']
assert measured['optimizer_param_ids_count'] == measured['optimizer_param_ids_unique_count'] == 755
assert measured['new_weights_written'] == measured['neural_forwards'] == measured['optimizer_updates'] == 0
metadata = 64 * 1024 ** 2
logs = 256 * 1024 ** 2
latest = measured['model_tensor_bytes'] + 2 * measured['non_roberta_tensor_bytes'] + metadata
best = measured['model_tensor_bytes'] + metadata
peak = max(2 * latest + best, latest + 2 * best)
budget = dict(model_tensor_bytes=measured['model_tensor_bytes'],
              frozen_roberta_tensor_bytes=measured['text_encoder_tensor_bytes'],
              all_potential_trainable_tensor_bytes_upper=measured['non_roberta_tensor_bytes'],
              complete_Adam_moment_bytes_upper=2 * measured['non_roberta_tensor_bytes'],
              latest_bytes_upper=latest, best_bytes_upper=best,
              metadata_serialization_config_history_rng_reserve_per_weight=metadata,
              atomic_save_peak_bytes_upper=peak, log_reserve_bytes=logs,
              required_free_bytes=peak + logs,
              measured_serialized_full_bytes=measured['serialized_full_bytes'],
              measured_serialized_model_only_bytes=measured['serialized_model_only_bytes'],
              optimizer_missing_state_count=len(measured['missing_state_ids']),
              includes_future_missing_Adam_states=True,
              rationale='Full model plus two moments for every non-RoBERTa tensor, including buffers as an upper bound; frozen RoBERTa, fixed FP32 shapes, amsgrad=False. Saves retain latest/best and one atomic replacement. No new weights or NN used.')
assert budget['required_free_bytes'] == 2664907480
(root / 'STORAGE_BUDGET.json').write_text(json.dumps(budget, indent=2) + '\n')
print(json.dumps(budget))
