"""Prepare common native loop selection and early stop for the new pair only."""
import ast
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parent
original = root.parent / 'source/main_utils.py'
output = root / 'source_overlay_v2/main_utils.py'
assert not output.exists()
raw = original.read_bytes()
text = raw.decode('utf-8').replace('\r\n', '\n')
changes = [
    ("def native_metric_order(metrics):\n    return (metrics['hits025'] >= 5658 and metrics['hits050'] >= 4850,\n"
     "            metrics['hits050'] >= 4850, metrics['hits025'], metrics['hits050'])",
     "def native_metric_order(metrics):\n    return (metrics['hits025'] >= 5705 and metrics['hits050'] >= 4945,\n"
     "            metrics['hits050'], metrics['hits025'])\n\n\n"
     "def consecutive_strict_declines(history):\n    count = 0\n"
     "    for index in range(len(history) - 1, 0, -1):\n"
     "        if history[index]['hits050'] >= history[index - 1]['hits050']:\n            break\n"
     "        count += 1\n    return count"),
    ("        assert args.val_freq == 1 and args.start_epoch <= args.max_epoch",
     "        assert args.val_freq == 1 and args.start_epoch == 1 and args.max_epoch == 8\n"
     "        assert not args.checkpoint_path\n"
     "        stop_reason = 'max_epochs'"),
    ("            record_native_metrics(args, 0, initial)",
     "            record_native_metrics(args, 0, initial)\n"
     "            args._epoch_metric_history = [dict(epoch=0, **initial)]"),
    ("            record_native_metrics(args, epoch, metrics)",
     "            record_native_metrics(args, epoch, metrics)\n"
     "            args._epoch_metric_history.append(dict(epoch=epoch, **metrics))"),
    ("            save_checkpoint(args, epoch, model, optimizer, scheduler, save_cur=True)\n"
     "        return os.path.join(args.log_dir, 'best.pth')",
     "            save_checkpoint(args, epoch, model, optimizer, scheduler, save_cur=True)\n"
     "            if consecutive_strict_declines(args._epoch_metric_history) >= 2:\n"
     "                stop_reason = 'two_consecutive_strict_declines'\n"
     "                self.logger.info('Stop after two complete Acc@0.5 epoch-to-epoch declines')\n"
     "                break\n"
     "        with open(os.path.join(args.log_dir, 'native_stop.json'), 'w') as stream:\n"
     "            json.dump(dict(reason=stop_reason, history=args._epoch_metric_history,\n"
     "                retained_metrics=args._retained_native_metrics, max_epochs=args.max_epoch,\n"
     "                equality_or_increase_resets=True, full_goal_complete=False), stream, indent=2)\n"
     "        return os.path.join(args.log_dir, 'best.pth')")]
for before, after in changes:
    assert text.count(before) == 1, before
    text = text.replace(before, after)
ast.parse(text)
output.write_bytes(text.encode('utf-8'))
result = dict(status='COMMON_NATIVE_LOOP_OVERLAY_PREPARED_NOT_EXECUTED',
    original_sha256=hashlib.sha256(raw).hexdigest(), output_sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
    scope='future native_control and selected_geometry only; current MCLN unchanged',
    maximum_epochs=8, patience=2, monitor='complete9508_last/bbs_hits050',
    history_includes_E0=True, equality_or_increase_resets=True,
    new_best_order='both60/52_passed_then_hits050_then_hits025',
    retrospective_old_checkpoint_selection_changed=False,
    cold_resume_supported=False, fresh_initialization_required=True,
    optimizer_scheduler_and_global_RNG_saved_by_existing_native_checkpoint=True,
    early_stop_history_saved_in_checkpoint_config=True,
    final_accuracy_asserted=False, new_neural_forwards=0, GPU_calls=0, remote_queries=0,
    normal_launcher_prepared=False, normal_training_launched=False)
(root / 'NORMAL_LOOP_OVERLAY_V2.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
print(json.dumps({key: result[key] for key in ('status', 'maximum_epochs', 'patience', 'new_neural_forwards')}))
