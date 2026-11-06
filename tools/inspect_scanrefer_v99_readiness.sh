#!/usr/bin/env bash
set -euo pipefail
/root/autodl-tmp/PV-Ground/env/bin/python - <<'PY'
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

root = Path('/root/autodl-tmp/mcln_pvground_ft_20261101')
source = root / 'scanrefer_restart_20261102/stage_b_source'
runner = root / 'scanrefer_qila_20261105/train_scanrefer_qila.py'
specifications = [
    (runner, ('def audit_checkpoint', 'model_only_initialization',
              'def recover_semantic_rows', 'audit_checkpoint(checkpoint, epoch',
              'bbf_quality_rank_loss_weight', 'qila_', 'BatchNorm',
              'batchnorm', 'freeze_bn')),
    (source / 'train_dist_mod.py', ('use_candidate_local_visual =',
              'use_pvground_innovations',
              'if use_candidate_local_visual:', 'model.decoder[-1].local_visual',
              'def train_one_epoch', '_set_source_moe_train_mode')),
    (source / 'main_utils.py', ('def _set_source_moe_train_mode',
              'model.train()', 'BatchNorm', 'freeze_bn',
              'use_qila', 'qila_topk', 'qila_tau_u', 'qila_ignore_iou',
              'pvground_span_gate_loss_weight')),
    (source / 'models/losses.py', ('def loss_sem_align', 'def loss_pos_align',
              'def loss_qila', 'qila_hardneg', 'qila_ignore', 'qila_tau_u',
              'bbf_quality_rank_loss_weight',
              'pvground_span_gate_loss_weight')),
]
files = []
for path, tokens in specifications:
    data = path.read_bytes()
    lines = data.decode('utf-8').splitlines()
    matches = [
        {'line': index + 1, 'text': line.strip()[:280]}
        for index, line in enumerate(lines) if any(token in line for token in tokens)
    ]
    files.append({'path': str(path),
                  'sha256': hashlib.sha256(data).hexdigest(),
                  'matches': matches})
print(json.dumps({
    'observed_utc': datetime.now(timezone.utc).isoformat(),
    'scope': 'Current ScanRefer runner and its deployed source only; file reads and hashes. No model import, checkpoint load, forward, test, evaluation, training, mutation, or stopped-run inspection.',
    'files': files,
}))
PY
