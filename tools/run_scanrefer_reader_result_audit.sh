#!/usr/bin/env bash
set -euo pipefail
root=/root/autodl-tmp/mcln_pvground_ft_20261101
trial="$root/scanrefer_reader_competitor_20261104"
audit="$root/offline_audits/reader_result_20261104"
staging="$root/offline_audits/observations_v2"
python=/root/autodl-tmp/PV-Ground/env/bin/python
if [ "${1:-prepare}" = run ]; then
    trap 'status=$?; printf "%s\n" "$status" > "$audit/pipeline_exit_code"' EXIT
    test ! -e "$audit/started.json"
    printf '{"pid":%d,"started_utc":"%s"}\n' "$$" "$(date -u -Is)" > "$audit/started.json"
    cd "$trial/source"
    export CUDA_VISIBLE_DEVICES=0
    export PYTHONPATH="$staging:$PWD/pointnet2:$PWD"
    export TOKENIZERS_PARALLELISM=false
    export OMP_NUM_THREADS=1
    checkpoint=$("$python" -c 'import json,sys; print(json.load(open(sys.argv[1]))["checkpoint"])' "$trial/reader_result.json")
    "$python" -m torch.distributed.launch --nproc_per_node 1 --master_port 29541 \
        "$staging/tools/eval_pvground_audit.py" \
        --local_rank 0 --num_decoder_layers 6 --use_color --weight_decay 0.0005 \
        --data_root /root/autodl-tmp/PV-Ground/data/ --batch_size 8 --num_workers 4 \
        --lr 2e-6 --lr_backbone 4e-6 --text_encoder_lr 0 --candidate_local_visual_lr 2e-5 \
        --dataset scanrefer --test_dataset scanrefer --detect_intermediate --joint_det \
        --use_soft_token_loss --use_contrastive_align --butd --self_attend --augment_det \
        --model MCLN --rng_seed 0 --use_pvground_innovations --eval \
        --expected_eval_sample_count 9508 --checkpoint_path "$checkpoint" --print_freq 100 \
        --audit-bypass none --audit-jsonl "$audit/none.jsonl" \
        --log_dir "$audit/logs" --exp reader_result_audit \
        2>&1 | tee "$audit/none.log"
    printf '{"completed_utc":"%s"}\n' "$(date -u -Is)" > "$audit/complete.json"
    exit
fi
test ! -e "$audit"
test "$(cat "$trial/pipeline_exit_code")" = 0
test "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)" -eq 0
CUDA_VISIBLE_DEVICES="" "$python" - <<'PY'
import datetime
import hashlib
import json
from pathlib import Path
import torch

torch.set_num_threads(1)
root = Path("/root/autodl-tmp/mcln_pvground_ft_20261101")
trial = root / "scanrefer_reader_competitor_20261104"
audit = root / "offline_audits/reader_result_20261104"
row = json.loads((trial / "reader_result.json").read_text())
checkpoint = Path(row["checkpoint"])
sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
assert sha == row["checkpoint_sha256"] == "67cef44cd6ec253806eced03b1ca79f115d13d4e681c9da69730ca767b0fda35"
saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
assert saved["epoch"] == 1 and len(saved["model"]) == 1169
assert saved["optimizer"]["state"] and saved["scheduler"]
assert not torch.cuda.is_initialized()
files = [
    trial / "source/models/candidate_local_visual.py",
    trial / "source/train_dist_mod.py",
    trial / "source/main_utils.py",
    root / "offline_audits/observations_v2/tools/eval_pvground_audit.py",
]
baseline = root / "offline_audits/b2_fixed_audit_20261103_lrpositive/none.jsonl"
assert baseline.is_file()
receipt = {
    "prepared_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "checkpoint": str(checkpoint), "checkpoint_sha256": sha,
    "own_cpu_epoch": 1, "model_tensors": 1169,
    "nonempty_optimizer": True, "has_scheduler": True,
    "source_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
    "baseline_jsonl": str(baseline), "expected_samples": 9508,
    "scope": "Read-only full evaluation; paired identities and point hashes required before comparing to B2. This is not the matched trained control.",
    "cuda_initialized": False,
}
audit.mkdir()
(audit / "preparation.json").write_text(json.dumps(receipt, indent=2) + "\n")
print(json.dumps(receipt))
PY
screen -dmS pvg_scanrefer_reader_result_audit bash /tmp/mcln_reader_result_audit_20261104.sh run
screen -ls | grep '[.]pvg_scanrefer_reader_result_audit[[:space:]]'
