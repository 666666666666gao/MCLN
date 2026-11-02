# Fixed-Checkpoint Forward Audit

Status: 15 focused CPU tests passed; the GPU evaluation entry point is unexecuted.
This tool does not attach to, change, stop, or resume an existing trainer.
Run it only when a GPU is available for isolated evaluation.

The retained ScanRefer epoch8 checkpoint has SHA256
`3572193fef0f32d1d6ca22b41cbbb02932a30de0a51b18deb3b0c40ee37bcf38`.
Use its fixed preserved path for every condition. The keeper's best alias
may change while training continues.

From the deployed source directory, an evaluation command is:

```bash
root=/root/autodl-tmp/mcln_pvground_ft_20261101
export PYTHONPATH="$root/source/pointnet2:$root/source"
cd "$root/source"
/root/autodl-tmp/PV-Ground/env/bin/python -m torch.distributed.launch \
  --nproc_per_node 1 --master_port 29511 \
  tools/eval_pvground_audit.py \
  --audit-jsonl "$root/scan-e8-none-forward.jsonl" --audit-bypass none \
  --eval --checkpoint_path "$root/runs/scanrefer/preserved_checkpoints/ckpt_epoch_8.pth" \
  --expected_eval_sample_count 9508 --rng_seed 0 \
  --num_decoder_layers 6 --use_color --weight_decay 0.0005 \
  --data_root /root/autodl-tmp/PV-Ground/data/ \
  --batch_size 8 --lr_backbone 2e-5 --lr 1e-5 \
  --dataset scanrefer --test_dataset scanrefer \
  --detect_intermediate --joint_det \
  --use_soft_token_loss --use_contrastive_align \
  --butd --self_attend --augment_det --use_pvground_innovations \
  --candidate_local_visual_lr 1e-5 \
  --log_dir "$root/offline_audits/scan-e8-none" \
  --model MCLN --exp scan_e8_none
```

The helpers (`eval_pvground_audit.py`, `pvground_forward_audit.py`, and
`pvground_semantic_audit.py`) must first be copied into the deployed source's
tools directory.
The command intentionally loads the full trained state, without
`model_only_initialization`, `checkpoint_start_epoch`, or initialization-only
`pp_checkpoint`. It has not been validated against remote CUDA dependencies.
Do not run concurrent with the occupied training GPU.

Repeat sequentially with `reader`, `support`, and `span`, changing the audit
output, log directory, and experiment name for each condition. Keep all
other arguments, source, checkpoint bytes, preprocessing, seed, and batch
unchanged. Record the checkpoint epoch from its CPU audit, the full command,
source commit, and full semantic evaluator log beside each JSONL.

The bypasses are:

- Reader: replace only its residual with zeros.
- Support: keep its measured superpoint geometry, return original member
  mask logits, and let downstream reference/Span recompute from those logits.
- Span: return the original native center and size exactly.

These are interventions on a trained network. They do not replace separately
trained ablations and do not establish gains over official initialization.
The official initialization also needs a matched complete evaluation before
claiming improvement over MCLN.

Each JSONL begins with checkpoint SHA256 and invocation metadata. Sample
records contain candidate counts, reference-valid counts, raw gate bounds
and clipping counts, gate means, and mean absolute axis displacements.
Additional `semantic_sample` records contain scan/target/utterance identity,
input point SHA256, unique/multiple group, selected query and raw/clamped
gate, selected and all-candidate oracle IoU, first good candidate rank at
strict025/050 thresholds, top10 candidates/IoUs, and native/reference/final
selected boxes with signed per-axis center/size errors against the target.

The completion footer is emitted only after both forward and semantic
records cover exactly9508 examples and per-example025/050 hit counts agree
with the evaluator's own last/bbf Top1 counters. Position/mask metrics must
not be substituted. Failed/incomplete evaluations have no completion footer.
The standalone evaluator's normal full log remains an independent artifact.

The semantic helper follows the current root-only ScanRefer branch without
detector filtering. Tests execute the tracked evaluator's own semantic
method and box IoU functions (isolated from CUDA-only imports), compare
selected boxes/IoUs/counters on three seeds, reject a deliberately altered
counter, and check strict-threshold rank behavior. This establishes CPU
agreement on those fixtures, not remote GPU validation.

All-candidate gate statistics can differ from the selected query distribution;
selected gates are recorded separately. All-candidate oracle coverage uses
the current predicted boxes, not ground-truth candidate boxes. Displacement
is distinct from the recorded GT errors. A module bypass can change rankings
and boxes together; paired identities and full metrics are needed to interpret
differences. These diagnostics do not themselves prove a causal failure.
No real gate-occupancy or bypass-accuracy result is available yet.
