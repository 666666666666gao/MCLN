# Fixed-Checkpoint Forward Audit

Status: CPU hook tests passed; the GPU evaluation entry point is unexecuted.
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

The helpers must first be copied into the deployed source's tools directory.
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
The footer confirms only that the forward loop returned, not metric
completeness. Match 9508 sample records AND the evaluator's complete
9508-example last semantic alignment Top1 counts. Position/mask metrics
must not be substituted.

All-candidate gate statistics can differ from the selected query distribution.
Displacement is not localization error. Ground-truth box errors, selected
query ranks, example IDs, and per-example predictions are not exported here
and remain required before diagnosing a ranking/localization failure.
No real gate-occupancy or bypass-accuracy result is available yet.
