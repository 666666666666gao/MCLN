SOURCE_ONLY verdict: **PASS**. No remaining blocking or nonblocking correctness findings in the sealed source. The two concrete blockers below were repaired during this review and the final source was read again.

This is a **same-family/provisional** review; the actual serving backend and independent backend calibration are **not attested**. Actual four-run M0, model construction, numerical equivalence, backward execution, optimizer restoration, GPU capacity and formal accuracy remain **unattested**. Future M0 is not evidence yet.

1. **Resolved: different Gumbel samples in the zero-initialization comparison.** Native PVGround calls `self.gumbel` at `models/pv_ground.py:316` and unconditionally invokes `F.gumbel_softmax` at line 609, including eval mode. The original two forwards did not replay RNG. The reviewed script now resets Python, NumPy, torch CPU and CUDA RNGs before both comparison forwards, at lines 119 and 136.

2. **Resolved: baseline forward before required observation installation.** The pinned VSA source unconditionally invokes `spatial_observation`, whose helper reads `grouper.observation`; native OpenPCDet `QueryAndGroup` does not create that attribute. The reviewed script installs observed groupers immediately after strict official loading, temporarily detaches the reader and sets `task_read=False` for the native baseline, then reattaches the same reader with `task_read=True` before the completed model's CUDA/eval conversion. This preserves the native Tensor branch and avoids the task reader's tuple entering it. The literal source generator reconstructs `pv_utils.py` to the exact registered SHA `3e8c069af235970c541aa07d9780e812e609d798d242a013d64abdf89307bc29` (23,949 bytes); this was text reconstruction, without module execution.

The original preflight SHA was `787d25def619a2386b3a5299f72cf286712e1475a7b7674a778b6b754c484022`; the RNG-only intermediate SHA was `fd26ca1d6da3f7c8d7990de28039dcb3848103266eb4934bd4278e4e50fcb078`. Both original findings are resolved in the final reviewed script:

| Sealed file | Bytes | SHA-256 |
|---|---:|---|
| referit_model_preflight.py | 15252 | `512f21026646458ad68dae8b4b1d37cf18ed3b1bfdb91dec9045a0a08cce2b34` |
| referit_training_targets.py | 6631 | `4718f1480f7f3502eeea0152ed4cb02cacaa3c00fc96ccc3e214ebbdb02c4eb6` |
| preflight_spec.json | 16877 | `7c9e5272eb9ffc59aa9f3f71a810678795961e01deb09e9626e7ca93e1fb1103` |
| preflight_controller.py | 1861 | `f5959600501270b277ecdeb5e254450c23bf74784b2ca0fcdbc6f30572e55da6` |
| launch_preflight_authorized.py | 4333 | `e8f70840b406fbc0d69cfbbc1c5a1029edf901168768a8e6cafcf0b711ca42a4` |

The substantive source checks passed:

- Native loss source uses Nr weights .6/.2/.2/.1 and Sr .625/.125/.125/.125, preserving unnormalized target mass, entropy, eos .1 and the last-layer **1/7** scale. The correction and its reconstruction agree with those formulas.
- `language_dataset` selects the native recipe; `sample_dataset` excludes ScanNet detection rows from new root targets. Detection semantic direct gradients are explicitly checked; extra geometry has no detection terms by construction. The full batch denominator remains eight.
- Native matching calls proposal first and last layer second, so `matches[1]` is correct. GT arrays are filtered by `box_label_mask` before match target indices are applied. Both new root routes protect every matched Query, and the base DFL covers all valid matches.
- The real panel is four saved fit referring IDs plus four detection IDs, uses native augmentation and the declared `butd_cls` instance-box/predicted-class protocol, and repeats one actual collated batch twice. GT eligibility and dataset tags are not model inputs.
- Each run strictly restores its corresponding 1235-state official checkpoint, then adds the fresh 37/10/23 states. No Scan G, geometry or R weight is loaded; the Scan-specific factory is not called. R is frozen, its output remains zero, and each existing Query has one final box.
- Initial geometry output is zero relative to its registered reference. All-256 raw-member extents have an independent planned witness. The zero-output numerical comparisons now use the correct native branch and matching RNG.
- Full-model native loss, new losses, backward, clipping, AdamW updates, frozen-parameter checks and complete model/optimizer BytesIO serialization/load/strict equality are present. The final CUDA conversion includes the restored reader and new heads. No checkpoint file is written.
- Native loss, main_utils, prepare_data, PVGround/modules, pv_backbone and decoder copies match their registered source-port digests. All 17 helper hashes and all 24 per-run source-file registrations match. The fixed sequential controller and exclusive isolated-root deployment are scoped to the four preflights.

The JSON report contains all **42** reviewed paths with byte counts and SHA-256 digests, detailed evidence, resolved findings and source-only limitations. Inspection used local reads/searches/hashing only. No project scripts, SSH/network actions, model loads, GPU work, package installs or deletions were performed by the reviewer. Only these two review artifacts were written.

