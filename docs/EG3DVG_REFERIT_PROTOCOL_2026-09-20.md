# EG-3DVG Nr3D/Sr3D resource and protocol check

This is CPU input preparation while the independent ScanRefer checkpoint upload
continues. It performs no model forwards, optimizer steps or Nr3D/Sr3D training.
Source is the same pinned official EG3DVG commit
`174e34894aea6513442da6b5dfa9b3e2bf8a1efa`.

## Actual input availability

The public loader expects `ReferIt3D/`, whereas the existing input store uses
`refer_it_3d/`. The first read failed at that literal path and was retained.
The check then read the existing files explicitly; no data was moved and the
active ScanRefer source/configuration was not modified.

Applying the conditions read from the author's loader to the actual CSV files:

| Dataset | Evaluation rows | Scenes | Selection condition |
|---|---:|---:|---|
| Nr3D | 7899 | 130 | test scenes and correct_guess=True |
| Sr3D | 17726 | 255 | test scenes and mentions_target_class=True |

Nr3D does not additionally remove descriptions that omit the target class.
All selected scenes have existing mesh-superpoint files and entries in the
author's `cls_results.json`. This check establishes file coverage and annotation
counts, not validity of every object slot, model compatibility or performance.
It does not construct a full Nr/Sr Dataset or rerun scene-graph parsing.

Actual CSV, split and class-file SHA256 values are in
`refine-logs/eg3dvg_pretrained_acceptance_20260920_v1/referit_protocol.json`.

## Author's actual output protocol

Both author test scripts enable `butd_cls`. Their object inputs use scene GT
instance boxes and predicted classes from `cls_results.json`, not the target's
GT class and not GroupFree predicted boxes. This must be reported as an object
proposal prior, not as an entirely GT-box-free pipeline.

`train_dist_mod.py` sets `filter_non_gt_boxes=args.butd_cls`. The evaluator
multiplies each query's score by a binary test: its predicted box must have
IoU > .25 with at least one valid scene object-input box. It then sorts the
original query array. Multiplying by zero is not equivalent to discarding a
query or replacing its score with negative infinity.

The selected output remains the average of its regressed box and predicted-mask
box; it is not snapped to the nearest GT object box. Acc@.25/.50 on those output
boxes must not be relabeled as discrete GT-object selection accuracy.

Current MCLN's ordinary `train_dist_mod.py` also connects the filter flag to
`butd_cls`. The separately prepared PV ReferIt evaluator explicitly disables
that flag. Therefore model-family results cannot silently share a single
"native" protocol label. Bind the actual input, filtering and box rules to any
future EG comparison.

## Verified resource boundary and next action

The pinned official README links one complete ScanRefer checkpoint. The public
[release page](https://github.com/Gwan9Wook/EG3DVG/releases), checked on
2026-09-20, contains no releases. No dedicated EG Nr3D/Sr3D checkpoint has been
verified. The generic `ckpt.pth` in test scripts does not prove their existence.
No message has been sent to the authors.

First finish the complete ScanRefer pretrained acceptance already queued.
Then choose and explicitly bind a real Nr/Sr initialization and a fixed-budget
native control before introducing one new mechanism. Using ScanRefer weights
on Nr/Sr without adaptation would be a transfer experiment, not a reproduction
of a dedicated author's Nr/Sr model. No old PV conditional training queue is
reused for EG, and no long baseline retraining or multi-seed run is started.
