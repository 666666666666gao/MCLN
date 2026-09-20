# EG-3DVG Nr3D zero-update candidate analysis

This is an offline analysis of the already audited 7899-expression evaluation using the author ScanRefer epoch69 checkpoint. It performs no model forward or optimization and does not change the active adaptation or native inference rules. It is not an Nr3D-specific pretrained model result.

| Count | IoU > .25 | IoU > .50 |
|---|---:|---:|
| Actual primary bbs hits | 3508 | 2909 |
| All256 raw regression boxes: GT oracle | 6702 | 5403 |
| All256 native averaged boxes: GT oracle | 6793 | 5789 |
| Native averaged boxes with author object support: GT oracle | 6793 | 5789 |
| Failures with no qualifying native box | 1106 | 2110 |
| Failures with a qualifying native box not selected | 3285 | 2880 |

The identical supported and unfiltered oracle counts are protocol-dependent: scene GT instance boxes, including the target object, are available as native object inputs. A target-overlapping box can therefore satisfy object support. This is not evidence of ground-truth-free candidate retrieval.

Among failures with a qualifying box, the highest-scored qualifying candidate has the following rank in the actual bbs scores:

| Rank | IoU > .25 | IoU > .50 |
|---|---:|---:|
| 2 | 513 | 513 |
| 3–16 | 573 | 599 |
| 17–64 | 1211 | 1033 |
| 65–256 | 988 | 735 |

Both optimistic and pessimistic rank bounds were computed to handle equal scores without inventing a tie order. They agree for every reported failed sample; no best-qualifying score ties occurred. These are diagnostic ranks, not a proposal to select a different rank at inference.

For the same bbs-selected Query, the raw regression box obtains3391/2661 hits; the author average of regression and predicted-mask-derived boxes obtains3508/2909. At .25 averaging repairs130 and breaks13; at .50 it repairs300 and breaks52. This conditions on the already selected Query and does not isolate a causal training effect.

The existing author object-support score multiplication changes the same-forward bbs result from3240/2778 to3508/2909: repairs268/131 and breaks0/0. It is multiplication by zero, not hard candidate removal;11 selected rows have no object support. The unfiltered diagnostic still uses GT object boxes as model inputs and must not be called a single-stage result.

The data support two simultaneous limitations in this cross-dataset starting point: insufficient qualifying-box coverage and poor ranking of many existing qualifying boxes. They do not establish semantic identity correctness, a scorer-only cause, or a particular new loss as the solution. No validation-based threshold, output-head, training-budget, or checkpoint selection follows from this analysis.

The same analyzer is queued after the fixed Nr3D adaptation endpoint and independent audit, allowing a like-for-like comparison of candidate coverage, ranks, raw/native geometry, and actual selected counts. CPU waiter6387 launched19:20:20 CST and was confirmed live19:20:44. It checks completion every300 seconds, uses no GPU, and does not change the active controller or training spec.

At19:20:44, active Nr3D fit had logged960/5614 steps with finite loss/gradient; it had not reached its endpoint. No new adaptation REC is reported here.

Evidence: `refine-logs/eg3dvg_nr3d_transfer_20260920_v1/formal/candidate_analysis.json`. Source: `scripts/analyze_eg3dvg_nr3d_candidates.py`, SHA256 `da10a62e17bcebd0450dbc3f576ac003cdd40ee0af0820ae4ebf84655a75fae3`. Input candidate SHA256 `217bedf262c9e9d389652f0157ccc551706937a0c8958e107bc0326ffd887728`; input row SHA256 `1278b8188111c2fe02d6f3509e3bf6f4105093d525d8f38a7ae200696b5cf97b`. The analyzer verifies the existing audit and input hashes and reproduces the completed primary counts before recording the decomposition.
