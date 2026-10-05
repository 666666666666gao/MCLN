# Query-supported geometry: auxiliary target coordinates

Status: PREPARED, not launched. This follows the completed4509/4499 responsibility comparison and the CPU64 native-target jitter check. Existing goals and formal evaluation remain unchanged.

The CPU check exactly reproduced augmented points and native noisy root labels. It found that395/1090 originally qualified Mask-supported, unmatched Box-poor candidates already exceed0.5 against the annotation-member box before independent native box jitter. Target qualification becomes861 rather than1090; fixed-cohort outside-range candidates decrease201→87. This is a bounded training-data counterfactual, not a bug finding or proof of formal-accuracy cause.

Next direct comparison: both arms restore the current full4509 geometry head, officialPV plus originalG, fresh optimizer, native+G+matchedDFL unchanged, additional geometry weight1. Parent/core/Mask/semantic/zeroR states frozen. Only the auxiliary geometry reference changes:

- control: native independently jittered root GT, reproducing the previous responsibility strategy.
- member_target: annotated object-member box from the same augmented native50000-point scan representation BEFORE independent GT-box jitter. Use this box consistently both for Box<=0.5 auxiliary qualification and extra L1/GIoU/DFL targets. This is the native annotation lookup, not a model-mask box or a claim about original full-resolution scans.

Both require ownQuery AND fused GT Mask IoU>0.5; exclude every original matched candidate; per-expression then batch average; empty rows zero. Native matching, native GT boxes, native losses, G labels, detection inputs and all evaluation fields are unchanged. Capture the extra root box without consuming RNG or changing native returns. The auxiliary change includes qualification and target definition; do not claim an isolated loss-only change.

No new model parameter, text/visual attention, reference head, score head, teacher, inference GT or candidate pruning. All256 and one last/bbs selecting sameQuery Box/Mask. Current33-node six-face geometry head stays456102 parameters.

Each arm:29778 fit rows once, batch8, accumulation1,3723 optimizer updates, tail batch2, seed2027, lr1e-5, WD0.0005, clip0.1. Initial/terminal6887 seen-scene module holdout and formal9508, augment/augment_det disabled before eval loaders. Current parent geometry has7446 fit updates; new terminal cumulative11169, originalG history separately reported.

Before formal fit, fresh source review and two-update realGPU sanity per arm must pass: exact shared native data, nonzero extra gradient only on qualified outputs, every native match excluded, frozen parent/Mask/bbs, finite gradients, memory-only strict save/restore, no weights. Sanity state never initializes formal fit. Source/result audits use fresh Astra/max requested, same-family/provisional when backend unattested.

Retain4509 until a verified formal better Acc@0.5 result; record Acc@0.25 and Mask costs. Delete our closed nonbest weights promptly, no new negative archives. OfficialPV/originalG and V99 chain remain protected. New files never reference the deleted4506/control heads.

Estimated pair budget about4.5hours using the last closed two-arm fit. First meaningful check near completion of control training+6887 evaluation, then180-300 seconds if incomplete. Preflight estimated15minutes, first check300seconds. No run is launched by preparation alone.
