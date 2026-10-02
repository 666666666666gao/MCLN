# Fixed-box P2 semantic diagnostic

Use the existing P2 step3723 checkpoint (SHA60371c5e...92d9e9) and the
unchanged full ScanRefer development validation protocol. The direct question:
does P2's final semantic evidence addition improve selection when all candidate
boxes are held to the actual trained P2 output? This is not a no-P2 training ablation.

One real model forward per batch. Cache the original D/G source reader's two
residuals before P2 addition. Reuse the trained final decoder tail to obtain
semantic values with and without that addition; return the actual semantic and
geometry outputs unchanged. Recompute only the existing semantic head for the
bypass. Both scoring paths use exactly the same actual candidate boxes and masks.

Do not change the source/model/checkpoint files, optimizer, sampling, score
definition, detector-assisted object-input protocol or random seed2027. Batch8,
no augmentation, eval/no_grad. No teacher, new loss, P3 or dataset inference gate.
Real GT is loaded only for offline IoU/evaluation; no GT enters model inputs.

Pilot8 followed by full9508 under the existing GPU lock. Require exact input
point/GT-row identity against prior formal records, recomputed actual semantic
logit equality, native/manual bbs hit agreement, identical candidate tensors,
unchanged model buffers and checkpoint hash. Save the results before the final
historical selected-query/hit assertion, preserving failures rather than retrying
or overwriting original formal results. Store all256 boxes, IoUs and both score
arrays in gzip JSONL; fresh output folders only. Archive scope/provenance with
receipts and never call the bypass path a different trained model.

Compare actual minus bypass: repairs and damages at0.25/0.50, selected-query
changes, Top16/32/64/256 geometric coverage. Scores and geometry are shared
except the semantic addition; any net benefit measures this direct forward
mechanism. It cannot establish why continued training lost original-G ability.

The controller runs the pilot then full serially, only if pilot passes. Its
storage check is based on actual compressed pilot bytes; it writes no model or
optimizer checkpoints. Estimate full runtime from the previous actual9508
forward (~17 minutes), then update using pilot/progress receipts. Poll only at
meaningful boundaries, with180–300s internal waits if still running.
