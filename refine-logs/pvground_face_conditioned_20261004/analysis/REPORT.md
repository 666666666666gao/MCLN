# Face-conditioned boundary decoder: actual bounded result

OriginalG fixed/eval, common originalG initialization; all256 candidates, same whole/local predicted support.
New64737-parameter face-conditioned head vs completed456102-parameter flat distribution head; architecture and capacity change together.
Each seed2027 run29778 fit rows once, effectivebatch8/tail2,3723 updates, native final-box loss plus matched DFL/7.
6887 module rows were pretrained-seen scenes;9508 formal rows are development validation. No teacher or extra deployed score.

| Stage / mode | New hits.25/.50 | Flat control.25/.50 |
|---|---:|---:|
| initial/bbs | 6176/5602 | 6176/5602 |
| initial/bbf | 6206/5647 | 6206/5647 |
| terminal/bbs | 6176/5605 | 6174/5616 |
| terminal/bbf | 6205/5647 | 6203/5654 |
| formal/bbs | 5615/4496 | 5616/4506 |
| formal/bbf | 5656/4524 | 5653/4526 |

CPU selected-box threshold recount:0 changes. Raw Masks and full candidate boxes are not downloaded; oracle ranks are GT diagnostics only.
Cross-process numeric differences are counted in SUMMARY, not assumed absent. Same-checkpoint target5615/4754 unmet.
Primary formal bbs changes vs originalG: 0 / 1.
Fresh experiment-audit is pending; this analyzer is evidence computation, not an independent integrity verdict.
