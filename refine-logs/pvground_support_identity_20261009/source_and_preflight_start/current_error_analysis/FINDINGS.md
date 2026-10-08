# Current retained model: descriptive error partition

Actual local analysis of all9508 immutable saved rows from the retained5599/4859
model, plus191 previously archived member cases. No neural forward, optimizer
update, new weight or metric promotion. This is executor descriptive analysis,
not an independent experiment audit.

| Acc@0.25 error geometry | Rows | Saved Mask IoU>0.5 |
|---|---:|---:|
| No intersection with root GT Box | 3074 | 0 |
| Root coverage>=0.5, Box precision<0.5 | 344 | 74 |
| Box precision>=0.5, root coverage<0.5 | 106 | 30 |
| Other partial overlap | 385 | 6 |
| Total | 3909 | 110 |

Root coverage is intersection/GT volume; Box precision is intersection/predicted
volume. These groups describe geometry and do not prove semantic instance identity.
The known validation GT pool contains2068 expressed targets in141 scenes, not all
scene objects.2673 wide errors overlap another expressed target Box>0.25, and3367
have a higher overlap with some other expressed target than with their root.
This is a proxy for investigating instance selection, not an oracle deployment rule.

The191 member cases were selected by old reference-model errors. Current full
point hashes and root targets agree with their archived slices.180 remain wrong
at0.25,187 at0.5;190 have valid current references.168 have an overextending face
whose possible coordinate sources are all background. Of761 overextending faces,
760 have only pure-background superpoints as possible sources. All180 missing
faces have an observed target member beyond them. No current foreground set or
old Query was reconstructed. This panel cannot estimate full-validation prevalence.

Implication: mixed-superpoint subdivision alone is not the first supported cause
in this panel. Most wide errors also have no target Box intersection. The next
controlled structural experiment should therefore ask whether final instance
judgement benefits from reading the actual candidate support's visual content.
It should preserve the strong Mask geometry, one native score and all256 candidates.

Suggested comparison: common Text support versus candidate fused support, using
the same zero-initialized semantic residual and unchanged native+G supervision.
Previous generic P2 attention and boundary-statistic R are historical negative
controls; new Mask-conditioned content is a hypothesis, not a promised gain.
