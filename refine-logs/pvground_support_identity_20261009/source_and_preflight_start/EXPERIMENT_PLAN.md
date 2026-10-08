# Candidate support content and native instance judgement

Status: implemented source draft; no GPU preflight or trained result yet.

Baseline remains PV-Ground. Start from the protected content checkpoint
6f9109cb690399880b8dc3bcf6e4b98d1053145732ee2a6b504bfaecbd1eaac2,
full9508 hits5599/4859. User targets are strictly>59.5/>51.0,
minimum5658/4850 in one complete model. Three effective contributions and final
independent author-pretrained Nr3D/Sr3D remain required; seed2027 only.

## Evidence and question

Executed current error analysis:3909 wide-threshold errors,3074 zero root Box
overlap,110 with saved Mask IoU>0.5. A pool containing2068 expressed GT instances
finds2673 wide errors whose selected Box overlaps another expressed GT>0.25.
These are geometry proxies, not proof of physical or language identity.
The191 historically selected member cases have168 current references with
overextending background-certain faces;760/761 overextending faces have only
pure-background superpoints as possible coordinate sources. Current point
identity matches the archived member slice, but current foregrounds were not
reconstructed. Mixed-superpoint subdivision is not the first supported cause.

Current corrector changes Query Mask and derived Box geometry; semantic scores
are frozen. Previous P2 general cross-attention and boundary-statistic R were
negative. This experiment reads visual content belonging to the actual predicted
candidate support, not additional boundary-quality statistics or a second score.
Whether that distinction helps must be tested, not assumed.

## Single structural change

The existing final semantic Query already contains language. For each candidate,
pool frozen288D superpoint visual content over foreground superpoints, weighted
by actual member counts. Use exactly the deployed sigmoid(logit)>0.5 calculation,
including its floating-point behavior. Compare projected Query/support content with a107040-parameter
residual MLP. Final native sem_cls head still executes exactly once.

Control: Text-only foreground gives each candidate common expression support.
Treatment: actual fused foreground gives each candidate its own deployed support.
All other code, parameters, initialization, parent outputs and losses agree.
Both new heads start with identical states and zero residual output. Empty support
has zero pooled content;40 invalid selected references are actually observed.
No foreground confidence/quality scalar is broadcast into token logits.

The hook replaces the old identically zero geometry R. Parent, G reader, trained
Mask corrector, Mask fusion, Box reference and contrastive projection are frozen.
No new textual cross-attention, Box regression, teacher, score mixing, top-k
pruning, GT inference gate, seed search or dataset-ID gate is added.

## Sanity before full training

Fresh source review, then actual B8 on the existing warm A100 environment.
Strictly reconstruct the best model from official/G/reference/support deltas.
Verify actual final semantic-head call count1, zero-init score/Box/Mask parity,
nonzero output-layer gradient and upstream gradient after output update, frozen
parent states, only new head optimizer parameters, save/load both model and Adam,
and actual integrated PV forward. Engineering checks are not accuracy results.
Do not restore prior preflight optimization into the formal experiment.

## Formal comparison after sanity passes

Use one shared frozen parent forward for both heads; identical source/input and
fresh optimizer initialization. Original native+G criterion, original final CE
coefficient, Hungarian matching and full GT protections retained. Only the new
head receives gradients from its last semantic CE route. Earlier contrastive,
Mask and geometry outputs remain unchanged; do not broaden G or add losses.
Physical/effective B8, accumulation1, LR1e-5, weight_decay5e-4, clip0.1,
29778 fit examples once /3723 updates, seed2027. No formal validation examples
enter training.6887 module holdout is author-seen-scene development, not formal
generalization. Report full9508 native last/bbs and same-query Mask per arm.

Success requires benefit over same-budget control and retained5599/4859;
joint5658/4850 is the user target. Also report per-row repairs/damage and support
content-selected identity proxies. A result that only beats a weak control is
insufficient. The107040-param module is not a proven third contribution yet.

## Resource and monitoring

Single A100 serial campaign, unchanged warm env hash
966235b2ead7fa5a1de63e9537745b457374a4e5749ac4a7a26566b7b630c82c.
Measure actual preflight capacity/throughput before sealing formal ETA.
First observe near estimated endpoint, then240sec only if the original process
is still live. Archive text/results and keep only selected best plus necessary
restore dependencies under user's standing cleanup authorization.
