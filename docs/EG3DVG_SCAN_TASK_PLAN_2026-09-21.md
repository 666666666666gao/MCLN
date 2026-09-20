# ScanRefer first: EG task-conditioned reading on the published checkpoint

## User priority and success condition

The user now explicitly prioritizes ScanRefer. The pending Nr3D task-read controller was cancelled before its initial GPU check or any optimizer update. Nr3D/Sr3D training must wait until a single ScanRefer model beats the actual native pretrained EG result on both formal REC thresholds. The nearly finished Sr3D baseline evaluation was allowed to finish; its rendering and training follow-ons were deferred.

The actual native EG reference is 5542/4952 of 9508 (58.2878/52.0825 percent). Strict improvement requires at least **5543/4953** from the same checkpoint and last/bbs path. The separately protected V99 result 5572/4797 remains preserved. Mask is not a promotion gate; native Mask losses and the author's average of regressed and Mask-derived boxes remain part of EG.

This is a ScanRefer development campaign on the established validation split, not an untouched-test claim. An adaptive decision to stop after meeting the prespecified REC condition is reported explicitly. Individual thresholds, epochs or bbf outputs will not be combined into one result.

## Transferred mechanism

Use the complete native EG-3DVG pinned at 174e34894aea6513442da6b5dfa9b3e2bf8a1efa and its official ScanRefer epoch69 checkpoint SHA256 785f46ca595aed3a06efdfc46dc1c4b0493b212ccb66062a76f519324fd24f7e.

After last-layer self-attention, PECA text and object attention, semantic and geometric requests are q+W_sem q and q+W_geo q. The two zero-initialized 288-by-288 matrices add 165888 parameters. Both requests access EG's original GMA point memory and SWA superpoint memory with shared native attention, normalization and feed-forward parameters. Their training dropout masks are shared, advancing the RNG only once. Semantic features feed the native soft-token and contrastive heads; geometry feeds center, size, Query Mask and last superpoint refinement. Matching and final instance selection remain shared.

This transfers the task-reading idea, not every historical PV module. There is no PV six-source VSA interface in EG and no claim that observation-state C or semantic-assignment G has already been transferred. No V99 sidecars, extra quality head or new auxiliary loss are included. The negative PV D result remains part of the history.

## Shared input correction

Code inspection confirms EG's supplied GroupFree box augmentation rotates before flipping, while its point-cloud augmentation flips before rotating. Both training arms use the corrected flip -> Z/X/Y rotation -> translation -> scale order. An independent actual-GroupFree check compares transformed box centers and extents to a matrix-based calculation. This repair is shared by the control and candidate and is not counted as a method contribution. Evaluation has no augmentation and retains the previously accepted native rules.

ScanRefer train annotations are generated with EG's own loader and scene-graph parser. All native ScanRefer training expressions are combined with the original ScanNet training detection prompts repeated ten times, as in the author's joint training. Validation scenes must be disjoint. Use RGB points, 50000 points per scene, GroupFree predicted boxes, butd=true, butd_cls=false, butd_gt=false and author augment_det=true. Do not reuse the Nr3D GT-object input protocol.

## Bounded first campaign

- Fixed seed2027, batch8, one A100, verified Torch1.12.0+cu116 runtime; no environment rebuild.
- Task and native controls independently load the same published weight. Both create fresh AdamW using the three learning rates saved in that published checkpoint. Rates stay fixed during this campaign; weight decay0.0005 and clip0.1 follow the native code.
- Keep all originally trainable native parameters, native matching/losses and positive/negative-expression forwards. Only the task arm installs the new matrices.
- Two discarded training batches test actual optimization before starting either fitted arm. The startup check replays the last layer on captured real ScanRefer GPU inputs and requires exact semantic, geometric, refined-superpoint outputs and RNG. Whole-model native repeat and task differences are recorded separately; the native model itself was measured to be non-bitwise-stable with fixed inputs, weights and RNG. No numerical tolerance is used to hide a mismatch in the inserted layer.
- Up to three complete continuation epochs, each visiting the full joint train set once. A single saved permutation per epoch is shared between arms. Log every step and augmented point hash; preserve every epoch's formal metrics, including negative results.
- Run task epoch1 and its formal evaluation first, then native epoch1 and its formal evaluation. Compare paired rows and training input hashes. If the task checkpoint exceeds both pretrained thresholds, stop this campaign at that complete paired epoch. Otherwise continue both arms to epoch2, then epoch3 under the same frozen setup.
- The same-budget native comparison determines whether the task mechanism adds value beyond ordinary continuation. Beating the pretrained reference alone does not establish innovation gain.
- Each fit checkpoint contains model, optimizer and RNG states. The latest checkpoint is atomically replaced for recovery and next-epoch continuation; no intermediate checkpoint is selected from training loss. Full-epoch evaluations are prespecified decision points, not a retrospective best-epoch search.
- No automatic Nr3D/Sr3D training launch is included. If this bounded campaign fails, preserve all results and continue ScanRefer research; the overall objective is not complete.

## Evidence and execution

Remote campaign: /root/autodl-tmp/mcln_eg3dvg_scan_task_20260921_v1.

Scripts: prepare_eg3dvg_scan_task_inputs.py, check_eg3dvg_scan_box_alignment.py, stage_eg3dvg_scan_task.py, preflight_eg3dvg_scan_task_initial.py, train_eg3dvg_scan_task.py, evaluate_eg3dvg_scan_task.py and control_eg3dvg_scan_task.py. Evaluation reuses the established CPU recount auditor. Input/cache hashes, exact row counts, checkpoint learning rates and script hashes are frozen in input_receipt.json, campaign.json and per-arm spec.json before optimization.

Expected cost must be updated from measured ScanRefer throughput. The previous native EG run took about 2.9 seconds per batch; the extra last-layer read may cost more. A complete ScanRefer formal pass was approximately29 minutes. These are estimates, not new accuracy results.

The official code and license attribution remain intact. CPU module checks established implementation behavior, not accuracy. No task-read ScanRefer result exists at plan creation.
