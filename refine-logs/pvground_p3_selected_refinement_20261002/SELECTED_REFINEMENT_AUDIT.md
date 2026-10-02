**Overall verdict: WARN. Blocking findings: none for the bounded archived analysis and the current publisher's static review.** The numerical findings reproduce from the saved rows. Classification: **archived real-GT descriptive analysis**, not an independent training ablation, new GPU evaluation, or cross-dataset result.

Reviewer: **gpt-6-astra / max**, the same reviewer continuing the original fresh-context review; this transport follow-up is not a new fresh-agent audit, `review_independence: same-family`, `acceptance_status: provisional`. Date: **2026-10-02**. I performed local reads, independent standard-library CPU calculations, AST inspection and read-only Git queries. I did not run the analysis writer, publisher, SSH, a model, training or any Git mutation. Only the two requested private audit reports were written.

The reviewed publisher is now **10,262 bytes**, SHA256 **`26b772d6060e6723ef855dbc27bf88afa7a61e82b191baab439b3e93f2fa64f5`**. This verdict applies to that version. An earlier publisher invocation was stopped according to the preserved helper record; its pre/post-stop boundaries show the old document and no publication directory. **The revised publisher has not been executed by this reviewer, and completed publication is not certified.**

Path aliases used in the file:line references and inventory:

- `P` = `C:\Users\gb\.codex\tmp\pvground_p3_next_20261002`
- `C` = `P\complete`
- `S` = `C:\Users\gb\.codex\tmp\pvground_fused_support_20261002`
- `V` = `C:\Users\gb\.codex_pvground_cs_20261002`
- `M` = `C:\Users\gb\.codex_mcln_g0_20260905`
- `O` = `C:\Users\gb\.codex_mcln_v99_internal_20260928`
- `D` = `C:\Users\gb\Desktop\document`
- `X` = `C:\Users\gb\.codex\tmp`
- `T` = `P\.aris\traces\experiment-audit\2026-10-02_run02`
- `A5` = `C:\Users\gb\AppData\Local\uv\cache\archive-v0\1LtJrl7zuSwXxZ4kp9EN4\Lib\site-packages`
- `A35` = `C:\Users\gb\AppData\Local\uv\cache\archive-v0\TXS0M1iUoilooWiVo-l0k\Lib\site-packages`
- `AJ` = `C:\Users\gb\AppData\Local\uv\cache\archive-v0\J0TesX7kd6okgp2DfA3XP`

| Check | Status | Finding |
|---|---|---|
| Dataset GT versus proxy labels | PASS, bounded | Saved root boxes follow the dataset target path; G qualification is a separate geometric training heuristic |
| Independent descriptive recount | PASS | All 1,044 numeric result leaves agree; no count or subgroup discrepancy |
| Denominators and thresholds | PASS | GT-axis normalization, continuous row means and strict thresholds are distinguished correctly |
| Result existence and execution evidence | PASS, bounded | Completed CPU result binds the current analysis source and both archived row files |
| Future Nr/Sr interface | PASS, bounded | Frozen criterion, dataset fields and historical zero-forward contracts support the note |
| Publication implementation | PASS, static only | Six substantive inputs are review-hash-bound; bounded writes, prefix preservation, scoped staging and nonforce push |
| Transport follow-up | PASS, bounded | Three API additions preserve original claims and checks; saved stop boundaries are consistent |
| Scientific strength and units | WARN | One archived run; saved/clamped geometry; inherited physical units; no raw-data or live-runtime revalidation |

**1. Ground-truth provenance and geometry scope**

The loader reads ScanRefer annotations, maps annotation `object_id` to `target_id`, obtains the target box from `scan.get_object_bbox(tid)`, converts it to center/size and returns `center_label` and `size_gts`. The runner saves these as `root_box`. Dataset instance-point indices provide mask truth. Evaluation disables augmentation before sample retrieval. No model-generated reference replaces this target.

Evidence: `C\source\imported\src.joint_det_dataset.py:596`, `:634`, `:1102`, `:1106`, `:1389`; `C\run.py:382`, `:401`, `:407`. This verifies the source-to-saved-target path, not an independent reconstruction of every target from original annotation/mesh/point assets, which are outside the supplied list.

G's unmatched-query eligibility uses predicted/GT IoU >0.5 after excluding matched queries. That is a **geometric proxy**, not an object-identity label; it does not define evaluation GT. The old Nr helper explicitly states the distinction. Evidence: `V\models\pvground_semantic_assignment.py:9`, `:23`; `V\models\nr3d_semantic_assignment.py:5`, `:21`, `:35`.

All 16,395 `bbs` rows have finite six-element coarse/final/GT boxes and positive saved sizes. The original runner clamps coarse and final prediction sizes to approximately `1e-6`. Terminal has **19 coarse / 20 final** selected rows with a clamped dimension; formal has **34 / 34**. These statistics describe emitted, size-clamped boxes, not raw head residuals or proof that raw dimensions were valid. Evidence: `C\run.py:392`, `:395`, `:429`; examples at `C\p3\terminal\rows.jsonl:1399` and `C\p3\formal\rows.jsonl:180`.

Physical distances are verified in stored coordinate units. Metre/mm labels inherit the native ScanNet convention; the supplied rows do not independently encode scale calibration. No original point-asset calibration was revalidated. The evaluation path applies no new coordinate normalization: `C\source\imported\src.joint_det_dataset.py:896`, `:911`, `:1274`; `C\run.py:382`. The **current** analyzer/result, findings and appendix adequately state both the clamp and unit qualifications: `P\analyze_selected_refinement.py:107`, `:108`; `P\SELECTED_REFINEMENT_FINDINGS.md:10`, `:14`; `P\publish_selected_refinement.py:49`.

**2. Independent recomputation**

I parsed both full row files without importing the executor's analysis. Independent center/size-to-face calculations, quantiles, counts and every subgroup reproduce **1,044 numeric leaves** in the current result. **963** are bitwise equal as Python numbers; the largest remaining arithmetic difference is **1.3877787807814457e-17**. There are no substantive mismatches. Both row hashes agree with the original intake, stage receipts and current magnitude result. All eight requested artifacts with intake entries match their recorded byte counts and hashes.

Quantiles use linear interpolation at zero-based `(n-1)*p`, not nearest rank. The six faces are `c_i +/- s_i/2`; each row's displacement is their maximum absolute coarse-to-final change. Evidence: `P\analyze_selected_refinement.py:13`, `:27`, `:41`, `:45`, `:77`; `C\SELECTED_REFINEMENT_MAGNITUDE.json:18`, `:702`.

| Saved `bbs` stage | Rows / scenes | Median max face shift | P99 | Maximum | Coarse >0.50 | Final >0.50 | Repairs / damages / net |
|---|---:|---:|---:|---:|---:|---:|---:|
| Terminal module holdout | 6,887 / 106 | 3.267288 mm | 7.310977 mm | 9.808376 mm | 5,603 | 5,614 | 23 / 12 / +11 |
| Formal development validation | 9,508 / 141 | 3.260598 mm | 7.305460 mm | 9.310782 mm | 4,403 | 4,401 | 15 / 17 / -2 |

All selected terminal and formal boxes move less than 0.01 native coordinate units on every face, meaning 1 cm under the inherited metre convention. The same selected `q` indexes coarse and final boxes. This is **one `bbs` query per row**, not all 256 candidates. Evidence: `C\run.py:419`, `:424`, `:429`; `P\SELECTED_REFINEMENT_FINDINGS.md:10`.

The independently verified formal coarse-IoU groups are:

| Lower-inclusive / upper-exclusive interval | Rows | Repairs >0.50 | Damages >0.50 | Mean stored IoU delta |
|---|---:|---:|---:|---:|
| [0, 0.25) | 3,910 | 0 | 0 | -0.000097497788 |
| [0.25, 0.45) | 869 | 0 | 0 | -0.000949369184 |
| [0.45, 0.50) | 326 | 15 | 0 | -0.000413641151 |
| [0.50, 0.55) | 348 | 0 | 16 | +0.000597865931 |
| [0.55, 1.000001) | 4,055 | 0 | 1 | +0.001069502237 |

The seventeenth damage is in the final group. No archived coarse IoU equals exactly 0.50. All groups are nonempty and partition the rows. GT-volume groups are deterministic **row-rank quartiles**, sorted by `(GT volume, row_id)`, with integer slice boundaries. Counts are terminal **1,721/1,722/1,722/1,722** and formal **2,377 each**. Equal-volume ties can cross boundaries; these are not disjoint numeric volume intervals or independent object cohorts. Evidence: `P\analyze_selected_refinement.py:56`, `:88`, `:92`; `C\SELECTED_REFINEMENT_MAGNITUDE.json:765`, `:903`, `:971`, `:1039`, `:1107`.

**3. Denominators, continuous values and strict decisions**

Normalized displacement is `max_face(abs(delta_face) / corresponding_GT_axis_size)`, computed before row summaries. Normalized face-error delta is final minus coarse of the mean of six GT-normalized absolute face errors. Positive **dataset GT axis sizes**, not prediction statistics, are the denominators. Formal median normalized displacement is **0.005282466160**, maximum **0.259888945354**; small absolute movement need not be negligible relative movement for every small object.

Formal stored IoU delta has mean **+0.0003369608464006434**, median 0, and positive/negative/zero counts **3,587/2,734/3,187**. Terminal mean is **+0.0005657726417250589**. Formal strict hits fall by 2 despite the positive continuous mean. The current findings and appendix state this correctly. These are row-weighted summaries, not independent-scene or independent-object estimates. Evidence: `P\analyze_selected_refinement.py:46`, `:50`, `:51`, `:58`; `P\SELECTED_REFINEMENT_FINDINGS.md:12`.

I also recomputed axis-aligned IoU from saved boxes/GT for **both `bbs` and `bbf`, both stages, selected and coarse**. Every strict `>0.25` and `>0.50` decision agrees with the stored value. The largest numerical discrepancy is **2.935893036542676e-6** (formal `bbf` coarse); formal primary `bbs` discrepancies are **2.8116331026728503e-6** selected and **2.6527254894936902e-6** coarse. Recomputed formal `bbs` mean delta is **+0.0003369610826833694**, preserving the reported sign and rounding. This does not establish bitwise GPU-arithmetic replay.

Original formal `bbs` remains **5,594/4,401** at >0.25/>0.50, matching its receipt and `CPU_RECOUNT.json`. Evidence: `C\p3\formal\receipt.json:18`; `C\CPU_RECOUNT.json:196`; `C\EXPERIMENT_AUDIT.md:80`. Same-query coarse 4,403 is an internal intermediate, not a separately trained baseline.

**4. Actual output and dormant-code boundaries**

The current magnitude result records completion at **2026-10-02T20:47:31.275928+08:00**. Its analyzer hash matches the actual **7,027-byte** source, SHA256 **`dfdc91ce34c11ee9ba9238972d4ed3d58746b51c610534f13bfdb78a90c9bf71`**. Its two row-source hashes match the archive. The top-level source calls every summary and subgroup before writing the result; no new reported statistic is dormant. Evidence: `P\analyze_selected_refinement.py:75`, `:84`, `:91`, `:103`, `:109`; `C\SELECTED_REFINEMENT_MAGNITUDE.json:2`, `:16`.

This is **artifact-backed execution evidence**, independently reproduced here. I did not witness the historical shell invocation or obtain an OS execution log. The source performs standard-library CPU/file work and records zero model forwards/optimizer steps; stored IoUs are historical GPU outputs, not newly evaluated scores.

The historical `verify_native_replacement` diagnostic is called only under `if not update`, while the shown P3 preflight and training calls use `step(..., True)`. It supplies no fresh reconstruction witness in this run and receives no credit here. The Nr/Sr note correctly makes that witness future work. Evidence: `C\run.py:299`, `:349`, `:495`; `S\REFERIT_G_INTERFACE_NOTES.md:19`.

Terminal IDs exactly match the 6,887-row holdout; formal IDs are 0 through 9,507. The 106 terminal and 141 formal physical scenes are disjoint. The split records 29,778 fit rows and explicitly says prior pretraining has seen the development holdout. I did not redo the historical checkpoint/training-budget audit. Evidence: `C\source\split_protocol.json:1`; `C\EXPERIMENT_AUDIT.md:123`, `:137`.

**5. Future Nr3D/Sr3D interface**

The actual frozen PV criterion uses ScanRefer/Nr3D token weights **0.6/0.2/0.2/0.1**, Sr3D **0.625/0.125/0.125/0.125**, and CE/contrastive multipliers **0.5** for ScanRefer and **1** for Nr/Sr inside `1/(num_decoder_layers+1)`. The current G helper hardcodes ScanRefer weights and **0.5/7**. Widening only its benchmark assertion would give the wrong correction. The weights are not automatically normalized to unit mass. Evidence: `C\source\imported\models.losses.py:484`, `:486`, `:944`, `:948`, `:953`; `V\models\pvground_semantic_assignment.py:26`, `:35`, `:47`.

`language_dataset = self.test_dataset` selects the native recipe; `sample_dataset = anno['dataset']` identifies the annotation source. Joint ScanNet detection rows can therefore share a benchmark recipe while retaining detection annotations. The old Nr adapter skips those rows and returns an unscaled CE difference for native aggregation. Its interrupted history is documented, not a completed negative PV/Sr experiment. Evidence: `C\source\imported\src.joint_det_dataset.py:1264`, `:1270`, `:1342`, `:1402`; `V\models\nr3d_semantic_assignment.py:16`, `:21`, `:46`, `:74`; `V\docs\MCLN_CURRENT_COMPLETE_HANDOFF_2026-08-15.md:19937`.

The saved Nr/Sr contracts contain **7,899/17,726** language-row hashes and expected formal rows, **130/255** listed scenes, `butd=false`, `butd_cls=true`, `butd_gt=false`, primary `bbs` and no Mask gate. Both record **zero model forwards, zero executed formal rows and zero optimizer steps**. `butd_cls` substitutes native scene-instance boxes with predicted categories; this object-input protocol must remain visible. Evidence: Nr contract `:4`, `:8050`, `:8056`; Sr contract `:4`, `:18002`, `:18008`; `C\source\imported\src.joint_det_dataset.py:1366`.

All four source hashes, both contract hashes and the note hash in `S\referit_g_interface_receipt.json` match the actual files read. The note correctly describes historical preparation and future adaptation, not new remote validation, preflight, training or accuracy. Remote assets were not revalidated.

**6. Static publication review**

The current `reviewed_sources` loop binds **analyzer, findings, magnitude JSON, publisher, interface note and interface receipt** to this review's SHA256 inventory before writes. Result-to-analyzer and interface-receipt-to-note bindings are checked again. Appendix numbers and substantive interface statements match the current evidence. The reviewer materials in the payload are the explicit request/reports; no raw reviewer conversation, memory file or private trace directory is in the payload set. The follow-up inventory binds private trace paths/hashes as provenance without adding those trace files to the publisher payload. Evidence: `P\publish_selected_refinement.py:15`, `:17`, `:20`, `:23`, `:25`, `:31`, `:45`, `:49`, `:51`, `:58`.

All four current handoff copies are **2,102,247 bytes**, SHA256 **`3f812d411328f8d77e170722b6f1f8a687e8c4cd4307c4d71c858054d58de5c9`**, and match the prior receipt. During the initial review, read-only Git checks found all three repositories clean at the receipt's recorded heads; those Git checks were not repeated in this bounded follow-up. Static appendix evaluation preserves the exact old raw prefix and one new section marker. Section 20.376.17 and its publication receipt are absent; all eight payload destinations are absent in the two target repositories.

The publisher checks the old remote handoff, constructs `new = old + UTF-8 appendix`, writes four local handoffs plus the remote copy, and checks remote payload/document readbacks. Destinations are bounded to the handoff, eight explicit evidence/analyzer files, appended manifest/attributes, the local synchronization guard's expected hash and its own publication receipt. No active model, training source/configuration or checkpoint is a destination; no deletion is present. Evidence: `P\publish_selected_refinement.py:35`, `:37`, `:55`, `:73`, `:77`, `:83`, `:94`, `:100`, `:124`.

Staging rejects unexpected status paths and checks every payload's staged bytes. The only push is **`git push origin HEAD:main`**, followed by remote-ref comparison; there is no force push. The first two committed handoff/payload blob sets are compared. The synchronization guard has exactly one current expected-hash replacement site. Evidence: `P\publish_selected_refinement.py:107`, `:109`, `:114`, `:118`, `:121`, `:126`; `X\sync_cs_handoff_remote_20260923.py:12`.

These are static checks and local preconditions, **not completed remote writes, commits, push or final readbacks**. Observer/next-check fields are inherited metadata, not fresh liveness evidence. Evidence: `P\publish_selected_refinement.py:128`, `:132`; `S\initial_boundary_publication.json:17`. Publication completion and live raw/fused progress require their actual runtime evidence.

**7. Bounded transport follow-up by the same reviewer**

This continuation reuses the successful original numeric and scientific review; **no numerical recount was repeated**. The preserved original JSON and verbatim response match SHA256 `14e79f3f5791014b92da7f2b20d9e3044d1f4f150b64b00c585afeb00cc0dab4` and `5d8a6816413528b85ad33a323c04211345299adadbf37262d87079d9e98ead55`. The archived waiting publisher is byte-identical to the 10,095-byte publisher bound by that original review. All **28 other original input files** still match their reviewed byte counts and hashes. This includes the scientific source, rows, result, findings, interface artifacts, handoffs and synchronization guard.

The actual diff adds exactly three calls: `prefetch(file_size=len(old), max_concurrent_requests=64)` before the old-document equality check, `set_pipelined(True)` on the document write handle, and `prefetch(file_size=len(new), max_concurrent_requests=64)` before final document equality. Removing only those three AST statements makes the current and preserved-old syntax trees identical. Scientific wording, payloads, six-input review binding, prefix construction, destinations, staging, nonforce push and all equality assertions are unchanged. Evidence: `P\publish_selected_refinement.py:74`, `:81`, `:101`; `P\publish_selected_refinement_sftp_waiting.py:73`, `:79`, `:98`.

| Preserved boundary record | CST time | Observed or recorded state |
|---|---|---|
| Before stop | 21:20:19.374380 | Remote handoff 2,102,247 bytes, original `3f812d...de5c9` SHA256; publication directory absent |
| Helper stop | 21:24:28.9442493 | Record names PID 32672, original publisher hash and four unchanged local documents; scope states own publisher only |
| After stop | 21:26:46.771027 | Same remote handoff length and full original SHA256; publication directory still absent |

The before/after receipts bracket the stop record chronologically. The inspector's remote code only reads the handoff bytes, hashes them and checks directory existence; it does not execute a model or mutate remote files. The old publisher's first remote mutation is the publication-directory creation, which precedes local handoff writes. These snapshots therefore support the bounded conclusion that publication had not visibly crossed that mutation boundary. Local handoff hashes also remain unchanged. Evidence: `P\publication_transport_boundary_before_stop.json:2`, `:4`, `:5`; `P\publication_helper_stop.json:2`, `:4`, `:5`, `:7`; `P\publication_transport_boundary.json:2`, `:4`, `:5`; `P\inspect_publication_transport.py:14`, `:19`, `:22`, `:25`; old publisher `:75`.

The stop record's textual reference to `publication_transport_boundary.json` at 21:20 is historical: that snapshot is now preserved as `publication_transport_boundary_before_stop.json`, while the current boundary file is the distinct 21:26 snapshot. I did not run the inspector, inspect or stop a live process, or independently attest PID ownership/termination. The stop receipt is evidence of the executor's recorded action, not a process-attestation artifact. The receipts are saved observations, not a live remote-state guarantee or proof of the precise cause of the earlier wait.

Local Paramiko **5.0.0** package metadata and implementation accept the new APIs. `prefetch` takes `file_size` and `max_concurrent_requests`; the latter is documented as added in 3.3. The supplied lengths control prefetch scheduling, not the equality check's length: `read()` still consumes to EOF, and `_read` performs ordinary requests when prefetched data is exhausted. The file context manager still calls `close`; pipelining documentation warns that write errors can surface at close rather than at `write`. The publisher retains its complete byte readback after closing. Evidence: `A5\paramiko-5.0.0.dist-info\METADATA:3`; `A5\paramiko\sftp_file.py:78`, `:179`, `:418`, `:438`, `:468`; `A5\paramiko\file.py:156`, `:177`; `A5\paramiko\util.py:288`; current publisher `:80`, `:100`, `:102`.

This is API/source compatibility verification, not a network benchmark, successful-transfer witness or certification of the interpreter selected by a future launcher. The supplied boundary receipts do not identify the Paramiko runtime version. No new fallback, retry, exception suppression, deletion, process-control or experiment operation is added by the diff. **Transport follow-up: PASS with those evidence limits; overall WARN and no blocker remain.** The revised publisher still requires actual successful runtime readbacks and a completion receipt before publication is called complete.

**Remaining qualifications and claim ceiling**

- Source-to-saved GT provenance is verified; original raw targets and physical scale were not independently reconstructed.
- Measurements concern saved/clamped `bbs` geometry at one selected query, not unclamped head outputs, all candidates, member-point support or raw Masks.
- Stored continuous IoUs have small arithmetic replay discrepancies; all audited strict decisions agree and quoted continuous means reproduce.
- One archived seeded ScanRefer run, prior-pretraining exposure and different terminal/formal scenes cannot establish significance, a causal generalization mechanism or P3's isolated training contribution.
- Nr/Sr artifacts remain historical zero-forward preparation and future interface guidance.
- The current publication code is reviewed statically. Saved pre/post-stop receipts support an unchanged publication boundary; live remote state, helper process ownership/termination, observer liveness, future launcher dependency selection and publication success are not independently certified.

The current artifacts state these limits adequately for their descriptive claims. **No additional training, model rerun, retuning or unrelated defensive refactor is required by this audit. Blocking findings: none.**

**Actual reviewed-file SHA256 inventory**

The inventory covers 47 actual files: the original 29 inputs with the current publisher, seven new transport/prior-trace artifacts, and eleven local Paramiko source/metadata files inspected for API compatibility. The original numerical review is retained; the follow-up verified unchanged hashes rather than repeating its calculations. Full absolute paths are recorded in `SELECTED_REFINEMENT_AUDIT.json`.

| File | Bytes | SHA256 |
|---|---:|---|
| `P\analyze_selected_refinement.py` | 7,027 | `dfdc91ce34c11ee9ba9238972d4ed3d58746b51c610534f13bfdb78a90c9bf71` |
| `P\publish_selected_refinement.py` | 10,262 | `26b772d6060e6723ef855dbc27bf88afa7a61e82b191baab439b3e93f2fa64f5` |
| `P\SELECTED_REFINEMENT_FINDINGS.md` | 3,375 | `c1e8e39a34836fa3445ed8405d231292e77b475ffba425dbbdc7184662888658` |
| `C\SELECTED_REFINEMENT_MAGNITUDE.json` | 51,567 | `b91303ffb3a5b8bc1b2a80730e93657133b88e5b6b0f9ed3c8b7f1ff0c2709d8` |
| `C\INTAKE.json` | 8,506 | `e07db4863d45da906f6e50960e250bacbb26a4c545ff56601c88b20a2a71bd21` |
| `C\CPU_RECOUNT.json` | 9,891 | `fc68682965073245d5371a1a0de71991bc78d1dc7b6d5da747f0b7647e9e2cd5` |
| `C\EXPERIMENT_AUDIT.md` | 15,050 | `5423833a305cb4e6dc32f0c39d4febcfff1f5bbc682d39415693f31a48ada61e` |
| `C\EXPERIMENT_AUDIT.json` | 30,267 | `402b10352355ea1a7db98a7f82f683b6edccafdea60d601db2235e7054593761` |
| `C\p3\terminal\rows.jsonl` | 8,460,376 | `fd3cbc72293629ac2c9d6a83514ca6fa30c630398bf82e34a7fc6d956aa5fa53` |
| `C\p3\terminal\receipt.json` | 683 | `7fd5f01506e83c4388c4aaf6b6ba97ba64841b10ba2a807d86bcf07b7203dddd` |
| `C\p3\formal\rows.jsonl` | 11,460,942 | `fe38d3383598aeb33e3d1c3ecea7b77e195b9e0db9f1a297babe4f74f1fe6c2f` |
| `C\p3\formal\receipt.json` | 685 | `832ae0181ca1927a7351010f8115e30a66013c33ea194d1eddc0efb15401bf02` |
| `C\run.py` | 34,094 | `c593a94fbccf0e1554499f7fe7e0d5ec41c54b4a2487954e8eda7c6616aa4d06` |
| `C\source\split_protocol.json` | 246,139 | `06d0b20a848be97827a4e0257b074afcfb91a04fc8e448f7de34eef453294461` |
| `C\source\imported\models.losses.py` | 40,468 | `920051cc7d1009493829eebbb6185da834a4efd037e9d73ad43ce918a81031de` |
| `C\source\imported\src.joint_det_dataset.py` | 75,981 | `3afebfc69232f9547e6810563ba6dd4f34a0f770f929df2f4203b21493de0e3d` |
| `V\models\pvground_semantic_assignment.py` | 5,158 | `bec33ac6f24b3b83ab4bdfdeaf19cd7d400a372539a023a9929dadccd3ecad6d` |
| `V\models\nr3d_semantic_assignment.py` | 3,790 | `9ddd4e902e63da28e89d29ab08be4b91b3bb6a9c3068b57a88a5823bdcc44886` |
| `S\REFERIT_G_INTERFACE_NOTES.md` | 2,751 | `65ab43e4c8773ab4f11807f42a6b63dcd6cab53033ef40ebf876d68088490d9c` |
| `S\referit_g_interface_receipt.json` | 2,855 | `0c9ee0d8f84b341e623b258a12277726ff39f12d3377bf2c09fc3ec90bcc5329` |
| `V\refine-logs\pvground_referit_formal_preparation_20260917_v1\nr3d_formal_input_contract.json` | 572,693 | `5339166b38dad06a7e46c73c2e9a27cbbc4e399a90a2622518a21b29e70c3903` |
| `V\refine-logs\pvground_referit_formal_preparation_20260917_v1\sr3d_formal_input_contract.json` | 1,282,738 | `e6ee8a201e997b0cde6bd7e58c66ed376956a1e87b17d23a1b80bd197f890a25` |
| `S\initial_boundary_publication.json` | 693 | `c4d1a955b6cbb1e6503e906b30f8767d99e064ef9b1423d68325d02a54af5248` |
| `M\docs\MCLN_CURRENT_COMPLETE_HANDOFF_2026-08-15.md` | 2,102,247 | `3f812d411328f8d77e170722b6f1f8a687e8c4cd4307c4d71c858054d58de5c9` |
| `V\docs\MCLN_CURRENT_COMPLETE_HANDOFF_2026-08-15.md` | 2,102,247 | `3f812d411328f8d77e170722b6f1f8a687e8c4cd4307c4d71c858054d58de5c9` |
| `O\docs\MCLN_CURRENT_COMPLETE_HANDOFF_2026-08-15.md` | 2,102,247 | `3f812d411328f8d77e170722b6f1f8a687e8c4cd4307c4d71c858054d58de5c9` |
| `D\MCLN_CURRENT_COMPLETE_HANDOFF_2026-08-15.md` | 2,102,247 | `3f812d411328f8d77e170722b6f1f8a687e8c4cd4307c4d71c858054d58de5c9` |
| `X\sync_cs_handoff_remote_20260923.py` | 1,435 | `839df7de6a3df8b7583a451ffbb0a1479ef6d27408c08c384f36f76c454ee21b` |
| `P\SELECTED_REFINEMENT_REVIEW_REQUEST.md` | 4,802 | `3baf17569e8b4a976624ed26137ab1975e66be82ffdbf685fb1aacf70628ea5f` |
| `P\publish_selected_refinement_sftp_waiting.py` | 10,095 | `c0374f2ac27e1a2d1b6ffd6c2e1936d21b03c6b68f79f93f53ea17815380f7c5` |
| `P\inspect_publication_transport.py` | 1,499 | `d0b163a1ee74a9d17daad165b2f92aecb98bd78ddf7d2eaa0ec5ce122d8b02df` |
| `P\publication_transport_boundary_before_stop.json` | 323 | `21f67130fbc7f72fb8d9f8813494e1ba243c33b992eb624952cc0baf359087ef` |
| `P\publication_helper_stop.json` | 400 | `54cbfe77ae53a9876e486681a2881b788aaf78a0259e56922d1a12440e9ab29a` |
| `P\publication_transport_boundary.json` | 323 | `ab2d6e692f530b2a9ef105e3f9df7b6c3a7eee888a1f4edebcfccc9837e4b3d2` |
| `T\001-original-final-audit.json` | 28,944 | `14e79f3f5791014b92da7f2b20d9e3044d1f4f150b64b00c585afeb00cc0dab4` |
| `T\001-magnitude.response.md` | 20,544 | `5d8a6816413528b85ad33a323c04211345299adadbf37262d87079d9e98ead55` |
| `A5\paramiko\__init__.py` | 3,450 | `770f0edde7d9cf058fea079f536e081892ad4533563e46353a2fdedc7584324e` |
| `A5\paramiko\sftp_file.py` | 21,820 | `36055f0e1c71511845ac4aa788825080a43ac258024e0395bb91b041ccd6fe19` |
| `A5\paramiko\file.py` | 19,063 | `3606e1523620acb87e1d0b6c75894f6770b2bd2d23857a9e3e4e391a11cf312a` |
| `A5\paramiko\sftp_client.py` | 35,855 | `7bfce2e95db7ded8f1dc31fd4c7eeb4432913be4c267fcf23a4070e2c491223a` |
| `A5\paramiko\util.py` | 9,556 | `6a940ee306d748cf2ef08119871867af0604e1ac69a0118dbe767120a1a14c89` |
| `A5\paramiko-5.0.0.dist-info\METADATA` | 3,666 | `94061b17f3d5e7fe935cc7d3bd01baed65533e43e2fcf39825e703d3629998f2` |
| `A35\paramiko\_version.py` | 80 | `b4b897a97c02386bdfaa4ae2c0b5918fee77aa00373f9b4de327fb25c4a8e3ff` |
| `A35\paramiko\__init__.py` | 4,446 | `df0807e520eaf3d3066e22087425a0fb0e3c795bbb9603fc6d43df482e09b6e7` |
| `A35\paramiko\sftp_file.py` | 21,820 | `36055f0e1c71511845ac4aa788825080a43ac258024e0395bb91b041ccd6fe19` |
| `AJ\paramiko\__init__.py` | 3,450 | `770f0edde7d9cf058fea079f536e081892ad4533563e46353a2fdedc7584324e` |
| `AJ\paramiko\sftp_file.py` | 21,820 | `36055f0e1c71511845ac4aa788825080a43ac258024e0395bb91b041ccd6fe19` |
