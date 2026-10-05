# Native target-box jitter source review

**Verdict: PASS. Execution scope: SOURCE_ONLY.** No blocking or non-blocking source defects were found. The actual augmented64 replay and clean-label results remain pending.

The capture is correct. `run_label_probe.py:68-74` obtains the root annotation bbox from the already augmented native scan immediately before calling the original `_get_target_boxes` once. `get_object_bbox` is a pure annotation-membership lookup plus bounding-box arithmetic. Float64 low/high-to-center/size conversion reproduces the native float64 buffer; the final float32 representation matches native returned labels.

The extra lookup consumes no RNG. The existing native point, target-box and detector augmentation calls are retained. The prior fit subset, seed2027, batch8, two workers, shuffle and explicit loader generator are reproduced. Exact row order, scan IDs, point-byte hashes, noisy labels and valid slots are mandatory gates; prior no-augmentation fixtures are also rechecked. No relaxed replay is accepted.

The analyzer correctly recounts all16384 cached candidates under both box definitions while holding saved matching and mask support fixed. It uses the native coarse-size floor, boundary scale4 and node endpoints[-4,4]. Geometry changes are reported on the fixed1090 noisy-qualified candidates, with clean-qualified groups named separately. It introduces no training/evaluation leakage and claims neither formal accuracy nor cause.

The launcher uses environment credentials, known-host verification, quoted commands and a fixed new directory. No concrete credential or network-boundary defect was found. The collector checks immediately on invocation; the parent explicitly confirmed invocation at least180 seconds after actual launch, followed by240-second checks. The credential wrapper was not read.

Local read-only checks passed:

- Native dataset snapshot SHA equals the actual prior imported dataset SHA; boundary helper SHA equals the prior spec. Prior runner/spec/rows match the closed receipt.
- All64 unique rows, eight batches and16384 candidates align. Saved mask support, matching, selected queries and per-row counts are consistent.
- Supported unmatched candidates:3479; native qualified:1090 on52 rows; qualified outside original nodes:201 candidates /482 faces.
- Every arm has zero CPU-vs-saved-GPU IoU>0.5 classification changes. Recomputed native boundary targets have zero outside-range classification changes.
- Isolated AST-extracted actual methods on64 synthetic annotated-object fixtures preserve NumPy RNG state and every native target-box return; capture equals native unjittered float32 labels. This is not the actual augmented data replay.
- Reviewed Python files parse. The reviewer used cached offline NumPy2.5.3, executed no experiment entrypoint, and made no network/model/GPU/checkpoint/install call.

Pre-jitter removes only the additional native six-coordinate target-box multiplier. It retains point augmentation, native scan representation and annotation membership. The actual64-row replay, clean/noisy differences and closed-result acceptance still require execution. The newly added `publish_closed_labels.py` is outside this review.

Requested reviewer: gpt-6-astra / max, fresh delegated context. Backend identity is unattested. Attribution is **same-family / provisional**; this is not independent or external acceptance.

Reviewed paths and SHA256:

| Path | SHA256 |
|---|---|
| `C:\Users\gb\.codex\tmp\pvground_target_jitter_20261005\EXPERIMENT_PLAN.md` | `2c98b73233e6225887b84f12055d484a87b5b0f47b72507d8066d948b4ca41e4` |
| `C:\Users\gb\.codex\tmp\pvground_target_jitter_20261005\run_label_probe.py` | `3ba2d9ec8327a520238cbd4f14f43144f30e1847eecc9c88b25d4e1082347a3d` |
| `C:\Users\gb\.codex\tmp\pvground_target_jitter_20261005\analyze_labels.py` | `bd6390d47e6e5a633760f513c45b6c3dfcdebd39d24e34b6e720fcd8a1897d21` |
| `C:\Users\gb\.codex\tmp\pvground_target_jitter_20261005\launch_cpu_authorized.py` | `846a1c8c26ae83d4ad21d933a516fc26056524b8c91c4b2ae32bcaac95493a35` |
| `C:\Users\gb\.codex\tmp\pvground_target_jitter_20261005\observe_collect_authorized.py` | `e155ebefdc60a151db43b750d3bf773351440a5f39b24d7b234fc2670251fec7` |
| `C:\Users\gb\.codex\tmp\pvground_target_jitter_20261005\spec.json` | `e86b2597e4510f1e980074a24ebddba36af9a67594aeff230c6f6d0ad3f56383` |
| `C:\Users\gb\.codex\tmp\pvground_query_geometry_cohort_20261005\run_cohort_probe.py` | `ff006960b9a1c0754ba6dc3ef163fc997c3594e2232e98c9bc25f0639766dd18` |
| `C:\Users\gb\.codex\tmp\pvground_query_geometry_cohort_20261005\spec.json` | `748809b0e1c740b3253d992de719b8e8e1bac55518e7e1e0a5a6651932f1de71` |
| `C:\Users\gb\.codex\tmp\pvground_query_geometry_cohort_20261005\cohort_metrics.py` | `608a717c2257844172705ba774f9e61bf17d1faa9748b8a773b5cd6ba750a90a` |
| `C:\Users\gb\.codex\tmp\pvground_query_geometry_cohort_20261005\complete\rows.jsonl` | `0e6236609932eab258ab52b9be725c5161fb1f55378fc565504dcb1386179f3d` |
| `C:\Users\gb\.codex\tmp\pvground_query_geometry_cohort_20261005\complete\imports.json` | `51fd47332fe71a1ce94c22eedd9256c095346f9808304630395e0f2c24c98a23` |
| `C:\Users\gb\.codex\tmp\pvground_query_geometry_cohort_20261005\complete\receipt.json` | `c33f90ee770fe6a27f70e1372eee62dba169d979f43957d650767866b2906460` |
| `C:\Users\gb\.codex\tmp\pvground_query_geometry_cohort_20261005\complete\batch_00.npz` | `b0fb8880b214544bfbab2c9c525dfce8e11df63b264baa5ba9ec8ee2aabc1f86` |
| `C:\Users\gb\.codex\tmp\pvground_query_geometry_cohort_20261005\complete\batch_01.npz` | `b4b7698f591a6ac995a030b4c2170ea73e5f4a189e9ded2d70e04c608a9fbd9f` |
| `C:\Users\gb\.codex\tmp\pvground_query_geometry_cohort_20261005\complete\batch_02.npz` | `99f8a09ab437e898fc68854017d8184fad781ce52420e1ab3a3225abdff60154` |
| `C:\Users\gb\.codex\tmp\pvground_query_geometry_cohort_20261005\complete\batch_03.npz` | `1fd009cdc0902ef768ebace44a24110fb7cf54a683fb82e91ffa9099b2243caf` |
| `C:\Users\gb\.codex\tmp\pvground_query_geometry_cohort_20261005\complete\batch_04.npz` | `bb42cbd58c8034aa4472a58ea96cf3e3af60fc8408ab21a0e34a7559fd26dfbd` |
| `C:\Users\gb\.codex\tmp\pvground_query_geometry_cohort_20261005\complete\batch_05.npz` | `c9f64b5913d752f5b7d3567e64634487566945392ef33de337fe0255bcc0452e` |
| `C:\Users\gb\.codex\tmp\pvground_query_geometry_cohort_20261005\complete\batch_06.npz` | `1aaef5edc68d4eb6c75d984bc1f65e7e1f3e5dbd25fc16fdf754678007d51cf8` |
| `C:\Users\gb\.codex\tmp\pvground_query_geometry_cohort_20261005\complete\batch_07.npz` | `4cafe6f27eed2a32a277db377acacfd1108e3a8b6c7d60642e506be4a7ace691` |
| `C:\Users\gb\.codex\tmp\pvground_g_p2_20261002\complete\source\joint_det_dataset.py` | `3afebfc69232f9547e6810563ba6dd4f34a0f770f929df2f4203b21493de0e3d` |
| `C:\Users\gb\.codex\tmp\pvground_runtime_bundle_20260908_v1\PV-Ground\src\visual_data_handlers.py` | `8858aa1d7019aa5f06065bd13fcc35fe130e347787c088269c3535939806060a` |
| `C:\Users\gb\.codex\tmp\pvground_final_quality_20261005\runtime_bundle\pvground_boundary_box_refiner.py` | `88debfef64f4295f87d697b7b9d4cd1eab17b39f4041f6d9ae773fc65dda9461` |

