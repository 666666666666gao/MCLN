**PV-Ground paired-tail review, round 2 — PASS**

PGTS-001 is resolved by the bounded fix. **No remaining code blocker** was found in this follow-up. The code is ready for its real preflight after the existing current-P3 completion gate; full training still requires actual passed runtime receipts. No GPU/model execution, deployment, accuracy, or runtime success is claimed here.

Reviewed at 2026-10-02T16:45:27.758509+08:00. Actual reviewer: `gpt-6-astra`, reasoning `max`; `review_independence: same-family`; `acceptance_status: provisional`. This review covers only the five changed files and their exact frozen/native dependencies. Round-1 algorithm conclusions carry forward because the other 12 reviewed inputs are byte-identical, including the running P3 source.

**PGTS-001 resolution**

`pvground_tail_preflight.py:13` checks all eight native Mask losses separately against the full refiner parameter tuple. The terms are the actual native Text focal/Dice, Query focal/Dice, corresponding focal/Dice and alpha-fused focal/Dice tensors exposed by the supplied criterion. The final Mask branch populates all eight; zero values from earlier non-Mask prefixes do not turn the accumulated result into a Python constant. `allow_unused=True` admits the expected absent paths; any returned gradient must be elementwise exactly zero. Separate assertions cannot hide cancellation between losses or parameters. `retain_graph=True` preserves the subsequent ordinary backward, and these calls do not accumulate into parameter `.grad`.

The observed order in helper `:55` matches the exact native model: proposal; six prediction heads; `x_query`; then, per sample, an initial Text Mask, three SWA/FFN/Text Mask sequences, final Text Mask and Query Mask; finally the enabled refiner. For batch 8 this is 105 events, or 104 for the deliberate bypass. Text Mask count is 40, Query Mask count is 8, each SWA and FFN count is 8, and the enabled tail appears exactly once after all Mask events. The entire list is compared, so reordering and missing/extra observed calls fail.

Pre-hooks only append Python labels and return `None`. The two native Mask computations are Python methods; their temporary intercepts call the saved bound methods with the same positional/keyword arguments and return the exact original values. Originals and hooks are restored after each completed disposable forward. The observer adds no tensor operation, parameter/buffer mutation, random draw or extra native computation. A forward or witness assertion failure prevents a passed receipt; no extra exception wrapper is needed for this disposable preflight.

Runner `:351` and `:356` observe enabled/bypass zero-initialization forwards. Both real update calls in `:390` also go through the observer and the per-loss checks (`:285`, `:313`). The receipt retains each step's eight values and all four complete call witnesses, and `:424` asserts their count before writing. Normal training retains its original direct `model(inputs)` call; formal evaluation remains unchanged.

Controller `:47` requires the two new passed flags and exactly four witnesses in addition to the existing proof fields before spawning training. Both normal and `--train-after-preflight` paths reach this block. An older proof without these fields fails; the current-P3, paired-raw and disk gates remain intact. The pending deployment must include the new helper and its existing-schema `tail_preflight_module_sha256` field.

**Executed checks**

- All five affected files pass AST parsing and `compile()`.
- All 18 builder replacements replay exactly to the current runner. The patch script's literal changes replay exactly from the frozen round-1 builder/controller to the current versions. These replays ran in memory and did not edit implementation files.
- All 12 unaffected round-1 inputs retain their recorded byte hashes.
- Python-only stand-ins checked argument/result identity, repeated restoration and hook removal, the 105/104 sequences, rejection of an extra Text Mask call, all eight gradient-check invocations, separate rejection of every nonzero loss gradient, and rejection of four old/incomplete proof cases. They validate Python mechanics, not real autograd or CUDA results.

**Remaining nonblocking limitation**

PGTS-W001 remains as documented: `verify_native_replacement` is under `if not update`, while both `step` callsites pass `True`. This inherited path does not provide a fresh G reconstruction/gradient witness. The bounded patch does not claim otherwise and does not alter G supervision.

Actual deployment imports, native Tensor/autograd execution, strict runtime restore, CUDA allocation/serialization size and completed paired metrics remain pending. This is a code-review PASS, not a runtime-preflight PASS. No implementation was edited by this reviewer; only the two round-2 review artifacts were written.

**Reviewed byte snapshots**

The following use the existing trace convention. The JSON additionally retains all unchanged round-1 input snapshots.

- `C:\Users\gb\.codex\tmp\pvground_fused_support_20261002\pvground_tail_preflight.py` (2898 bytes): `823f27d14e333a2fb8f15e6bd41a81abe3717c34a4adcb77aebbbb050ebcdbdb`
- `C:\Users\gb\.codex\tmp\pvground_fused_support_20261002\build_runner.py` (8176 bytes): `de615001e4d9fae9d050248b8a821912d8c206749ecece68e6571cc8f13e74d1`
- `C:\Users\gb\.codex\tmp\pvground_fused_support_20261002\run_pvground_tail_support.py` (38567 bytes): `483a7db4e0d11d8b79af63ea6e02f6d9455bed218190efc17d42e48929648346`
- `C:\Users\gb\.codex\tmp\pvground_fused_support_20261002\run_pvground_tail_support_control.py` (4120 bytes): `21774197eb1ee5a9c4e7742953c1d496e75d6192c80c82de24eb6084398bdcb6`
- `C:\Users\gb\.codex\tmp\pvground_fused_support_20261002\patch_runtime_gates.py` (3166 bytes): `167cf92c073d16ab5a1131f0c178dc0f66296b533524c4a090ac64a771562237`
- `C:\Users\gb\.codex\tmp\pvground_fused_support_20261002\review_round1\CODE_REVIEW.json` (15010 bytes): `3d1c77ae51ec143a55fabc412271b614b209b0aa5ba09148997b3bc8bab2c10e`
- `C:\Users\gb\.codex\tmp\pvground_fused_support_20261002\review_round1\run_pvground_tail_support.py` (37656 bytes): `6ac3a2b2eeaf2d6cbb17a8f1a209650fda31b7e7b7c721431c904d71b52158c2`
- `C:\Users\gb\.codex\tmp\pvground_fused_support_20261002\review_round1\run_pvground_tail_support_control.py` (3857 bytes): `fc474f713eb6fb07dfc65b266deaf0d1d364e3306999e4a27d63841c8ac18fbe`
- `C:\Users\gb\.codex\tmp\pvground_fused_support_20261002\review_round1\build_runner.py` (6126 bytes): `a84b0ddd966a7023a94d13fa5d85c6df7048f53bffe50a4d110421398880fce1`
- `C:\Users\gb\.codex_pvground_cs_20261002\scripts\run_pvground_p3.py` (34094 bytes): `c593a94fbccf0e1554499f7fe7e0d5ec41c54b4a2487954e8eda7c6616aa4d06`
- `C:\Users\gb\.codex\tmp\pvground_p3_next_20261002\parent_source\models\pv_ground.py` (24844 bytes): `20c353512939ac9087cd8f3d07b82a648bcefe42510563e26afc37235facdb7c`
- `C:\Users\gb\.codex\tmp\pvground_g_p2_20261002\complete\source\ported\models\losses.py` (40468 bytes): `920051cc7d1009493829eebbb6185da834a4efd037e9d73ad43ce918a81031de`
