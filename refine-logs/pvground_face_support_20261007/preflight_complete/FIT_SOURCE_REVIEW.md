# PV-Ground paired formal lifecycle SOURCE review

Verdict: **PASS**. Blocking findings: **0**. Nonblocking findings: **0**.

Execution scope: **SOURCE_ONLY**. Independent context, **same-family / provisional**. Requested gpt-6-astra/max; actual backend and effort **UNATTESTED**.

No remote query, neural execution, optimizer update, checkpoint operation, array deletion, or executor-source edit was performed. The running M0 was left untouched.

## Findings

No concrete defect was found in the formal launcher, observer, collector or preparer. All49 files from the prior paired-runner review remain byte-identical.

## Checks

- **prior source identity — STATIC_CHECK_PASS**: All49 files in the existing paired SOURCE manifest still match their actual SHA-256. The parent/controller/spec/paired loop, both factories, sampler, native dependency graph, dataset/evaluator and M0 deploy/observer therefore remain the reviewed versions. This FIT manifest also binds the new launcher, observer, collector and preparer themselves.
- **actual M0 gate before fit — SOURCE_PASS_RUNTIME_PENDING**: launch_fit_authorized.py:18-40 requires local preflight_wait observer_closed, no live controller, exit0 and complete controller status; it requires the collected real preflight.json to say pass, two updates per arm, two frozen-parent forwards, no weights, shared native score/Mask/reference and separate gradients. Each arm must have exact full1304-state CPU reconstruction, declared sampler, exact Adam keys/moments/steps/groups, one deployed head, neutral initial reference and positive second-step parameter gradients. Both witnesses must record center selection, independent rectangle checks, all256 member reference, zero-R identity, one semantic call and no cross-head gradients. Native producer assertions check the actual numerical tolerances and Adam step2. No runtime M0 artifact was read or accepted by this review.
- **M0 provenance and existing environment — SOURCE_PASS**: The launcher checks current spec digest against M0 proof, byte-compares all9 actual remote runner/spec/controller sources to local reviewed versions, and byte-compares the remote proof to the collected proof. The remote probe requires preflight controller exit0/complete/pass before a new fit status exists. Existing runtime Python and GPU flock are reused. The unchanged runner rechecks env/source/official/G/protected-parent identities before Torch/model execution; there is no environment rebuild.
- **disk capacity and deletion authority — SOURCE_PASS**: launch_fit_authorized.py:44 retains reserve=3*max(actual M0 serialized arm payload bytes)+900*1024**2. The remote probe asserts actual free bytes >= that reserve before launching. No lifecycle source deletes arrays or weights, changes reserve, treats older-root authorization as new-root permission, or relies on the proposed cleanup having occurred. Supplied491466752B free is below even the fixed943718400B part; this source PASS is not a capacity PASS.
- **fresh protected start and lifecycle order — SOURCE_PASS**: controller.py fit runs initial_formal,train,formal as separate existing-runtime subprocesses and stops on any nonzero child exit. Initial/formal entry construction loads the protected step0 selected reference with empty optimizer, official+G parent and current declared sampler. Train creates fresh independent AdamW heads and never loads M0 state; formal evaluation alone restores fit terminal deltas and3723-step Adam. Parent weights are checked before and after the controller; fit budgets and history remain29778 rows once,3723 updates per arm, B8/lastB2, seed2027 and11169+3723 hidden updates.
- **single scheduled observer and bounded window — SOURCE_PASS**: observe_fit_authorized.py rejects existing wait/start files, writes one owner record, waits12600s from launch, then queries only the owned controller pattern/status/exit at240s intervals. estimate13500s yields61 planned observations, with the last planned query at27000s=2xestimate (plus command overhead). It raises for manual diagnosis after the bounded window, with no restart, kill or additional training path. It invokes the collector only after no matching controller, exit0 and status complete.
- **closed-only streamed collection — SOURCE_PASS**: collect_closed_fit_authorized.py first requires closed local fit_wait with exit0/complete, then the remote code independently requires actual fit_controller.exit0 and fit_status complete at the declared root. It streams a remote manifest followed by tar without making an archive on the constrained remote disk. Every actual regular artifact except .pth/.pt/.tmp is included; local extraction checks member identity/type, root containment, size and per-file streaming SHA, then exact expected/seen file sets and remote command exit0. It performs no neural replay, checkpoint copy, checkpoint promotion or deletion.
- **formal and holdout evidence scope — SOURCE_PASS**: The unchanged paired loop records one shared-parent all256 NPZ per batch for9508 initial_formal and9508 final formal, containing shared GT/score/reference/prior/validity and both final geometry arms. The6887 holdout remains selected-row JSON plus all256 coverage summaries. Collector includes these actual non-weight artifacts recursively. Success of collection is not an accuracy, contribution or final-model-selection verdict.
- **preparer and Python APIs — STATIC_CHECK_PASS**: Prepared observer/collector text exactly matches the closed predecessor with the declared spec/root/log-label substitutions. Four new Windows lifecycle files, stable controller and loop, and both embedded remote snippets parse with Python3.7 grammar using existing E:\python.exe3.12.6 and stdlib only. shlex.join is executed on the Windows launcher/collector side; embedded remote code uses Python3.7-supported syntax/APIs and standard tar. No executor source was executed or edited.

## Required runtime gates and limits

- The supplied task reports M0 launched but no result and no formal fit. This review neither queried the running preflight nor inspected runtime M0 results.
- Actual closed M0 PASS, numerical/restore witnesses and the unlowered disk reserve must all pass before the launcher can start formal fit.
- The supplied free-space snapshot491466752B is below the fixed reserve alone. No cleanup was performed; deletion still requires the exact applicable user authorization.
- Observer estimate is a schedule, not a measured completion promise. The source has no automated restart after timeout or failure.
- Artifact intake preserves closed non-weight evidence; subsequent independent result audit and same-model5620/4764 plus three-effective-contribution judgment remain separate work.
- Same-family independent context remains provisional. Actual model/backend and effort identity are unattested; no cross-family or runtime acceptance is claimed.

## Exact versions

`FIT_SOURCE_REVIEW.json` binds 57 actual files, including the four current formal lifecycle files, stable prior49 dependencies, both prior review records, and the two predecessor templates. No runtime preflight result or synthetic source is represented as reviewed.

The preserved capacity formula is **3 × maximum actual M0 serialized payload + 900 MiB**. Formal fitting must start fresh from the protected5598/4848 reference, not the M0 updates.
