# Candidate-consistency pair launcher review

Verdict: **WARN — no blocking code findings; launch remains held for insufficient actual output-directory capacity.**

Review time: 2026-10-03 06:18 CST. Reviewer attribution: delegated Codex, **same-family / provisional**. Scope is only the newly prepared `launch_pair_authorized.py` and its interaction with the already reviewed pair/environment. The earlier code and runtime verdicts are preserved unchanged. No model, GPU, SSH/network command, or launcher was executed by this review.

The launcher correctly rejects the supplied 06:12 CST snapshot: **437,506,048 free bytes < 1,295,019,283 required bytes**, a shortfall of 857,513,235 bytes. It must not launch now. If adequate actual output-directory space later becomes available by an authorized means, the already authorized bounded experiment can use this launcher without executing any deletion. The pending exact three-file deletion request is separate and is not used by this script.

## Findings

**Blocking code findings: none. Nonblocking code defects: none.** WARN records the current resource hold and the launcher's NOT_RUN status; it is not a new method or loss verdict.

## Execution and receipt boundaries checked

- **Prior evidence gates are present.** Lines 10-17 reject an existing local launch receipt, require accepting code/runtime reviews with empty blocking findings, require a passed batch-8/two-update/zero-new-state preflight, and require `preflight.exit == 0`. The actual available artifacts satisfy these historical gates. They certify the completed disposable sanity, not any full training updates.
- **Remote preflight identity is checked.** Lines 27-28 compare the parsed remote preflight receipt with the local actual receipt, rather than accepting only a generic remote status. The reviewed `pair.py:18-21` additionally requires `g_strict_restore` and `semantic_consistency`. The remote equality check remains unexecuted by this review.
- **Actual save-path capacity is used before launch.** The remote read-only snippet checks the real pair root, rejects prior `pair_status.json`, `pair_launch.json`, or `pair.exit`, obtains `shutil.disk_usage(root).free`, and asserts the measured requirement before reaching `screen` (launcher lines 30-45, 52). With serialization size 342,194,609, the formula `3*S + 256 MiB` is exactly 1,295,019,283 bytes. This does not rely on the misleading free-space values obtainable through parent-checkpoint paths.
- **The controller retains its own capacity gates.** The unchanged `pair.py:28` repeats the pair-level check before spawning children. `pair.py:38-42` requires `2*S + 128 MiB = 818,606,946 bytes` before each training arm. A passing launcher check does not bypass the reviewed controller's checks.
- **GPU ownership is respected.** The resource snippet requires an empty `nvidia-smi --query-compute-apps=pid --format=csv,noheader` result. The actual environment has one A100 and sets `CUDA_VISIBLE_DEVICES=0`; querying compute processes across the host is consistent with that environment. Each stage still acquires the existing nonblocking GPU lock through `flock -n` in `pair.py:44-47`. The launcher does not stop or preempt another process. The supplied snapshot reports 1 MiB GPU use and no compute process, but that does not override the failed disk gate.
- **It launches the existing serial controller.** Lines 46-52 use the existing runtime interpreter, prepend the common experiment root to its existing `PYTHONPATH`, and execute `pair.py --root <same root>` inside detached `screen`. The reviewed pair waits for each child and runs control train, consistent train, control formal9508, and consistent formal9508 in that order. This script does not invoke a new model, alter specs, reuse disposable preflight weights, install packages, or change the runtime. The unchanged training runner reloads original G and creates fresh AdamW as previously reviewed.
- **Exit and observation boundaries are honest.** The shell wrapper redirects controller output to `pair.log`, captures the controller's exit code immediately, writes `pair.exit`, and exits with that code. Lines 53-56 require a successful screen command and then exactly one matching live controller process. Local and exclusive remote launch receipts are written only after that observation (lines 57-65). Such a receipt would prove one live controller at that observation boundary, not successful E0, completed original-G restoration in the new process, any number of full-training updates, or final accuracy. The receipt explicitly sets `new_accuracy_available=False` and `deletion_executed=False`.
- **There is no automatic retry or cleanup.** The launcher contains no deletion, checkpoint movement, environment installation, retry loop, fallback, or restart. The absence or presence of previously discussed old files has no branch in this script. Original G and the still-pending negative-terminal cleanup candidates are not altered by it.

## SSH and shell checks

The new launcher retains the existing Paramiko/screen pattern from `C:/Users/gb/.codex/tmp/launch_pvg_g_p2_pair_20261002.py`. It loads known-host keys and does not install an accept-unknown-host policy. The password is taken from `MCLN_SSH_PASSWORD`, not embedded or printed (lines 20-23).

The remote resource command is assembled with `shlex.join`; environment assignments and interpreter/controller arguments are quoted separately; the whole inner shell program is passed through `shlex.quote` to `bash -c`. Log/exit paths are quoted. The fixed screen name is a simple literal. The anchored `pgrep -af` pattern matches the intended interpreter/controller/root command and excludes the shell/pgrep command lines. No unquoted user-controlled shell fragment, host-verification bypass, or credential disclosure was found in this actual path.

The remote snippet uses syntax and subprocess arguments compatible with the pinned Python 3.7 runtime. `shlex.join` runs in the local launcher environment, as in the existing successful deployment pattern; it is not invoked by the Python 3.7 remote snippet. No compatibility shim is needed.

## Verification performed and limits

I read the entire new launcher, the existing pattern, the unchanged pair controller, the actual preflight/exit evidence, relevant specs, and archived environment. Local parsing passed for the launcher and for its embedded snippet using Python 3.7 grammar. A local `shlex` argument round-trip preserved the intended `screen -> bash -c -> env -> interpreter/pair.py` arguments. This was string analysis, not shell or process execution. Scalar arithmetic confirmed both storage requirements and rejection of the supplied capacity. Existing identities confirmed `run.py` and the loss module still match their reviewed source, and `pair.py` is byte-identical to the reviewed copy; no earlier loss review was repeated.

**NOT_RUN:** the new launcher, fresh remote receipt equality, fresh disk/GPU/process queries, `screen`, `pgrep`, any live-controller observation, new launch-receipt writing, full-pair training/evaluation, or a new accuracy result. There is no local `pair_launch.json` at this review boundary. The 06:12 CST resource state is supplied read-only evidence, not a fresh measurement by this reviewer.

All prior review artifacts remain unchanged. No implementation file was edited and no deletion was performed. The code-review gate for this launcher is satisfied, but the current capacity gate is not: do not execute a launch now or describe the pair as live.
