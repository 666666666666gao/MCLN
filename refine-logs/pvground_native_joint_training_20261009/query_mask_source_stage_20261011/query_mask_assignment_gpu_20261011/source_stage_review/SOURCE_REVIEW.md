# Source-only staging review

Verdict: **WARN — source correctness PASS; no blocking findings and no required changes.**

Reviewed at: 2026-10-11T05:25:19.885369+08:00. Fresh reviewer; requested `gpt-6-astra` / `max`; actual identity **UNATTESTED**. Review independence is `same-family` and acceptance is `provisional`.

The reviewed `stage_native_source_authorized_20261011.py` correctly defines the authorized isolated source-copy step. All 24 supplied pins match, including the stager itself. The existing 14-file matcher variant remains byte-identical to its prior reviewed hashes. This is source readiness evidence only; this reviewer has not executed remote staging.

## Source checks

- The pinned parent manifest contains **116 files / 24,660,224 bytes**, with all parent paths under the original `/root/autodl-tmp/pvground_native_joint_training_20261010/PV-Ground`. The remote body verifies every original byte count and SHA256 before writing and rehashes final files. The 14-file overlay changes exactly `main_utils.py` and `models/losses.py`. Expected source size after overlay is **24,660,643 bytes**; this is calculated locally, not a remote measurement. Evidence: stager lines 18, 47, 58 and 72.
- The destination is fixed to a **previously nonexistent** `/root/autodl-tmp/pvground_query_mask_assignment_20261011/PV-Ground`. Source and asset destinations are checked for containment. Existing source and environment metadata are read only. There is no deletion, process control, active-training mutation, package install, controller launch or automatic retry. Failure leaves the partial directory for inspection. Evidence: stager lines 39–65.
- Exactly the four prepared assets are copied: preflight, init, text protocol and query protocol. Both admission fields are JSON `false` in both protocols; only `matcher_mask_source` differs. Their source maps equal the reviewed overlay. The preflight rejects unadmitted execution at line 26, before output-directory creation or NumPy/Torch imports. The stager never invokes the preflight. Evidence: stager lines 28, 68; preflight line 26.
- The staging logic imports only stdlib modules. The embedded remote program only reads, verifies, copies and `ast.parse`s the supplied files, then writes a manifest. The PointNet compiled extension is copied as bytes; no build is invoked. All 16 pinned local Python files parse. The overlay, preflight and embedded remote body pass Python 3.7 grammar parsing with the local stdlib parser. No project code was imported or executed. Evidence: stager lines 2, 37, 60 and 66.
- Transport uses the existing OpenSSH/AskPass paths and exact `root@region-9.autodl.pro:33476` endpoint with strict host checking and `ProxyCommand=none`. Public known-host metadata matches the witness's `ssh-ed25519` algorithm. Shell command arguments are quoted and payload data goes through stdin. The private authorization wrapper and AskPass contents were not read. Evidence: stager lines 87–98.
- The proposed receipt explicitly keeps shared-runtime dependency true, import verification false, both admissions false and model/data/GPU/training claims unexecuted. Exact return code, stdout and private stderr are saved before failure propagation. No inherited import/constructor-success claim is copied into the receipt. Evidence: stager lines 74–106.

## Limits retained by WARN

The 116-file manifest is a bound partial deployment with shared runtime/data dependencies. It does **not** separately contain `utils/scatter.py` and is not an independently self-contained repository. Remote original bytes and actual destination contents have not been verified in this review; the script defines those checks for the later authorized copy. Python grammar parsing is not an actual Python 3.7 import, compiled-extension compatibility check, constructor check, GPU preflight, training result or accuracy measurement.

The current single-A100 C-off training was neither queried nor modified. The controlling next observation remains **2026-10-11 09:07:13.156040 CST**, supplied by the current task/plan; the older timestamp in historical readiness metadata is not used. Its terminal result and decision still precede any new GPU admission. Both `serial_gpu_preflight_admitted` and `full_training_admitted` remain false. No new model effectiveness, three-contribution, Nr3D/Sr3D or full-goal success is established.

The default PATH `python` failed at startup with `No pyvenv.cfg file`. Static checks succeeded using `C:/Users/gb/AppData/Roaming/uv/python/cpython-3.11.14-windows-x86_64-none/python.exe -S -B`. The later authorized invocation should use the working explicit local interpreter. This is an operational note, not a required source patch.

No actionable source defect was found. The source gate has no blocker for the already-authorized isolated copy; GPU and training admission remain closed. The JSON companion records all 24 verified input hashes and the review's execution limits.
