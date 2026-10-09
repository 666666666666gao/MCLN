SOURCE_ONLY engineering source review — native_joint_v2 R3

Verdict: WARN; source revision PASS, blocking_findings=[].

This limited continuation checks the post-R2 transport correction. Among the 53 R2 input files only launch_engineering_authorized.py changed. Its sole textual change is an explicit local SSH '-o','ProxyCommand=none'; the file also mechanically changed LF line endings to CRLF. Host-key checking, destination, credentials mechanism, stdin payload, source/CPU/parent SHA gates, fresh-directory assertion, GPU/storage checks and remote flock/Popen command are unchanged. Remote Python 3.7 syntax still passes. No fallback, retry loop or cleanup was added.

The normal launcher has the same local argv addition plus line-ending normalization; reversing this addition in normalized UTF-8 bytes produces its previously observed bb4bde929da656fb5547e4e224593910ab0c3d42a4b2960fd1d09910719276bf hash. Normal/observer file hashes from DIRECT_SSH_SCOPE_CORRECTION.json were checked, but full normal/queue admission is a separate review.

The preserved original SSH attempt has exit 255, zero stdout, and Connection closed by UNKNOWN port 65535. The supplied correction record identifies the inherited project-unwanted ProxyCommand and records controller_launch_verified=false. Private SSH configuration, proxy code, AUTH, askpass and known-host files were not read. This source review does not establish whether a remote controller ran and does not label the event a model failure. The next authorized execution retains the assertion that the remote engineering directory must not already exist.

All R2 source conclusions carry forward: full whole restoration is strict and has zero new updates, fresh extremal performs two ordinary-trainer updates with both span groups' gradient/update checks, and no normal training or cleanup stage is added. Actual GPU engineering remains pending/unverified.

R2 report/markdown/original seal are archived verbatim as SOURCE_ENGINEERING_REVIEW_R2.* with an archive mapping; R1 and all raw inputs/receipts remain unchanged. One read-only verification initially asserted byte equality without considering line-ending normalization; its AssertionError and the corrected exact textual/line-ending evidence are retained. Another earlier current-input check detected this authorized edit; it is not a runtime failure.

Requested gpt-6-astra/max; actual route/model/effort UNATTESTED, same-family/provisional. Original task fresh=true; this continuation is not a new fresh audit. No SSH, NN/model construction, GPU, pickle/checkpoint load, deployment or launcher execution by the reviewer.
