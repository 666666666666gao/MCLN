# Publication v2 source review

Verdict: PASS. Blocking findings: 0. Scope: SOURCE_ONLY.

This is a same-context follow-up, not a fresh or independent-family review. Same-family/provisional; actual backend identity is NOT_ATTESTED.

The v2 source equals exactly the four substitutions declared by prepare_publication_v2.py: independent v2 review gate, three added evidence/report names, explicit stdin flush before shutdown_write, and exit/stderr handling before JSON parsing. The embedded remote receiver and all old Doc79/SHA/HEAD/clean, payload exclusion, byte preservation and active-source isolation checks are unchanged.

The saved inspection records publication session 87407 exit 1 and a successful read-only inspection: original Doc79 SHA, absent evidence directory, no evidence paths. Current local read-only Git checks found all three old heads, clean worktrees and absent new publication prefixes. All 43 original immutable-input SHA checks pass; original publisher and review reports remain unchanged.

The inspection does not contain the failed receiver stderr. Its native session/exit fields were supplied to that inspection script. Therefore the failure root cause remains unproven; this review does not call the flush a demonstrated root-cause fix or claim retry success. It does confirm that a nonzero receiver exit will now expose stderr before json.loads.

The resolved payload contains 47 files, including the two new review outputs and generated .gitattributes. root*.py additionally brings in the v2 preparation/publisher and read-only failure-inspection source; all were read and hashed. No NPZ, weights, .aris or complete/formal subtree is included. The unchanged Doc80 describes M0 and the recorded formal launch, not completed formal results or an accuracy gain. Prepared CPU analysis remains source only.

Validation used local stdlib AST/compile and SHA checks plus read-only git rev-parse/status. No publisher, preparer or inspection entrypoint was run; no network, remote command, GPU, training, checkpoint, git write or active-source mutation occurred. Python3.7 grammar acceptance is not a Python3.7 runtime test. No current formal state was polled.

v2 publisher SHA256: 69fbf996249cb075e3661eba0e77d6fc32aca69008bcbe9ad29617cb349cb9e2.

The JSON report binds 49 actual immutable inputs as path/sha256 entries. The two current report outputs are excluded from their own manifest to avoid recursive hashes.
