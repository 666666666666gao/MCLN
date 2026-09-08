# Source Query control code review

Fixed base: 2a64610956797f2bb6aef3acfaca7647df7e8a54. Two fresh-context Codex reviewers used by code-review skill; same-family, provisional. Model/effort inherited by these reviewers, not independently overridden or advertised as cross-family acceptance. Read-only source reviews; no SSH/GPU by reviewers.

## Standards

No actionable documented-standard violation in the staged module, isolated source port and interface preparation. The pinned string-substitution staging is maintenance-sensitive but justified by exact replacement counts and minimal task scope; no generic refactor requested. The initial review noted eval-only replay did not establish training RNG equality.

## Spec

Initial finding: paired replay was eval-only and did not compare ending RNG. Source architecture otherwise matched the spec. Added explicit buffer/all-RNG restoration and training replay. Follow-up review confirmed the restoration logic; subsequent runtime evidence showed full-train source features already have tiny repeat differences while reader is disabled. v3 therefore makes a narrower claim: on identical captured real last-layer inputs in train mode, zero-residual output and ending RNG are exact, while full training repeats are recorded separately. No full-model or Mask bitwise claim.

Production bridge review found no blocker: strict native parent load before reader install; expanded initial snapshot contains all added state; capacity backward does not update, then strict complete restore/RNG reset; evaluator leaves initial state equal; all new trainable state goes into deltas with module/spec/port metadata; same fixed data, seed and budget. This is source review, not a claim that training/capacity/checkpoint restore has already completed.

Actual v3 check: 4 full eval forwards + 4 full train forwards + 2 extra last-layer forwards; 2 ephemeral optimizer updates, no saved checkpoint. 807 trainable tensors/28674139 parameters, 714528 new; frozen text unchanged. Six source projections and attention receive nonzero gradients on step2. Terminal quality unmeasured.
