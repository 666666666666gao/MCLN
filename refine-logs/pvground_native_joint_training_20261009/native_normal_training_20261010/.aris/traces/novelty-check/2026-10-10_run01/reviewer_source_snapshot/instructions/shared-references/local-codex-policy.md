# Local Codex Model Policy

The user selected Codex-only execution on 2026-09-06. This policy overrides
inherited backend, model, effort, and cross-family routing examples in this
installed ARIS package. Preserve each workflow's substantive research steps.

- Writing and research reasoning run in the current Codex agent. Every newly
  created writer or reviewer agent explicitly uses `model: gpt-6-astra` and
  `reasoning_effort: max`, preserving the user's existing reasoning preference.
- A reviewer starts fresh with `fork_turns: none` and reads the supplied source
  artifacts directly. Pass the model and effort as separate parameters;
  `gpt-6-astra-max` is not a model identifier.
- Use the available native Codex agent tools. Do not create a separate
  user-visible task just to perform a review. Existing user authorization and
  the host's delegation rules still apply.
- Do not call Claude, Gemini, MiniMax, DeepSeek, Oracle, `llm-chat`, or any other
  external model backend for writing, research reasoning, or review. Do not
  install or activate review overlays. There is no automatic model fallback.
- If the requested Codex reviewer cannot start, record `REVIEW_UNAVAILABLE`
  with the actual error. Do not silently switch models, lower effort, fabricate
  a reviewer response, or turn the executor's own judgment into an independent
  reviewer result. Continue only work that does not depend on that result.
- Fresh Codex reviewer output is independent context but the same model family:
  record `review_independence: same-family` and `acceptance_status: provisional`.
  A substantive PASS/WARN/FAIL may guide revisions and workflow progress; do
  not claim cross-family acceptance or start another model to obtain it.
- Deterministic tools, paper databases, web search, compilation, and plotting
  remain available. Record the actual model/effort on new review records and
  preserve historical records with their original attribution.

## Current Codex host

Use actual callable tool schemas. Older `send_input` examples mean the host's
agent-continuation operation: in the desktop collaboration API use
`followup_task(target=<returned agent id>, message=...)` to resume an idle agent,
and `send_message` for a running agent. Never invent a missing tool.

Inherited references to Claude as "this agent" mean the current executor role;
they do not establish a separate reviewer. Image generation remains an image
tool capability, not a reason to launch an older text model. Do not replace a
genuinely required cross-family verdict with a same-family PASS; report the
unfulfilled requirement and continue independent work.

Keep completed work moving toward the user's requested deliverable. Read only
the references needed for the active phase, reuse existing authorization, and
verify the relevant result without repeating successful checks absent new evidence.

Explicit later user instructions can change this local policy.
