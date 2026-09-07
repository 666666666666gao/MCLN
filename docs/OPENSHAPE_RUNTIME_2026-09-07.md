# OpenShape object-source runtime ledger

This is a composed inference runtime, not an MCLN training environment change.

- Provider: authenticated direct SSH, `region-9.autodl.pro:33476`, user root.
- Remote root: `/root/autodl-tmp/mcln_openshape_object_source_20260907_v1`.
- Base: existing conda `/root/miniconda3/envs/bdetr`, Python 3.7.11,
  Torch 1.10.2+cu111. The base must not be updated.
- Added package prefix: the remote root's `deps/`, containing only
  `dgl-cu111==0.9.1.post1`, `torch.redstone==0.0.6`, `networkx==2.6.3` and their metadata.
- Source prefix: the root's `vendor/`, three unchanged official support files.
- Base dependencies include einops0.6.1 and huggingface-hub0.4.0; the actual
  probe checks installed requirement resolution before running the model.
- Spec: `runtime_spec_v2.json`, canonical sorted compact JSON SHA256
  `fe8ac66b8f51ed0e179a279717d0d1c2213e9a6b463785ccaa6a6e5eaf2385ac`.
- Weight: `assets/model.pt`, official G14 RGB, 388091433 bytes, SHA256
  `34949c162aca01b6fd3147ed7ccf34b448a34bdebc4a857605ea62412ad54fb9`.
- Resources: GPU0 A10040GB; OMP/MKL/OpenBLAS threads4.

The local transfer controller stages all files before starting
`build_runtime.py`. That script verifies every spec hash, installs wheel files
offline with `--no-deps --target`, compares base package inventories, and runs
the real checkpoint through the official G14 factory. Its `kernel_pass`
receipt alone is not independent approval or a scientific quality result.
The first build exited1 before the model forward because the base lacked
networkx. The original spec/log/exit remain intact. The v2 spec explicitly
adds networkx2.6.3; `build_runtime_v2.py` verifies the existing assets, adds
only this missing package, checks the base inventory and reruns the witness.
It does not reinstall DGL or download the checkpoint again.

## Exact independent witness invocation

Run this command verbatim on the authenticated server after `build_v2.exit` is 0:

```bash
bash /root/autodl-tmp/mcln_openshape_object_source_20260907_v1/witness.sh
```

That shell file sets PYTHONPATH to the two prefixes and invokes:

```bash
/root/miniconda3/envs/bdetr/bin/python -u probe_objects.py --root /root/autodl-tmp/mcln_openshape_object_source_20260907_v1 --witness-only
```

Expected: exit0, `OPENSHAPE_WITNESS` followed by a JSON shape `[1,1280]`,
positive finite norm, actual CUDA device and parameter count. The factory uses
the unmodified DGL FPS and the actual pinned pretrained checkpoint. No MCLN
checkpoint, dataset, optimizer or formal evaluator is opened by this witness.

On the current Windows controller, the prepared local transport helper is
`C:/Users/gb/.codex/tmp/review_openshape_witness_20260907.py`. With the already
authorized session SSH credential placed in `MCLN_SSH_PASSWORD`, its exact
invocation is:

```powershell
uv run --no-project --with paramiko python -u C:/Users/gb/.codex/tmp/review_openshape_witness_20260907.py
```

It only checks the build receipt and runs the remote witness above, saving the
actual stdout/stderr/exit to `reviewer_witness_command.json`. Do not print or
write the credential to the report. Do not fix the environment during review;
report any doc-versus-runtime mismatch to the main agent.

Review status: PASS, observed on 2026-09-08 by a fresh Codex agent, same-family
independence and provisional acceptance. The documented invocation exited0,
stderr was empty, and the actual output was `[1,1280]`, norm1134.4815673828125,
NVIDIA A100-PCIE-40GB, Torch1.10.2+cu111, DGL0.9.1post1, 32326080 parameters.
The first review had no credential access and executed nothing; its blocked
record remains separate. A temporary Windows-user-encrypted credential
handoff supplied the existing environment precondition for the same command
and was removed afterward. No plaintext credential file was created.

This status update records the completed review; the reviewed witness command,
runtime spec and dependency recipe are unchanged. The result covers actual
loaded CUDA inference only, not object semantic quality or REC performance.
