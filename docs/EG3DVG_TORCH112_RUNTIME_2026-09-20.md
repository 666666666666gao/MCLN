# EG-3DVG Torch 1.12 runtime ledger and witness

Provider: the existing authorized SSH server; one A100 40GB. This separate
runtime follows the author's Torch 1.12 API requirement after the existing
Torch 1.10.2 runtime failed on Tensor.is_nested in unchanged eg_attention.py.
The old PV runtime is retained. No attention API compatibility shim is added.

Declarative spec: `EG3DVG_TORCH112_ENV_2026-09-20.json`.

Canonical JSON SHA256:
`81a835e8144f7070d66feb0afa460911dec610052ced6b96c98097352cd022fd`.

Runtime root: `/root/mcln_eg3dvg_torch112_20260920_v1` on the system filesystem
(8.7 GiB available before installation). A Python 3.7.11 venv inherits existing
base dependencies and references the installed PV add-on site-packages; its
own exact Torch 1.12.0+cu116 takes precedence. No existing environment is changed.
The required resolved versions and seeded kernel checks are in the spec.

Initial build entrypoint was acceptance-root `install_torch112.py` with the official cu113 index; the final build entrypoint is `install_cu116_and_build.py` with the official cu116 index, as detailed below. Build stdout/exit and launch PID
are recorded under `/root/autodl-tmp/mcln_eg3dvg_acceptance_20260920_v1`.
The first pip20.1.1 installation failed with ENOSPC: its actual install_wheel code unpacks the whole wheel before copying it. The old Torch1.10.2 runtime was verified unchanged. Only the new venv installer was upgraded to pinned pip23.1.2, whose install_wheel reads directly from the ZipFile. The separate torch112_install_direct.py retry uses the same official Torch wheel, --ignore-installed and no dependency changes. First-attempt logs are retained as torch112_install_first.*. No old checkpoint or environment was deleted. Install success alone does not establish extension ABI or full-model readiness.

## Exact independent witness invocation

From PowerShell with login disabled, execute this exact command without editing
code or repairing the environment:

```powershell
uv run --offline --no-project --with paramiko python -u C:/Users/gb/.codex/tmp/run_mcln_authorized_20260908.py 'D:/Program Files/UserCache/gb/codex/tmp/eg3dvg_acceptance_20260920/run_torch112_witness.py'
```

It uses the existing authorized runtime credential wrapper without printing the
credential. It uploads the documented witness unchanged, acquires the existing
GPU lock and executes the new interpreter. Report any failure verbatim except
secrets; do not improvise a fix.

Expected: exit0 and `EG_TORCH112_WITNESS` with Torch1.12.0+cu116, NumPy1.21.5,
Transformers4.17.0, spaCy3.4.4, FPS[2,16], grouped features[2,8,16,8] and actual
author SWA output[1,256,288], all finite. The CUDA tests exercise the real
PointNet2 extension and author's masked attention; imports alone are not enough.
The env_spec_sha256 must match above. No optimizer or formal rows are produced.

Torch installation completed with exit0. The first kernel check exposed an actual TensorImpl undefined-symbol ABI error in the old PointNet2 binary. A separate extension is now compiled from all nine unchanged author C++/CUDA sources, using CUDA11.6 and the actual A100 architecture8.0; the author setup hardcoded8.6/8.9, so the build recipe is kept separately without modifying inference source. The source hashes and build settings are in the updated spec. The new extension must resolve under this venv, and pass real CUDA execution.

The cu113 extension build then failed before compiling: Torch1.12 requires the exact CUDA compiler version, but this host only contains nvcc11.6. The pinned Torch was changed to official1.12.0+cu116; this CUDA build differs from author cu113 and is explicitly recorded. Only the newly installed cu113 package was uninstalled after verifying its path lies in the isolated venv. install_cu116_and_build.py installs the aligned wheel and rebuilds the unchanged extension. Prior cu113 logs/spec are archived with .cu113 suffix. No runtime CUDA-version checks are disabled.

The CUDA extension built successfully. The first package import then required the author Python wrappers under pointnet2; the three unchanged author wrapper files were copied into both the installed package and its rebuild package, with hashes recorded. Executor kernel witness now passes all specified CUDA checks, including the exact newly compiled extension path.

Status when documented: runtime build and executor kernel witness passed; independent
agent-follows-doc pass pending. Full strict-load eight-row model preflight and
formal9508 must run after the runtime is validated. Neither a kernel witness nor
the fresh agent check is a REC precision result.
