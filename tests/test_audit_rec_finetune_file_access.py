import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
import sys

import pytest

import scripts.audit_rec_finetune_file_access as access_audit
from scripts.audit_rec_finetune_file_access import audit


VIOLATION_FIELDS = {
    "code",
    "pid",
    "trace_file",
    "entry_line",
    "resume_line",
    "timestamp",
    "syscall",
    "success",
    "errno",
    "raw_path",
    "candidate_path",
    "resolved_path",
    "rule",
    "detail",
}


def _c_string(value):
    return json.dumps(str(value))


def _openat(cwd, raw, result, timestamp="1710000000.000001",
            flags="O_RDONLY|O_CLOEXEC"):
    return '{} openat(AT_FDCWD<{}>, {}, {}) = {}'.format(
        timestamp, cwd, _c_string(raw), flags, result
    )


@pytest.fixture
def case(tmp_path):
    data_root = tmp_path / "DATA_ROOT"
    cwd = tmp_path / "repo"
    data_root.mkdir()
    cwd.mkdir()
    inputs_dir = data_root / "initial"
    inputs_dir.mkdir()
    inputs = []
    for name in ("backbone.pth", "parent.pth", "geometry.pth"):
        path = inputs_dir / name
        path.write_bytes(name.encode("ascii"))
        inputs.append(path)

    output_dir = data_root / "published" / "rec_run"
    output_dir.mkdir(parents=True)
    receipt = output_dir / "smoke-receipt.json"
    receipt.write_text('{"ok":true}', encoding="ascii")
    exit_file = tmp_path / "runner.exit"
    exit_file.write_text("0\n", encoding="ascii")
    report = tmp_path / "audit.json"
    expected_runner_argv = [
        os.path.abspath(sys.executable),
        str(Path(__file__).parents[1] / "scripts" /
            "train_scanrefer_rec_finetune.py"),
        "--mode",
        "smoke",
    ]
    return {
        "trace_prefix": tmp_path / "trace",
        "data_root": data_root,
        "initial_cwd": cwd,
        "input_artifacts": inputs,
        "output_dir": output_dir,
        "mode": "smoke",
        "runner_exit_code_file": exit_file,
        "receipt_path": receipt,
        "report_path": report,
        "expected_initial_runner_argv": expected_runner_argv,
    }


@pytest.fixture
def source_gate_case(case):
    input_dir = case["data_root"].parent / "source-gate-inputs"
    input_dir.mkdir()
    inputs = []
    for name in ("backbone.pth", "parent.pth", "geometry.pth"):
        path = input_dir / name
        path.write_bytes(name.encode("ascii"))
        path.chmod(0o444)
        inputs.append(path)

    train_data = case["data_root"] / "train_v3scans.pkl"
    train_data.write_bytes(b"synthetic train data")
    output_dir = case["data_root"].parent / "source-gate-output"
    output_dir.mkdir()
    runtime_scratch = case["data_root"].parent / "source-gate-runtime-scratch"
    runtime_scratch.mkdir(mode=0o700)
    runtime_scratch.chmod(0o700)
    receipt = output_dir / "smoke-receipt.json"
    receipt.write_text('{"ok":true}', encoding="ascii")
    receipt.chmod(0o444)
    runner = (
        Path(__file__).parents[1] / "scripts" /
        "probe_scanrefer_rec_source_gate.py"
    )
    case.update({
        "input_artifacts": inputs,
        "output_dir": output_dir,
        "runtime_scratch_dir": runtime_scratch,
        "receipt_path": receipt,
        "expected_initial_runner_argv": [
            os.path.abspath(sys.executable),
            str(runner),
            "--data-root", str(case["data_root"]),
            "--backbone-checkpoint", str(inputs[0]),
            "--parent-reranker", str(inputs[1]),
            "--geometry-reranker", str(inputs[2]),
            "--output-dir", str(output_dir),
            "--device", "cuda:0",
            "--probe-steps", "1",
        ],
    })
    return case


def _c_string_array(values):
    return "[{}]".format(", ".join(_c_string(value) for value in values))


def _standard_exec_lines(case, initial_argv=None):
    if initial_argv is None:
        initial_argv = case["expected_initial_runner_argv"]
    return [
        '1709999999.000001 execve({}, {}, 0x7fff) = 0'.format(
            _c_string(sys.executable), _c_string_array(initial_argv)
        ),
        '1709999999.000002 execve("/bin/sh", ["/bin/sh", "-c", '
        '"uname -p 2> /dev/null"], 0x7fff) = 0',
        '1709999999.000003 execve("/usr/bin/uname", ["uname", "-p"], '
        '0x7fff) = 0',
    ]


def _source_gate_success_lines(case):
    reads = [
        case["expected_initial_runner_argv"][1],
        case["data_root"] / "train_v3scans.pkl",
    ] + list(case["input_artifacts"])
    read_lines = [
        _openat(
            case["initial_cwd"], str(path),
            "{}<{}>".format(index + 3, path),
            "1710000000.{:06d}".format(index + 1),
        )
        for index, path in enumerate(reads)
    ]
    output = case["output_dir"]
    parent = output.parent
    staging = parent / ".{}.staging-ABC123".format(output.name)
    temporary = staging / ".smoke-receipt.json.ABC123.tmp"
    staged_receipt = staging / "smoke-receipt.json"
    proc_temporary = (
        Path("/proc/self/fd/7") / staging.name / temporary.name
    )
    runtime_lines = [
        _openat(
            case["initial_cwd"], "/dev/null", "9</dev/null<char 1:3>>",
            "1710000000.000006", "O_RDWR|O_CLOEXEC",
        ),
        _openat(
            case["initial_cwd"], "/dev/nvidia0",
            "10</dev/nvidia0<char 195:0>>", "1710000000.000007",
            "O_RDWR|O_CLOEXEC",
        ),
        _openat(
            case["initial_cwd"], "/proc/self/task/101/comm",
            "11</proc/101/task/101/comm>", "1710000000.000008",
            "O_WRONLY|O_CLOEXEC",
        ),
        _openat(
            case["initial_cwd"], "/dev/shm/torch_101_42",
            "12</dev/shm/torch_101_42>", "1710000000.000009",
            "O_RDWR|O_CREAT|O_EXCL|O_CLOEXEC",
        ),
        _openat(
            case["initial_cwd"], "/dev/shm/a1B2c3",
            "13</dev/shm/a1B2c3>", "1710000000.000010",
            "O_RDWR|O_CREAT|O_EXCL|O_CLOEXEC",
        ),
        _openat(
            case["initial_cwd"],
            str(case["runtime_scratch_dir"] / "_r1811cr"),
            "14<{}>".format(case["runtime_scratch_dir"] / "_r1811cr"),
            "1710000000.000011", "O_RDWR|O_CREAT|O_EXCL|O_CLOEXEC",
        ),
    ]
    publication_lines = [
        _openat(
            case["initial_cwd"], str(proc_temporary),
            "9<{}>".format(temporary),
            "1710000000.000012",
            "O_WRONLY|O_CREAT|O_EXCL|O_CLOEXEC",
        ),
        _openat(
            case["initial_cwd"], str(staged_receipt),
            "10<{}>".format(staged_receipt),
            "1710000000.000013",
        ),
        _openat(
            case["initial_cwd"], str(case["receipt_path"]),
            "11<{}>".format(case["receipt_path"]),
            "1710000000.000014",
        ),
    ]
    return read_lines + runtime_lines + publication_lines


def _write_source_gate_trace(case, initial_argv=None,
                             initial_executable=None,
                             initial_exec_count=1):
    if initial_argv is None:
        initial_argv = case["expected_initial_runner_argv"]
    if initial_executable is None:
        initial_executable = sys.executable
    exec_lines = _standard_exec_lines(case, initial_argv=initial_argv)
    exec_lines[0] = '1709999999.000001 execve({}, {}, 0x7fff) = 0'.format(
        _c_string(initial_executable), _c_string_array(initial_argv)
    )
    if initial_exec_count == 0:
        del exec_lines[0]
    elif initial_exec_count == 2:
        duplicate = exec_lines[0].replace(
            "1709999999.000001", "1709999999.000002"
        )
        exec_lines.insert(1, duplicate)
    return _write_trace(
        case, exec_lines + _source_gate_success_lines(case),
        include_exec_contract=False,
    )


def _set_source_gate_output(case, output_dir):
    output_dir.mkdir(parents=True)
    receipt = output_dir / "smoke-receipt.json"
    receipt.write_text('{"ok":true}', encoding="ascii")
    receipt.chmod(0o444)
    case["output_dir"] = output_dir
    case["receipt_path"] = receipt
    argv = case["expected_initial_runner_argv"]
    argv[argv.index("--output-dir") + 1] = str(output_dir)


def _write_trace(case, lines, pid=101, include_exec_contract=True):
    path = Path(str(case["trace_prefix"]) + ".{}".format(pid))
    if include_exec_contract:
        lines = _standard_exec_lines(case) + list(lines)
    path.write_text("\n".join(lines) + ("\n" if lines else ""),
                    encoding="ascii")
    return path


def _run(case, **overrides):
    kwargs = dict(case)
    kwargs.update(overrides)
    status = audit(**kwargs)
    report = json.loads(Path(kwargs["report_path"]).read_text("ascii"))
    return status, report


def _codes(report):
    return [item["code"] for item in report["violations"]]


def _tree_snapshot(root):
    root = Path(root)
    records = []
    for path in [root] + sorted(root.rglob("*")):
        metadata = path.lstat()
        relative = "." if path == root else str(path.relative_to(root))
        mode = stat.S_IMODE(metadata.st_mode)
        if path.is_symlink():
            payload = ("symlink", os.readlink(str(path)))
        elif path.is_file():
            payload = ("file", path.read_bytes())
        else:
            payload = ("directory", None)
        records.append((relative, mode, payload))
    return records


def test_exact_source_gate_smoke_trace_passes_with_external_output(
        source_gate_case):
    case = source_gate_case
    assert not str(case["output_dir"]).startswith(
        str(case["data_root"]) + os.sep
    )
    _write_trace(case, _source_gate_success_lines(case))

    status, report = _run(case)

    assert status == 0
    assert report["pass"] is True
    assert report["violations"] == []
    assert report["inputs"]["expected_initial_runner_argv"] == (
        case["expected_initial_runner_argv"]
    )
    assert report["inputs"]["runtime_scratch_dir"] == str(
        case["runtime_scratch_dir"]
    )


def test_exact_source_gate_full_probe_trace_uses_same_audit_contract(
        source_gate_case):
    case = source_gate_case
    case["expected_initial_runner_argv"][-1] = "306"
    _write_trace(case, _source_gate_success_lines(case))

    status, report = _run(case)

    assert status == 0
    assert report["pass"] is True
    assert report["violations"] == []


def test_source_gate_runtime_writes_bind_to_all_trace_pids(source_gate_case):
    case = source_gate_case
    _write_trace(case, _source_gate_success_lines(case), pid=101)
    _write_trace(case, [
        _openat(
            case["initial_cwd"], "/dev/shm/torch_101_77",
            "15</dev/shm/torch_101_77>", "1710000001.000001",
            "O_RDWR|O_CREAT|O_EXCL|O_CLOEXEC",
        ),
        _openat(
            case["initial_cwd"], "/proc/self/task/102/comm",
            "16</proc/101/task/102/comm>", "1710000001.000002",
            "O_WRONLY|O_CLOEXEC",
        ),
    ], pid=102, include_exec_contract=False)

    status, report = _run(case)

    assert status == 0
    assert report["pass"] is True
    assert report["violations"] == []


@pytest.mark.parametrize("candidate,resolved", [
    ("/dev/nvidia5", "/dev/nvidia5<char 195:5>"),
    ("/dev/nvidiactl", "/dev/nvidiactl<char 195:255>"),
    ("/dev/nvidia-modeset", "/dev/nvidia-modeset<char 195:254>"),
    ("/dev/nvidia-uvm", "/dev/nvidia-uvm<char 507:0>"),
    ("/dev/nvidia-uvm-tools", "/dev/nvidia-uvm-tools<char 507:1>"),
])
def test_source_gate_runtime_allows_real_nvidia_device_writes(
        source_gate_case, candidate, resolved):
    case = source_gate_case
    lines = _source_gate_success_lines(case) + [
        _openat(
            case["initial_cwd"], candidate, "17<{}>".format(resolved),
            "1710000000.000015", "O_RDWR|O_CLOEXEC",
        )
    ]
    _write_trace(case, lines)

    status, report = _run(case)

    assert status == 0
    assert report["pass"] is True
    assert report["violations"] == []


@pytest.mark.parametrize("candidate,resolved", [
    ("/dev/shm/torch_999_1", "/dev/shm/torch_999_1"),
    ("/proc/self/task/999/comm", "/proc/101/task/999/comm"),
    ("/proc/self/task/101/comm", "/proc/999/task/101/comm"),
    ("/proc/101/task/101/comm", "/proc/101/task/101/comm"),
])
def test_source_gate_runtime_writes_reject_unbound_pid_paths(
        source_gate_case, candidate, resolved):
    case = source_gate_case
    lines = _source_gate_success_lines(case) + [
        _openat(
            case["initial_cwd"], candidate,
            "17<{}>".format(resolved), "1710000000.000015",
            "O_RDWR|O_CREAT|O_CLOEXEC",
        )
    ]
    _write_trace(case, lines)

    status, report = _run(case)

    assert status == 1
    assert report["pass"] is False
    assert _codes(report) == ["write_allow_miss"]


@pytest.mark.parametrize("candidate,resolved", [
    ("/dev/nullx", "/dev/nullx"),
    ("/dev/null", "/dev/null<char 1:4>"),
    ("/dev/nvidia", "/dev/nvidia"),
    ("/dev/nvidia0.tmp", "/dev/nvidia0.tmp"),
    ("/dev/nvidiamodeset", "/dev/nvidiamodeset"),
    ("/dev/nvidiauvm", "/dev/nvidiauvm"),
    ("/dev/nvidiauvm-tools", "/dev/nvidiauvm-tools"),
    ("/dev/shm/torch_101_1_extra", "/dev/shm/torch_101_1_extra"),
    ("/dev/shm/torch_101_x", "/dev/shm/torch_101_x"),
    ("/dev/shm/a1B2c", "/dev/shm/a1B2c"),
    ("/dev/shm/a1B2c3x", "/dev/shm/a1B2c3x"),
])
def test_source_gate_runtime_write_near_misses_fail_closed(
        source_gate_case, candidate, resolved):
    case = source_gate_case
    lines = _source_gate_success_lines(case) + [
        _openat(
            case["initial_cwd"], candidate,
            "17<{}>".format(resolved), "1710000000.000015",
            "O_RDWR|O_CREAT|O_CLOEXEC",
        )
    ]
    _write_trace(case, lines)

    status, report = _run(case)

    assert status == 1
    assert report["pass"] is False
    assert _codes(report) == ["write_allow_miss"]


@pytest.mark.parametrize("path", [
    "/home/example/.cache/model.bin",
    "/opt/cache/value.bin",
    "/opt/validation/value.bin",
    "/opt/val/value.bin",
    "/opt/test/value.bin",
])
def test_source_gate_smoke_rejects_external_forbidden_dependency_directory(
        source_gate_case, path):
    case = source_gate_case
    lines = _source_gate_success_lines(case) + [
        _openat(
            case["initial_cwd"], path, "17<{}>".format(path),
            "1710000000.000015",
        )
    ]
    _write_trace(case, lines)

    status, report = _run(case)

    assert status == 1
    assert report["pass"] is False
    assert _codes(report) == ["deny_path"]


def test_source_gate_smoke_allows_system_cache_files_and_pycache(
        source_gate_case):
    case = source_gate_case
    paths = [
        "/etc/ld.so.cache",
        "/usr/lib/gconv/gconv-modules.cache",
        "/usr/lib/python3.7/__pycache__/module.cpython-37.pyc",
    ]
    lines = _source_gate_success_lines(case) + [
        _openat(
            case["initial_cwd"], path,
            "{}<{}>".format(index + 17, path),
            "1710000000.{:06d}".format(index + 15),
        )
        for index, path in enumerate(paths)
    ]
    _write_trace(case, lines)

    status, report = _run(case)

    assert status == 0
    assert report["pass"] is True
    assert report["violations"] == []


def test_source_gate_smoke_allows_exact_named_runtime_code_dependencies(
        source_gate_case):
    case = source_gate_case
    project = case["initial_cwd"]
    paths = [
        project / "scripts" / "cache_scanrefer_rec_candidates.py",
        project / "scripts" / "rec_geometry_cache.py",
        project / "scripts" / "__pycache__"
        / "cache_scanrefer_rec_candidates.cpython-37.pyc",
        project / "scripts" / "__pycache__"
        / "rec_geometry_cache.cpython-37.pyc",
    ]
    lines = _source_gate_success_lines(case) + [
        _openat(
            case["initial_cwd"], path,
            "{}<{}>".format(index + 17, path),
            "1710000000.{:06d}".format(index + 15),
        )
        for index, path in enumerate(paths)
    ]
    _write_trace(case, lines)

    status, report = _run(case)

    assert status == 0
    assert report["pass"] is True
    assert report["violations"] == []


@pytest.mark.parametrize("relative", [
    "scripts/cache_scanrefer_rec_candidates_extra.py",
    "scripts/rec_geometry_cache_backup.py",
    "scripts/__pycache__/cache_scanrefer_rec_candidates_extra.cpython-37.pyc",
])
def test_source_gate_named_runtime_code_exception_rejects_near_collisions(
        source_gate_case, relative):
    case = source_gate_case
    path = case["initial_cwd"] / relative
    lines = _source_gate_success_lines(case) + [
        _openat(
            case["initial_cwd"], path, "17<{}>".format(path),
            "1710000000.000015",
        )
    ]
    _write_trace(case, lines)

    status, report = _run(case)

    assert status == 1
    assert report["pass"] is False
    assert _codes(report) == ["deny_path"]


@pytest.mark.parametrize("location", [
    "inside_output", "inside_staging", "outside_output",
])
def test_source_gate_smoke_rejects_undeclared_write(
        source_gate_case, location):
    case = source_gate_case
    if location == "inside_output":
        undeclared = case["output_dir"] / "undeclared-checkpoint.pth"
    elif location == "inside_staging":
        undeclared = (
            case["output_dir"].parent /
            ".{}.staging-ABC123".format(case["output_dir"].name) /
            "undeclared-checkpoint.pth"
        )
    else:
        undeclared = (
            case["output_dir"].parent / "undeclared-checkpoint.pth"
        )
    lines = _source_gate_success_lines(case) + [
        _openat(
            case["initial_cwd"], str(undeclared),
            "10<{}>".format(undeclared),
            "1710000000.000007",
            "O_WRONLY|O_CREAT|O_CLOEXEC",
        )
    ]
    _write_trace(case, lines)

    status, report = _run(case)

    assert status == 1
    assert report["pass"] is False
    assert _codes(report) == ["write_allow_miss"]
    assert report["violations"][0]["resolved_path"] == str(undeclared)


def test_source_gate_smoke_rejects_receipt_not_mode_0444(source_gate_case):
    case = source_gate_case
    case["receipt_path"].chmod(0o644)
    _write_trace(case, _source_gate_success_lines(case))

    status, report = _run(case)

    assert status == 1
    assert report["pass"] is False
    assert _codes(report) == ["receipt_invalid"]
    assert report["violations"][0]["rule"] == (
        "source-gate-receipt-publication"
    )


def test_source_gate_smoke_rejects_multiple_published_files(source_gate_case):
    case = source_gate_case
    (case["output_dir"] / "extra.json").write_text(
        "{}", encoding="ascii"
    )
    _write_trace(case, _source_gate_success_lines(case))

    status, report = _run(case)

    assert status == 1
    assert report["pass"] is False
    assert _codes(report) == ["receipt_invalid"]
    assert report["violations"][0]["rule"] == (
        "source-gate-receipt-publication"
    )


@pytest.mark.parametrize("write_count", [0, 2])
def test_source_gate_smoke_requires_exactly_one_bound_receipt_write(
        source_gate_case, write_count):
    case = source_gate_case
    lines = _source_gate_success_lines(case)
    publication = [
        line for line in lines
        if "/proc/self/fd/7/" in line and "O_WRONLY" in line
    ]
    assert len(publication) == 1
    lines.remove(publication[0])
    if write_count == 2:
        lines.extend([
            publication[0],
            publication[0].replace(
                "1710000000.000012", "1710000000.000015"
            ),
        ])
    _write_trace(case, lines)

    status, report = _run(case)

    assert status == 1
    assert report["pass"] is False
    assert _codes(report) == ["receipt_invalid"]
    assert report["violations"][0]["rule"] == (
        "source-gate-publication-write-exactly-once"
    )


def test_source_gate_smoke_rejects_direct_final_receipt_write(
        source_gate_case):
    case = source_gate_case
    lines = [
        line for line in _source_gate_success_lines(case)
        if not ("/proc/self/fd/7/" in line and "O_WRONLY" in line)
    ] + [
        _openat(
            case["initial_cwd"], str(case["receipt_path"]),
            "17<{}>".format(case["receipt_path"]),
            "1710000000.000015", "O_WRONLY|O_CLOEXEC",
        )
    ]
    _write_trace(case, lines)

    status, report = _run(case)

    assert status == 1
    assert report["pass"] is False
    assert _codes(report) == ["receipt_invalid"]
    assert report["violations"][0]["rule"] == (
        "source-gate-publication-write-exactly-once"
    )


@pytest.mark.parametrize("mutation", [
    "suffix_mismatch", "nonnumeric_fd", "resolved_outside",
])
def test_source_gate_smoke_rejects_unbound_proc_fd_output_alias(
        source_gate_case, mutation):
    case = source_gate_case
    output = case["output_dir"]
    staging = output.parent / ".{}.staging-ABC123".format(output.name)
    resolved = staging / ".smoke-receipt.json.ABC123.tmp"
    candidate = Path("/proc/self/fd/7") / staging.name / resolved.name
    if mutation == "suffix_mismatch":
        candidate = candidate.with_name(
            ".smoke-receipt.json.DIFFERENT.tmp"
        )
    elif mutation == "nonnumeric_fd":
        candidate = Path("/proc/self/fd/not-a-fd") / staging.name / resolved.name
    else:
        resolved = output.parent / "undeclared-output.pth"
    lines = _source_gate_success_lines(case) + [
        _openat(
            case["initial_cwd"], str(candidate),
            "17<{}>".format(resolved), "1710000000.000015",
            "O_WRONLY|O_CREAT|O_CLOEXEC",
        )
    ]
    _write_trace(case, lines)

    status, report = _run(case)

    assert status == 1
    assert report["pass"] is False
    assert _codes(report) == ["write_allow_miss"]


def test_source_gate_smoke_rejects_staging_sibling_residue(source_gate_case):
    case = source_gate_case
    residue = (
        case["output_dir"].parent /
        ".{}.staging-LEFTOVER".format(case["output_dir"].name)
    )
    residue.mkdir()
    _write_trace(case, _source_gate_success_lines(case))

    status, report = _run(case)

    assert status == 1
    assert report["pass"] is False
    assert _codes(report) == ["receipt_invalid"]
    assert report["violations"][0]["rule"] == (
        "source-gate-receipt-publication"
    )


def test_source_gate_smoke_rejects_missing_receipt(source_gate_case):
    case = source_gate_case
    case["receipt_path"].unlink()
    _write_trace(case, _source_gate_success_lines(case))

    status, report = _run(case)

    assert status == 1
    assert report["pass"] is False
    assert _codes(report) == ["receipt_missing"]


def test_source_gate_smoke_rejects_wrong_receipt_name(source_gate_case):
    case = source_gate_case
    wrong = case["receipt_path"].with_name("source-gate-receipt.json")
    case["receipt_path"].rename(wrong)
    _write_trace(case, _source_gate_success_lines(case))

    status, report = _run(case, receipt_path=wrong)

    assert status == 2
    assert report["pass"] is False
    assert _codes(report) == ["configuration_error"]
    assert report["violations"][0]["rule"] == "cli-contract"


@pytest.mark.parametrize("location", [
    "data_root", "project", "input_tree", "output", "control", "cache",
])
def test_source_gate_smoke_rejects_overlapping_runtime_scratch(
        source_gate_case, location):
    case = source_gate_case
    if location == "data_root":
        scratch = case["data_root"] / "runtime-scratch"
    elif location == "project":
        scratch = case["initial_cwd"] / "runtime-scratch"
    elif location == "input_tree":
        scratch = case["input_artifacts"][0].parent / "runtime-scratch"
    elif location == "output":
        scratch = case["output_dir"] / "runtime-scratch"
    elif location == "control":
        scratch = case["trace_prefix"]
    else:
        scratch = case["data_root"].parent / ".cache" / "runtime-scratch"
    scratch.mkdir(parents=True)
    scratch.chmod(0o700)
    case["runtime_scratch_dir"] = scratch
    _write_trace(case, _source_gate_success_lines(case))

    status, report = _run(case)

    assert status == 2
    assert report["pass"] is False
    assert _codes(report) == ["configuration_error"]


@pytest.mark.parametrize("mutation", [
    "missing", "mode", "symlink", "nonempty",
])
def test_source_gate_smoke_requires_strict_runtime_scratch(
        source_gate_case, mutation):
    case = source_gate_case
    scratch = case["runtime_scratch_dir"]
    lines = _source_gate_success_lines(case)
    if mutation == "missing":
        case["runtime_scratch_dir"] = None
    elif mutation == "mode":
        scratch.chmod(0o755)
    elif mutation == "symlink":
        alias = scratch.with_name("runtime-scratch-alias")
        alias.symlink_to(scratch, target_is_directory=True)
        case["runtime_scratch_dir"] = alias
    else:
        (scratch / "leftover.tmp").write_bytes(b"leftover")
    _write_trace(case, lines)

    status, report = _run(case)

    assert status == 2
    assert report["pass"] is False
    assert _codes(report) == ["configuration_error"]


def test_source_gate_smoke_rejects_output_inside_data_root(source_gate_case):
    case = source_gate_case
    _set_source_gate_output(
        case, case["data_root"] / "published-source-gate"
    )
    _write_trace(case, _source_gate_success_lines(case))

    status, report = _run(case)

    assert status == 2
    assert report["pass"] is False
    assert _codes(report) == ["configuration_error"]


@pytest.mark.parametrize("location", [
    "report_inside", "report_receipt", "report_staging",
    "trace_inside", "exit_inside", "report_input", "report_input_tree",
    "report_exit", "report_trace", "report_data", "report_project",
    "report_scratch",
])
def test_source_gate_smoke_rejects_control_path_in_output_namespace(
        source_gate_case, location):
    case = source_gate_case
    output = case["output_dir"]
    if location == "report_inside":
        case["report_path"] = output / "audit.json"
    elif location == "report_receipt":
        case["report_path"] = case["receipt_path"]
    elif location == "report_staging":
        case["report_path"] = (
            output.parent /
            ".{}.staging-REPORT".format(output.name) /
            "audit.json"
        )
    elif location == "trace_inside":
        case["trace_prefix"] = output / "trace"
    elif location == "exit_inside":
        relocated = output / "runner.exit"
        case["runner_exit_code_file"].rename(relocated)
        case["runner_exit_code_file"] = relocated
    elif location == "report_input":
        case["report_path"] = case["input_artifacts"][0]
    elif location == "report_input_tree":
        case["report_path"] = (
            case["input_artifacts"][0].parent / "audit.json"
        )
    elif location == "report_exit":
        case["report_path"] = case["runner_exit_code_file"]
    elif location == "report_trace":
        case["report_path"] = Path(str(case["trace_prefix"]) + ".101")
    elif location == "report_data":
        case["report_path"] = case["data_root"] / "audit.json"
    elif location == "report_project":
        case["report_path"] = case["initial_cwd"] / "audit.json"
    else:
        case["report_path"] = case["runtime_scratch_dir"] / "audit.json"
    _write_trace(case, _source_gate_success_lines(case))

    if location.startswith("report_"):
        report_path = Path(case["report_path"])
        existed = report_path.is_file()
        before_bytes = report_path.read_bytes() if existed else None
        before_mode = (
            stat.S_IMODE(report_path.stat().st_mode) if existed else None
        )
        receipt_bytes = case["receipt_path"].read_bytes()
        receipt_mode = stat.S_IMODE(case["receipt_path"].stat().st_mode)
        output_entries = sorted(case["output_dir"].iterdir())

        status = audit(**case)

        assert status == 2
        if existed:
            assert report_path.read_bytes() == before_bytes
            assert stat.S_IMODE(report_path.stat().st_mode) == before_mode
        else:
            assert not report_path.exists()
        assert case["receipt_path"].read_bytes() == receipt_bytes
        assert stat.S_IMODE(case["receipt_path"].stat().st_mode) == receipt_mode
        assert sorted(case["output_dir"].iterdir()) == output_entries
        assert list(case["runtime_scratch_dir"].iterdir()) == []
    else:
        status, report = _run(case)
        assert status == 2
        assert report["pass"] is False
        assert _codes(report) == ["configuration_error"]


@pytest.mark.parametrize("kind", ["file", "symlink", "directory"])
def test_source_gate_smoke_requires_fresh_report_path(
        source_gate_case, kind):
    case = source_gate_case
    report_path = case["report_path"]
    target = report_path.with_name("existing-report-target")
    if kind == "file":
        report_path.write_bytes(b"existing report")
    elif kind == "symlink":
        target.write_bytes(b"symlink target")
        report_path.symlink_to(target)
    else:
        report_path.mkdir()
    _write_trace(case, _source_gate_success_lines(case))

    if kind == "file":
        before = report_path.read_bytes()
    elif kind == "symlink":
        before = target.read_bytes()
    else:
        before = list(report_path.iterdir())

    status = audit(**case)

    assert status == 2
    if kind == "file":
        assert report_path.read_bytes() == before
    elif kind == "symlink":
        assert report_path.is_symlink()
        assert target.read_bytes() == before
    else:
        assert report_path.is_dir()
        assert list(report_path.iterdir()) == before


@pytest.mark.parametrize("invalid_config", [
    "artifact_count", "mode", "data_root",
])
@pytest.mark.parametrize("report_target", ["input", "receipt"])
def test_source_gate_early_configuration_error_preserves_unsafe_report_target(
        source_gate_case, invalid_config, report_target):
    case = source_gate_case
    target = (
        case["input_artifacts"][0]
        if report_target == "input" else case["receipt_path"]
    )
    before_bytes = target.read_bytes()
    before_mode = stat.S_IMODE(target.stat().st_mode)
    case["report_path"] = target
    if invalid_config == "artifact_count":
        case["input_artifacts"] = case["input_artifacts"][:2]
    elif invalid_config == "mode":
        case["mode"] = "invalid"
    else:
        case["data_root"] = case["data_root"].with_name("missing-data-root")

    status = audit(**case)

    assert status == 2
    assert target.read_bytes() == before_bytes
    assert stat.S_IMODE(target.stat().st_mode) == before_mode


@pytest.mark.parametrize("control", [
    "report_data_alias", "trace_output_alias", "exit_input_alias",
])
def test_source_gate_smoke_rejects_control_symlink_ancestor(
        source_gate_case, control):
    case = source_gate_case
    alias = case["data_root"].parent / "control-alias"
    if control == "report_data_alias":
        alias.symlink_to(case["data_root"], target_is_directory=True)
        case["report_path"] = alias / "audit.json"
        protected_path = case["data_root"] / "audit.json"
    elif control == "trace_output_alias":
        alias.symlink_to(case["output_dir"], target_is_directory=True)
        case["trace_prefix"] = alias / "trace"
        protected_path = case["output_dir"] / "trace.101"
    else:
        input_parent = case["input_artifacts"][0].parent
        alias.symlink_to(input_parent, target_is_directory=True)
        relocated = alias / "runner.exit"
        case["runner_exit_code_file"].rename(relocated)
        case["runner_exit_code_file"] = relocated
        protected_path = input_parent / "runner.exit"
    _write_trace(case, _source_gate_success_lines(case))

    status = audit(**case)

    assert status == 2
    if control == "report_data_alias":
        assert not protected_path.exists()
    else:
        assert protected_path.exists()


def test_source_gate_smoke_rejects_write_to_input(source_gate_case):
    case = source_gate_case
    input_path = case["input_artifacts"][0]
    lines = _source_gate_success_lines(case) + [
        _openat(
            case["initial_cwd"], str(input_path),
            "12<{}>".format(input_path),
            "1710000000.000009",
            "O_WRONLY|O_CLOEXEC",
        )
    ]
    _write_trace(case, lines)

    status, report = _run(case)

    assert status == 1
    assert report["pass"] is False
    assert _codes(report) == ["write_allow_miss"]
    assert report["violations"][0]["resolved_path"] == str(input_path)


@pytest.mark.parametrize("target", ["input", "undeclared"])
def test_source_gate_smoke_rejects_write_via_open_by_handle(
        source_gate_case, target):
    case = source_gate_case
    if target == "input":
        path = case["input_artifacts"][0]
    else:
        path = case["output_dir"].parent / "undeclared-handle.pth"
    lines = _source_gate_success_lines(case) + [
        "1710000000.000015 open_by_handle_at(5<{}>, "
        "{{handle_bytes=8, handle_type=1, f_handle=0x01}}, "
        "O_WRONLY|O_CLOEXEC) = 17<{}>".format(
            case["data_root"], path
        )
    ]
    _write_trace(case, lines)

    status, report = _run(case)

    assert status == 1
    assert report["pass"] is False
    assert _codes(report) == ["write_allow_miss"]
    assert report["violations"][0]["resolved_path"] == str(path)


@pytest.mark.parametrize("forbidden", [
    "validation", "val", "test", "cache",
])
def test_source_gate_smoke_rejects_forbidden_project_data_path(
        source_gate_case, forbidden):
    case = source_gate_case
    path = case["initial_cwd"] / forbidden / "payload.bin"
    lines = _source_gate_success_lines(case) + [
        _openat(
            case["initial_cwd"], str(path), "12<{}>".format(path),
            "1710000000.000009",
        )
    ]
    _write_trace(case, lines)

    status, report = _run(case)

    assert status == 1
    assert report["pass"] is False
    assert _codes(report) == ["deny_path"]
    assert report["violations"][0]["resolved_path"] == str(path)


@pytest.mark.parametrize("token_index", range(16))
def test_source_gate_smoke_rejects_every_initial_argv_token_change(
        source_gate_case, token_index):
    case = source_gate_case
    actual = list(case["expected_initial_runner_argv"])
    actual[token_index] += ".mutated"
    _write_source_gate_trace(case, initial_argv=actual)

    status, report = _run(case)

    assert status == 2
    assert report["pass"] is False
    assert "trace_parse_error" in _codes(report)


@pytest.mark.parametrize("mutation", [
    "reordered", "duplicate_token", "missing_token",
])
def test_source_gate_smoke_rejects_initial_argv_shape_change(
        source_gate_case, mutation):
    case = source_gate_case
    actual = list(case["expected_initial_runner_argv"])
    if mutation == "reordered":
        actual[-4:] = actual[-2:] + actual[-4:-2]
    elif mutation == "duplicate_token":
        actual.extend(("--probe-steps", "1"))
    else:
        del actual[-2:]
    _write_source_gate_trace(case, initial_argv=actual)

    status, report = _run(case)

    assert status == 2
    assert report["pass"] is False
    assert "trace_parse_error" in _codes(report)


@pytest.mark.parametrize("token_index", range(2, 16))
def test_source_gate_rejects_synchronized_cli_token_change(
        source_gate_case, token_index):
    case = source_gate_case
    case["expected_initial_runner_argv"][token_index] += ".mutated"
    _write_source_gate_trace(case)

    status, report = _run(case)

    assert status == 2
    assert report["pass"] is False
    assert _codes(report) == ["configuration_error"]


@pytest.mark.parametrize("mutation", [
    "reordered", "duplicate_token", "missing_token",
])
def test_source_gate_rejects_synchronized_cli_shape_change(
        source_gate_case, mutation):
    case = source_gate_case
    argv = case["expected_initial_runner_argv"]
    if mutation == "reordered":
        argv[-4:] = argv[-2:] + argv[-4:-2]
    elif mutation == "duplicate_token":
        argv.extend(("--probe-steps", "1"))
    else:
        del argv[-2:]
    _write_source_gate_trace(case)

    status, report = _run(case)

    assert status == 2
    assert report["pass"] is False
    assert _codes(report) == ["configuration_error"]


def test_source_gate_rejects_samefile_runner_alias_in_config(source_gate_case):
    case = source_gate_case
    runner = Path(case["expected_initial_runner_argv"][1])
    alias = case["data_root"].parent / "source-gate-runner-alias.py"
    alias.symlink_to(runner)
    case["expected_initial_runner_argv"][1] = str(alias)
    _write_source_gate_trace(case)

    status, report = _run(case)

    assert status == 2
    assert report["pass"] is False
    assert _codes(report) == ["configuration_error"]


def test_source_gate_rejects_relative_samefile_runner_in_config(
        source_gate_case):
    case = source_gate_case
    runner = Path(case["expected_initial_runner_argv"][1])
    (case["initial_cwd"] / "scripts").symlink_to(
        runner.parent, target_is_directory=True
    )
    case["expected_initial_runner_argv"][1] = (
        "scripts/probe_scanrefer_rec_source_gate.py"
    )
    _write_source_gate_trace(case)

    status, report = _run(case)

    assert status == 2
    assert report["pass"] is False
    assert _codes(report) == ["configuration_error"]


def test_source_gate_rejects_module_runner_in_config(source_gate_case):
    case = source_gate_case
    direct = case["expected_initial_runner_argv"]
    case["expected_initial_runner_argv"] = [
        direct[0], "-m", "scripts.probe_scanrefer_rec_source_gate",
    ] + direct[2:]
    _write_source_gate_trace(case)

    status, report = _run(case)

    assert status == 2
    assert report["pass"] is False
    assert _codes(report) == ["configuration_error"]


def test_source_gate_smoke_rejects_different_initial_interpreter(
        source_gate_case):
    case = source_gate_case
    other = case["data_root"].parent / "other-python"
    other.write_bytes(b"not the configured interpreter")
    actual = list(case["expected_initial_runner_argv"])
    actual[0] = str(other)
    _write_source_gate_trace(
        case, initial_argv=actual, initial_executable=other
    )

    status, report = _run(case)

    assert status == 2
    assert report["pass"] is False
    assert "trace_parse_error" in _codes(report)


@pytest.mark.parametrize("initial_exec_count", [0, 2])
def test_source_gate_smoke_rejects_missing_or_duplicate_initial_exec(
        source_gate_case, initial_exec_count):
    case = source_gate_case
    _write_source_gate_trace(case, initial_exec_count=initial_exec_count)

    status, report = _run(case)

    assert status in (1, 2)
    assert report["pass"] is False
    assert report["violations"]


def test_source_gate_dependency_rule_does_not_match_other_smoke_runner(case):
    path = case["initial_cwd"] / "cache" / "ordinary-dependency.bin"
    _write_trace(case, [
        _openat(
            case["initial_cwd"], str(path), "12<{}>".format(path)
        )
    ])

    status, report = _run(case)

    assert status == 0
    assert report["pass"] is True
    assert report["violations"] == []


def test_other_smoke_runner_cannot_write_source_gate_undeclared_output(case):
    path = case["output_dir"] / "source-gate-checkpoint.pth"
    _write_trace(case, [
        _openat(
            case["initial_cwd"], str(path), "12<{}>".format(path),
            flags="O_WRONLY|O_CREAT|O_CLOEXEC",
        )
    ])

    status, report = _run(case)

    assert status == 1
    assert report["pass"] is False
    assert _codes(report) == ["write_allow_miss"]


def test_source_gate_smoke_output_rule_does_not_match_production_mode(
        source_gate_case):
    case = source_gate_case
    selection = case["receipt_path"].with_name("selection.json")
    case["receipt_path"].rename(selection)
    case["receipt_path"] = selection
    _write_trace(case, _source_gate_success_lines(case))

    status, report = _run(case, mode="production")

    assert status == 2
    assert report["pass"] is False
    assert _codes(report) == ["configuration_error"]


class _BytesPathLike:
    def __init__(self, path):
        self._path = os.fsencode(path)

    def __fspath__(self):
        return self._path


@pytest.mark.parametrize("wrapper", [os.fsencode, _BytesPathLike])
@pytest.mark.parametrize("field", [
    "trace_prefix",
    "data_root",
    "initial_cwd",
    "input_artifacts",
    "output_dir",
    "runner_exit_code_file",
    "receipt_path",
    "report_path",
])
def test_bytes_public_paths_write_canonical_configuration_report(
        case, field, wrapper):
    _write_trace(case, [
        _openat(case["initial_cwd"], "safe", "3<{}>".format(
            case["input_artifacts"][0]
        ))
    ])
    kwargs = dict(case)
    if field == "input_artifacts":
        kwargs[field] = [wrapper(path) for path in kwargs[field]]
    else:
        kwargs[field] = wrapper(kwargs[field])

    status = audit(**kwargs)

    assert status == 2
    report_bytes = case["report_path"].read_bytes()
    report = json.loads(report_bytes.decode("ascii"))
    assert _codes(report) == ["configuration_error"]
    assert report_bytes == json.dumps(
        report,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


@pytest.mark.parametrize("mutation", [
    "too_few",
    "too_many",
    "non_iterable",
    "bad_path",
    "missing",
    "duplicate",
    "canonical_duplicate",
    "symlink_alias",
])
def test_invalid_input_artifact_config_writes_status2_report(case, mutation):
    artifacts = list(case["input_artifacts"])
    if mutation == "too_few":
        artifacts.pop()
    elif mutation == "too_many":
        extra = case["data_root"] / "initial" / "extra.pth"
        extra.write_bytes(b"extra")
        artifacts.append(extra)
    elif mutation == "non_iterable":
        artifacts = None
    elif mutation == "bad_path":
        artifacts[1] = object()
    elif mutation == "missing":
        artifacts[1] = artifacts[1].with_name("missing.pth")
    elif mutation == "duplicate":
        artifacts[1] = artifacts[0]
    elif mutation == "canonical_duplicate":
        artifacts[1] = (
            artifacts[0].parent / ".." / artifacts[0].parent.name /
            artifacts[0].name
        )
    elif mutation == "symlink_alias":
        alias = artifacts[0].with_name("backbone-alias.pth")
        alias.symlink_to(artifacts[0])
        artifacts[1] = alias

    status, report = _run(case, input_artifacts=artifacts)

    assert status == 2
    assert report["pass"] is False
    assert _codes(report) == ["configuration_error"]
    assert report["violations"][0]["rule"] == "cli-contract"


@pytest.mark.parametrize("mutation", [
    "none",
    "empty",
    "non_array",
    "non_string",
    "nul",
    "relative_argv0",
    "different_interpreter",
])
def test_invalid_expected_runner_argv_writes_status2_report(case, mutation):
    argv = list(case["expected_initial_runner_argv"])
    if mutation == "none":
        argv = None
    elif mutation == "empty":
        argv = []
    elif mutation == "non_array":
        argv = {"argv": argv}
    elif mutation == "non_string":
        argv[1] = None
    elif mutation == "nul":
        argv[1] += "\x00suffix"
    elif mutation == "relative_argv0":
        argv[0] = "python"
    elif mutation == "different_interpreter":
        other = case["data_root"].parent / "different-python"
        other.write_bytes(b"not the audit interpreter")
        argv[0] = str(other)

    status, report = _run(case, expected_initial_runner_argv=argv)

    assert status == 2
    assert report["pass"] is False
    assert _codes(report) == ["configuration_error"]
    assert report["violations"][0]["rule"] == "cli-contract"


def test_safe_resolved_openat_passes_and_report_is_canonical(case):
    trace = _write_trace(case, [
        _openat(case["initial_cwd"], "ignored-name", "3<{}>".format(
            case["input_artifacts"][0]
        ))
    ])

    status, report = _run(case)

    assert status == 0
    assert report["schema"] == "rec-finetune-file-access-audit-v1"
    assert report["pass"] is True
    assert report["counts"]["syscalls"] == 4
    assert report["inputs"]["traces"] == [{
        "path": str(trace),
        "pid": 101,
        "sha256": hashlib.sha256(trace.read_bytes()).hexdigest(),
        "size": trace.stat().st_size,
    }]
    encoded = json.dumps(
        report, sort_keys=True, separators=(",", ":"),
        ensure_ascii=True, allow_nan=False,
    ).encode("ascii")
    assert case["report_path"].read_bytes() == encoded


def test_deny_checks_both_raw_and_success_resolved_paths(case):
    safe = case["input_artifacts"][0]
    denied = case["data_root"] / "val_v3scans.pkl"
    _write_trace(case, [
        _openat(case["initial_cwd"], "safe-name", "3<{}>".format(denied)),
        _openat(case["initial_cwd"], "ScanRefer_filtered_val.json",
                "4<{}>".format(safe), "1710000000.000002"),
    ])

    status, report = _run(case)

    assert status == 1
    assert _codes(report) == ["deny_path", "deny_path"]
    assert report["violations"][0]["resolved_path"] == str(denied)
    assert report["violations"][1]["raw_path"] == (
        "ScanRefer_filtered_val.json"
    )


def test_failed_denied_attempt_records_errno_and_candidate(case):
    raw = "val_v3scans.pkl"
    _write_trace(case, [
        _openat(case["initial_cwd"], raw,
                "-1 ENOENT (No such file or directory)")
    ])

    status, report = _run(case)

    violation = report["violations"][0]
    assert status == 1
    assert violation["success"] is False
    assert violation["errno"] == "ENOENT"
    assert violation["candidate_path"] == str(case["initial_cwd"] / raw)


def test_relative_at_fdcwd_and_annotated_numeric_dirfd_are_recovered(case):
    _write_trace(case, [
        '{} openat(AT_FDCWD<{}>, "val_v3scans.pkl", O_RDONLY) = '
        '-1 ENOENT ({})'.format(
            "1710000000.000001", case["initial_cwd"], "missing"
        ),
        '{} openat(7<{}>, "val/scene.bin", O_RDONLY) = -1 EACCES ({})'.format(
            "1710000000.000002", case["data_root"] / "superpoints",
            "denied",
        ),
    ])

    status, report = _run(case)

    assert status == 1
    assert [v["candidate_path"] for v in report["violations"]] == [
        str(case["initial_cwd"] / "val_v3scans.pkl"),
        str(case["data_root"] / "superpoints" / "val" / "scene.bin"),
    ]


def test_unknown_relative_dirfd_fails_closed_as_uncertain(case):
    _write_trace(case, [
        '1710000000.000001 openat(7, "ordinary.json", O_RDONLY) = '
        '-1 ENOENT (missing)'
    ])

    status, report = _run(case)

    assert status == 1
    assert _codes(report) == ["uncertain_path"]
    assert report["violations"][0]["candidate_path"] is None


def test_data_root_component_boundary_does_not_capture_data_root2(case):
    neighbor = Path(str(case["data_root"]) + "2") / "secret.bin"
    _write_trace(case, [
        _openat(case["initial_cwd"], str(neighbor), "3<{}>".format(neighbor))
    ])

    status, report = _run(case)

    assert status == 0
    assert report["violations"] == []


def test_openat2_nested_struct_parses_and_uses_allowlist(case):
    path = case["data_root"] / "roberta-base" / "config.json"
    _write_trace(case, [
        '1710000000.000001 openat2(AT_FDCWD<{}>, {}, '
        '{{flags=O_RDONLY|O_CLOEXEC, resolve=RESOLVE_BENEATH|'
        'RESOLVE_NO_MAGICLINKS}}, 24) = 8<{}>'.format(
            case["initial_cwd"], _c_string(path), path
        )
    ])

    status, report = _run(case)

    assert status == 0
    assert report["counts"]["syscalls"] == 4


def test_success_under_data_root_outside_allowlist_is_violation(case):
    path = case["data_root"] / "private" / "ordinary.bin"
    _write_trace(case, [
        _openat(case["initial_cwd"], str(path), "3<{}>".format(path))
    ])

    status, report = _run(case)

    assert status == 1
    assert _codes(report) == ["data_root_allow_miss"]


def test_all_declared_train_inputs_are_allowed(case):
    root = case["data_root"]
    paths = list(case["input_artifacts"]) + [
        root / "train_v3scans.pkl",
        root / "scanrefer" / "ScanRefer_filtered_train.txt",
        root / "ScanRefer" / "ScanRefer_filtered_train.json",
        root / "superpoints" / "train" / "scene0001_02_superpoint.pth",
        root / "group_free_pred_bboxes" / "group_free_pred_bboxes_train" /
        "scene0001_02.npy",
        root / "roberta-base" / "tokenizer_config.json",
        root / "roberta-base" / "tokenizer.json",
        root / "roberta-base" / "vocab.json",
        root / "roberta-base" / "merges.txt",
        root / "roberta-base" / "pytorch_model.bin",
    ]
    _write_trace(case, [
        _openat(case["initial_cwd"], str(path),
                "{}<{}>".format(index + 3, path),
                "1710000000.{:06d}".format(index + 1))
        for index, path in enumerate(paths)
    ])

    status, report = _run(case)

    assert status == 0
    assert report["counts"]["syscalls"] == len(paths) + 3


def test_obsolete_root_level_group_free_train_path_is_not_allowed(case):
    path = (
        case["data_root"] / "group_free_pred_bboxes_train" /
        "scene0001_02.npy"
    )
    _write_trace(case, [
        _openat(case["initial_cwd"], str(path), "3<{}>".format(path))
    ])

    status, report = _run(case)

    assert status == 1
    assert _codes(report) == ["data_root_allow_miss"]


@pytest.mark.parametrize("mode,final_name,temporary_name", [
    ("smoke", "smoke-receipt.json", ".smoke-receipt.json.ABC123.tmp"),
    ("production", "backbone.pth", ".backbone.pth.ABC123.tmp"),
])
def test_mode_specific_staging_final_and_tempfiles_are_allowed(
        case, mode, final_name, temporary_name):
    output = case["output_dir"]
    if mode == "production":
        receipt = output / "selection.json"
        receipt.write_text("{}", encoding="ascii")
    else:
        receipt = case["receipt_path"]
    staging = output.parent / (".{}.staging-ABC123".format(output.name))
    paths = [staging, staging / temporary_name, staging / final_name,
             output / final_name, output.parent]
    _write_trace(case, [
        _openat(case["initial_cwd"], str(path),
                "{}<{}>".format(index + 3, path),
                "1710000000.{:06d}".format(index + 1))
        for index, path in enumerate(paths)
    ])

    status, report = _run(
        case, mode=mode, receipt_path=receipt,
    )

    assert status == 0
    assert report["violations"] == []


def test_production_artifact_and_runner_tempfile_forms_are_exact(case):
    output = case["output_dir"]
    receipt = output / "selection.json"
    receipt.write_text("{}", encoding="ascii")
    staging = output.parent / (".{}.staging-ABC123".format(output.name))
    allowed = [
        staging / ".backbone.pth.runner123.tmp",
        staging / ".parent.pth.runner123.tmp",
        staging / ".geometry.pth.runner123.tmp",
        staging / ".selection.json.runner123.tmp",
        staging / "parent.pth.tmp.model123",
        staging / "geometry.pth.tmp.model123",
    ]
    denied = [
        staging / "backbone.pth.tmp.model123",
        staging / "selection.json.tmp.model123",
        staging / "parent.pth.tmp.",
    ]
    _write_trace(case, [
        _openat(
            case["initial_cwd"], str(path),
            "{}<{}>".format(index + 3, path),
            "1710000000.{:06d}".format(index + 1),
        )
        for index, path in enumerate(allowed + denied)
    ])

    status, report = _run(
        case, mode="production", receipt_path=receipt,
    )

    assert status == 1
    assert _codes(report) == ["data_root_allow_miss"] * len(denied)
    assert [item["resolved_path"] for item in report["violations"]] == [
        str(path) for path in denied
    ]


def test_unsafe_staging_file_is_denied_before_allow_miss(case):
    output = case["output_dir"]
    path = output.parent / (".{}.staging-ABC".format(output.name)) / (
        "metrics_validation.json"
    )
    _write_trace(case, [
        _openat(case["initial_cwd"], str(path), "3<{}>".format(path))
    ])

    status, report = _run(case)

    assert status == 1
    assert _codes(report) == ["deny_path"]


def test_unfinished_and_resumed_syscall_is_reconstructed(case):
    denied = case["data_root"] / "val_v3scans.pkl"
    _write_trace(case, [
        '1710000000.000001 openat(AT_FDCWD<{}>, {}, O_RDONLY '
        '<unfinished ...>'.format(case["initial_cwd"], _c_string(denied)),
        '1710000000.100001 <... openat resumed>) = -1 ENOENT (missing)',
    ])

    status, report = _run(case)

    violation = report["violations"][0]
    assert status == 1
    assert violation["entry_line"] == 4
    assert violation["resume_line"] == 5
    assert violation["timestamp"] == "1710000000.000001"


@pytest.mark.parametrize("lines", [
    ['1710000000.000001 <... openat resumed>) = 3</tmp/x>'],
    ['1710000000.000001 openat(AT_FDCWD</tmp>, "x", O_RDONLY '
     '<unfinished ...>'],
])
def test_orphan_resume_and_eof_pending_are_parse_errors(case, lines):
    _write_trace(case, lines)

    status, report = _run(case)

    assert status == 2
    assert report["pass"] is False
    assert "trace_parse_error" in _codes(report)


def test_c_escaped_path_is_decoded_before_deny(case):
    escaped = str(case["data_root"]) + "/ScanRefer_filtered_\\x76al.json"
    resolved = case["data_root"] / "ScanRefer_filtered_val.json"
    _write_trace(case, [
        '1710000000.000001 openat(AT_FDCWD<{}>, "{}", O_RDONLY) = '
        '3<{}>'.format(case["initial_cwd"], escaped, resolved)
    ])

    status, report = _run(case)

    assert status == 1
    assert report["violations"][0]["raw_path"] == str(resolved)


@pytest.mark.parametrize("path_token,result", [
    ("0x7ffee1234000", "-1 EFAULT (Bad address)"),
    ('"/tmp/partial"...', "-1 ENOENT (missing)"),
    ('"/tmp/x"', "? ERESTARTSYS (To be restarted)"),
])
def test_pointer_truncated_path_and_unknown_result_are_parse_errors(
        case, path_token, result):
    _write_trace(case, [
        '1710000000.000001 openat(AT_FDCWD<{}>, {}, O_RDONLY) = {}'.format(
            case["initial_cwd"], path_token, result
        )
    ])

    status, report = _run(case)

    assert status == 2
    assert "trace_parse_error" in _codes(report)


@pytest.mark.parametrize("result,expected", [
    ("-1 EPERM (Operation not permitted)", "uncertain_path"),
    ("9", "uncertain_path"),
    ("9<{denied}>", "deny_path"),
])
def test_open_by_handle_at_fails_closed(case, result, expected):
    denied = case["data_root"] / "official_result.json"
    result = result.format(denied=denied)
    _write_trace(case, [
        '1710000000.000001 open_by_handle_at(5<{}>, '
        '{{handle_bytes=8, handle_type=1, f_handle=0x01}}, O_RDONLY) = {}'.format(
            case["data_root"], result
        )
    ])

    status, report = _run(case)

    assert status == 1
    assert _codes(report) == [expected]


def test_successful_chdir_and_fchdir_are_violations_and_update_cwd(case):
    new_cwd = case["initial_cwd"] / "subdir"
    _write_trace(case, [
        '1710000000.000001 chdir({}) = 0'.format(_c_string(new_cwd)),
        '1710000000.000002 openat(AT_FDCWD<{}>, "val_v3scans.pkl", O_RDONLY) = '
        '-1 ENOENT (missing)'.format(new_cwd),
        '1710000000.000003 fchdir(8<{}>) = 0'.format(case["initial_cwd"]),
    ])

    status, report = _run(case)

    assert status == 1
    assert _codes(report) == ["chdir_success", "deny_path", "fchdir_success"]
    assert report["violations"][1]["candidate_path"] == str(
        new_cwd / "val_v3scans.pkl"
    )


def test_successful_io_uring_setup_is_violation_but_failure_is_not(case):
    _write_trace(case, [
        '1710000000.000001 io_uring_setup(64, {flags=0}) = 9<anon_inode:[io_uring]>',
        '1710000000.000002 io_uring_setup(64, {flags=0}) = -1 EPERM (denied)',
    ])

    status, report = _run(case)

    assert status == 1
    assert _codes(report) == ["io_uring_setup_success"]


def test_execve_and_execveat_receive_path_policy(case):
    _write_trace(case, _standard_exec_lines(case) + [
        '1710000000.000001 execveat(4<{}>, "validation", ["x"], '
        '0x7fff, 0) = -1 ENOENT (missing)'.format(case["data_root"]),
    ], include_exec_contract=False)

    status, report = _run(case)

    assert status == 1
    assert _codes(report) == ["deny_path"]
    assert report["violations"][0]["syscall"] == "execveat"


def test_root_line_one_current_python_exec_is_stably_bound(case):
    lines = _standard_exec_lines(case)
    _write_trace(case, lines + [
        _openat(
            case["initial_cwd"], "safe", "3<{}>".format(
                case["input_artifacts"][0]
            ), "1710000000.000002",
        ),
    ], include_exec_contract=False)

    status, report = _run(case)

    assert status == 0
    interpreter = report["inputs"]["initial_interpreter"]
    assert interpreter["logical_path"] == os.path.abspath(sys.executable)
    assert interpreter["resolved_path"] == os.path.realpath(sys.executable)
    assert interpreter["sha256"] == hashlib.sha256(
        Path(os.path.realpath(sys.executable)).read_bytes()
    ).hexdigest()
    assert report["inputs"]["expected_initial_runner_argv"] == (
        case["expected_initial_runner_argv"]
    )


@pytest.mark.parametrize("mutation", [
    "missing",
    "added",
    "removed",
    "replaced",
    "reordered",
    "argv0",
    "path",
])
def test_initial_runner_exec_argv_and_path_are_exact(case, mutation):
    expected = list(case["expected_initial_runner_argv"])
    actual = list(expected)
    executable = sys.executable
    if mutation == "missing":
        actual = []
    elif mutation == "added":
        actual.append("--unexpected")
    elif mutation == "removed":
        actual.pop()
    elif mutation == "replaced":
        actual[-1] = "production"
    elif mutation == "reordered":
        actual[1], actual[2] = actual[2], actual[1]
    elif mutation == "argv0":
        actual[0] = "python"
    elif mutation == "path":
        other = case["data_root"].parent / "trace-other-python"
        other.write_bytes(b"not the configured interpreter")
        executable = str(other)
    lines = _standard_exec_lines(case, initial_argv=actual)
    lines[0] = '1709999999.000001 execve({}, {}, 0x7fff) = 0'.format(
        _c_string(executable), _c_string_array(actual)
    )
    _write_trace(case, lines, include_exec_contract=False)

    status, report = _run(case)

    assert status == 2
    assert _codes(report) == ["uncertain_path", "trace_parse_error"]
    assert report["violations"][0]["rule"] == (
        "successful-exec-without-authoritative-target"
    )
    assert report["violations"][1]["rule"] == "exec-contract-exactly-once"


def test_initial_exec_rejects_canonical_alias_of_expected_argv0(case):
    expected = case["expected_initial_runner_argv"]
    alias = case["data_root"].parent / "python-command-alias"
    alias.symlink_to(expected[0])
    lines = _standard_exec_lines(case)
    lines[0] = '1709999999.000001 execve({}, {}, 0x7fff) = 0'.format(
        _c_string(alias), _c_string_array(expected)
    )
    _write_trace(case, lines, include_exec_contract=False)

    status, report = _run(case)

    assert status == 2
    assert _codes(report) == ["uncertain_path", "trace_parse_error"]
    assert report["violations"][0]["rule"] == (
        "successful-exec-without-authoritative-target"
    )
    assert report["violations"][1]["rule"] == "exec-contract-exactly-once"


def test_duplicate_initial_exec_fails_closed(case):
    lines = _standard_exec_lines(case)
    duplicate = lines[0].replace(
        "1709999999.000001", "1709999999.000002"
    )
    lines.insert(1, duplicate)
    _write_trace(case, lines, include_exec_contract=False)

    status, report = _run(case)

    assert status == 1
    assert _codes(report) == ["uncertain_path"]
    assert report["violations"][0]["entry_line"] == 2
    assert report["violations"][0]["rule"] == (
        "successful-exec-without-authoritative-target"
    )


def test_initial_exec_moved_from_root_line_one_is_fatal(case):
    standard = _standard_exec_lines(case)
    python_line = standard[0].replace(
        "1709999999.000001", "1709999999.000002"
    )
    shell_line = standard[1].replace(
        "1709999999.000002", "1709999999.000001"
    )
    _write_trace(
        case, [shell_line, python_line, standard[2]],
        include_exec_contract=False,
    )

    status, report = _run(case)

    assert status == 2
    assert _codes(report) == ["uncertain_path", "trace_parse_error"]
    assert report["violations"][0]["entry_line"] == 2
    assert report["violations"][1]["rule"] == "exec-contract-exactly-once"


def test_later_symlink_and_proc_fd_successful_execs_fail_closed(case):
    denied_target = case["data_root"] / "val_v3scans.pkl"
    denied_target.write_bytes(b"not really executable")
    safe_looking = case["data_root"].parent / "safe-python"
    safe_looking.symlink_to(denied_target)
    _write_trace(case, _standard_exec_lines(case) + [
        '1710000000.000002 execve({}, ["safe-python"], 0x7fff) = 0'.format(
            _c_string(safe_looking)
        ),
        '1710000000.000003 execve("/proc/self/fd/9", ["python"], '
        '0x7fff) = 0',
    ], include_exec_contract=False)

    status, report = _run(case)

    assert status == 1
    assert _codes(report) == ["uncertain_path", "uncertain_path"]
    assert all(item["resolved_path"] is None for item in report["violations"])
    assert all(item["rule"] == "successful-exec-without-authoritative-target"
               for item in report["violations"])


def test_successful_exec_in_non_root_pid_is_not_initial_interpreter(case):
    _write_trace(case, [
        _openat(case["initial_cwd"], "safe", "3<{}>".format(
            case["input_artifacts"][0]
        ))
    ], pid=101)
    _write_trace(case, [
        '1710000000.000002 execve({}, ["python"], 0x7fff) = 0'.format(
            _c_string(sys.executable)
        )
    ], pid=102, include_exec_contract=False)

    status, report = _run(case)

    assert status == 1
    assert _codes(report) == ["uncertain_path"]


def test_exact_platform_probe_execs_are_stably_bound_and_allowed(case):
    _write_trace(
        case, _standard_exec_lines(case), include_exec_contract=False
    )

    status, report = _run(case)

    assert status == 0
    probes = report["inputs"]["platform_probe_executables"]
    assert [probe["logical_path"] for probe in probes] == [
        "/bin/sh", "/usr/bin/uname"
    ]
    assert [probe["argv"] for probe in probes] == [
        ["/bin/sh", "-c", "uname -p 2> /dev/null"],
        ["uname", "-p"],
    ]
    for probe in probes:
        resolved = Path(probe["resolved_path"])
        assert resolved.is_file()
        assert probe["sha256"] == hashlib.sha256(resolved.read_bytes()).hexdigest()
        assert probe["size"] == resolved.stat().st_size


def test_platform_probe_exec_deviations_and_repeats_fail_closed(case):
    _write_trace(case, [
        '1710000000.000001 execve({}, {}, 0x7fff) = 0'.format(
            _c_string(sys.executable),
            _c_string_array(case["expected_initial_runner_argv"]),
        ),
        '1710000000.000002 execve("/bin/sh", ["/bin/sh", "-c", '
        '"uname -p 2> /dev/null"], 0x7fff) = 0',
        '1710000000.000003 execve("/bin/sh", ["/bin/sh", "-c", '
        '"uname -p 2> /dev/null"], 0x7fff) = 0',
        '1710000000.000004 execve("/bin/sh", ["/bin/sh", "-c", '
        '"uname -m 2> /dev/null"], 0x7fff) = 0',
        '1710000000.000005 execve("/usr/bin/uname", ["uname", "-p"], '
        '0x7fff) = 0',
        '1710000000.000006 execve("/usr/bin/uname", ["uname", "-p"], '
        '0x7fff) = 0',
        '1710000000.000007 execve("/bin/uname", ["uname", "-p"], '
        '0x7fff) = 0',
        '1710000000.000008 execve("/usr/bin/uname", '
        '["uname", "-p", "extra"], 0x7fff) = 0',
    ], include_exec_contract=False)

    status, report = _run(case)

    assert status == 2
    assert _codes(report) == (
        ["uncertain_path"] * 5 + ["trace_parse_error"]
    )
    assert all(item["resolved_path"] is None for item in report["violations"])
    assert all(
        item["rule"] == "successful-exec-without-authoritative-target"
        for item in report["violations"][:5]
    )
    assert report["violations"][-1]["rule"] == "exec-contract-exactly-once"


@pytest.mark.parametrize("missing_index,missing_name", [
    (0, "initial Python interpreter"),
    (1, "/bin/sh platform probe"),
    (2, "/usr/bin/uname platform probe"),
])
def test_missing_required_exec_contract_is_fatal(
        case, missing_index, missing_name):
    lines = _standard_exec_lines(case)
    del lines[missing_index]
    lines.append(_openat(
        case["initial_cwd"], "safe", "3<{}>".format(
            case["input_artifacts"][0]
        ), "1710000000.000001",
    ))
    _write_trace(case, lines, include_exec_contract=False)

    status, report = _run(case)

    assert status == 2
    assert _codes(report) == ["trace_parse_error"]
    violation = report["violations"][0]
    assert violation["rule"] == "exec-contract-exactly-once"
    assert missing_name in violation["detail"]


def test_false_positive_names_are_not_denied(case):
    paths = [
        "/tmp/final_contract.json",
        "/tmp/value.json",
        "/usr/lib/python3.7/site-packages/pkg/validation.py",
    ]
    _write_trace(case, [
        _openat(case["initial_cwd"], path, "{}<{}>".format(index + 3, path),
                "1710000000.{:06d}".format(index + 1))
        for index, path in enumerate(paths)
    ])

    status, report = _run(case)

    assert status == 0
    assert report["violations"] == []


def test_generic_output_tokens_are_scoped_away_from_dependency_pyc(case):
    safe_paths = [
        "/usr/lib/python3.7/site-packages/pkg/__pycache__/"
        "validation.cpython-37.pyc",
        "/usr/lib/python3.7/site-packages/scipy/stats/__pycache__/"
        "_validation.cpython-37.pyc",
    ]
    denied = (
        case["data_root"] / "private" / "validation.cpython-37.pyc"
    )
    _write_trace(case, [
        _openat(
            case["initial_cwd"], path,
            "{}<{}>".format(index + 3, path),
            "1710000000.{:06d}".format(index + 1),
        )
        for index, path in enumerate(safe_paths)
    ] + [
        _openat(
            case["initial_cwd"], str(denied), "9<{}>".format(denied),
            "1710000000.000003",
        )
    ])

    status, report = _run(case)

    assert status == 1
    assert _codes(report) == ["deny_path"]
    assert report["violations"][0]["resolved_path"] == str(denied)
    assert report["violations"][0]["rule"] == (
        "independent-output-token:validation"
    )


def test_data_root_with_symlink_ancestor_is_configuration_error(case):
    alias = case["data_root"].parent / "alias"
    alias.symlink_to(case["data_root"].parent, target_is_directory=True)
    aliased_root = alias / case["data_root"].name
    aliased_output = alias / "DATA_ROOT" / "published" / "rec_run"
    aliased_inputs = [
        alias / "DATA_ROOT" / "initial" / path.name
        for path in case["input_artifacts"]
    ]
    _write_trace(case, [
        _openat(
            alias / "repo", "safe", "3<{}>".format(case["input_artifacts"][0])
        )
    ])

    status, report = _run(
        case,
        data_root=aliased_root,
        initial_cwd=alias / "repo",
        input_artifacts=aliased_inputs,
        output_dir=aliased_output,
        receipt_path=aliased_output / "smoke-receipt.json",
    )

    assert status == 2
    assert _codes(report) == ["configuration_error"]
    assert "symlink" in report["violations"][0]["detail"]


def test_success_candidate_inside_data_root_cannot_escape_via_symlink(case):
    outside = case["data_root"].parent / "outside"
    outside.mkdir()
    secret = outside / "secret.bin"
    secret.write_bytes(b"secret")
    link = case["data_root"] / "training-cache"
    link.symlink_to(outside, target_is_directory=True)
    candidate = link / "secret.bin"
    _write_trace(case, [
        _openat(
            case["initial_cwd"], str(candidate), "3<{}>".format(secret)
        )
    ])

    status, report = _run(case)

    assert status == 1
    assert _codes(report) == ["data_root_allow_miss"]
    assert report["violations"][0]["candidate_path"] == str(candidate)
    assert report["violations"][0]["resolved_path"] == str(secret)


def test_allowlisted_candidate_cannot_resolve_outside_data_root(case):
    outside = case["data_root"].parent / "outside-allowlisted"
    outside.mkdir()
    secret = outside / "secret.bin"
    secret.write_bytes(b"secret")
    candidate = case["data_root"] / "train_v3scans.pkl"
    candidate.symlink_to(secret)
    _write_trace(case, [
        _openat(
            case["initial_cwd"], str(candidate), "3<{}>".format(secret)
        )
    ])

    status, report = _run(case)

    assert status == 1
    assert _codes(report) == ["data_root_allow_miss"]
    violation = report["violations"][0]
    assert violation["candidate_path"] == str(candidate)
    assert violation["resolved_path"] == str(secret)
    assert violation["rule"] == "data-root-resolution-membership"


def test_annotated_numeric_dirfd_path_may_contain_commas(case):
    base = case["data_root"] / "dir,with,commas"
    raw = "../superpoints/val/scene0001_02_superpoint.pth"
    _write_trace(case, [
        '1710000000.000001 openat(7<{}>, {}, O_RDONLY) = '
        '-1 ENOENT (missing)'.format(base, _c_string(raw))
    ])

    status, report = _run(case)

    assert status == 1
    assert _codes(report) == ["deny_path"]
    assert report["violations"][0]["candidate_path"] == str(
        case["data_root"] / "superpoints" / "val" /
        "scene0001_02_superpoint.pth"
    )


def test_violations_and_trace_inputs_are_deterministically_sorted(case):
    path_a = case["data_root"] / "official_result.json"
    path_b = case["data_root"] / "val_v3scans.pkl"
    _write_trace(case, [
        _openat(case["initial_cwd"], str(path_a), "3<{}>".format(path_a),
                "1710000000.000001")
    ], pid=3)
    _write_trace(case, [
        _openat(case["initial_cwd"], str(path_b), "4<{}>".format(path_b),
                "1710000000.000002")
    ], pid=20, include_exec_contract=False)

    status, first = _run(case)
    first_bytes = case["report_path"].read_bytes()
    second_report_path = case["report_path"].with_name("audit-second.json")
    status_again, second = _run(case, report_path=second_report_path)

    assert status == status_again == 1
    assert first == second
    assert first_bytes == second_report_path.read_bytes()
    assert [item["pid"] for item in first["inputs"]["traces"]] == [3, 20]
    assert [item["pid"] for item in first["violations"]] == [3, 20]


def test_runner_exit_and_receipt_gates_are_reported(case):
    _write_trace(case, [
        _openat(case["initial_cwd"], "safe", "3<{}>".format(
            case["input_artifacts"][0]
        ))
    ])
    case["runner_exit_code_file"].write_text("17\n", encoding="ascii")
    case["receipt_path"].unlink()

    status, report = _run(case)

    assert status == 1
    assert _codes(report) == ["receipt_missing", "runner_exit_nonzero"]
    assert all(VIOLATION_FIELDS.issubset(item) for item in report["violations"])


def test_receipt_hash_is_bound_into_report(case):
    _write_trace(case, [
        _openat(case["initial_cwd"], "safe", "3<{}>".format(
            case["input_artifacts"][0]
        ))
    ])

    status, report = _run(case)

    assert status == 0
    assert report["inputs"]["receipt"]["path"] == str(case["receipt_path"])
    assert report["inputs"]["receipt"]["sha256"] == hashlib.sha256(
        case["receipt_path"].read_bytes()
    ).hexdigest()


def test_empty_thread_traces_are_allowed_but_all_empty_is_parse_error(case):
    _write_trace(case, [
        _openat(case["initial_cwd"], "safe", "3<{}>".format(
            case["input_artifacts"][0]
        ))
    ], pid=1)
    _write_trace(case, [], pid=2, include_exec_contract=False)

    status, report = _run(case)
    assert status == 0
    assert report["counts"]["empty_trace_files"] == 1

    Path(str(case["trace_prefix"]) + ".1").write_bytes(b"")
    status, report = _run(
        case,
        report_path=case["report_path"].with_name("all-empty-audit.json"),
    )
    assert status == 2
    assert "trace_parse_error" in _codes(report)


def test_timestamped_signal_records_are_ignored(case):
    _write_trace(case, [
        _openat(case["initial_cwd"], "safe", "3<{}>".format(
            case["input_artifacts"][0]
        )),
        "1710000000.100001 --- SIGCHLD {si_signo=SIGCHLD, "
        "si_code=CLD_EXITED, si_status=0} ---",
    ])

    status, report = _run(case)

    assert status == 0
    assert report["counts"]["syscalls"] == 4


def test_trace_symlink_is_integrity_error(case):
    target = case["trace_prefix"].with_name("real-trace")
    target.write_text(
        _openat(case["initial_cwd"], "safe", "3<{}>".format(
            case["input_artifacts"][0]
        )) + "\n",
        encoding="ascii",
    )
    os.symlink(str(target), str(case["trace_prefix"]) + ".7")

    status, report = _run(case)

    assert status == 2
    assert _codes(report) == ["trace_integrity_error"]


def _cli_command(case, report, style="primary", expected_argv_json=None):
    if expected_argv_json is None:
        expected_argv_json = json.dumps(
            case["expected_initial_runner_argv"], separators=(",", ":")
        )
    use_aliases = style != "primary"
    receipt_option = {
        "primary": "--receipt-path",
        "aliases-selection": "--selection-path",
        "aliases-receipt": "--receipt-or-selection-path",
    }[style]
    return [
        sys.executable,
        str(Path(__file__).parents[1] / "scripts" /
            "audit_rec_finetune_file_access.py"),
        "--trace-prefix", str(case["trace_prefix"]),
        "--data-root", str(case["data_root"]),
        "--initial-cwd", str(case["initial_cwd"]),
        "--input-backbone" if use_aliases else "--backbone-checkpoint",
        str(case["input_artifacts"][0]),
        "--input-parent" if use_aliases else "--parent-reranker",
        str(case["input_artifacts"][1]),
        "--input-geometry" if use_aliases else "--geometry-reranker",
        str(case["input_artifacts"][2]),
        "--output-dir", str(case["output_dir"]),
        "--mode", "smoke",
        "--runner-exit-code-file", str(case["runner_exit_code_file"]),
        receipt_option, str(case["receipt_path"]),
        ("--expected-initial-runner-argv" if use_aliases
         else "--expected-initial-runner-argv-json"),
        expected_argv_json,
        "--report-path", str(report),
    ]


@pytest.mark.parametrize("style", [
    "primary", "aliases-selection", "aliases-receipt",
])
def test_cli_accepts_complete_contract_and_returns_audit_status(case, style):
    _write_trace(case, [
        _openat(case["initial_cwd"], "safe", "3<{}>".format(
            case["input_artifacts"][0]
        ))
    ])
    cli_report = case["report_path"].with_name(
        "cli-report-{}.json".format(style)
    )
    command = _cli_command(case, cli_report, style=style)

    completed = subprocess.run(command, stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, text=True)

    assert completed.returncode == 0, completed.stderr
    assert json.loads(cli_report.read_text("ascii"))["pass"] is True


@pytest.mark.parametrize("style", ["primary", "aliases-selection"])
def test_cli_expected_argv_primary_and_alias_reject_invalid_json(case, style):
    report = case["report_path"].with_name("invalid-cli-{}.json".format(style))
    completed = subprocess.run(
        _cli_command(case, report, style=style, expected_argv_json="not-json"),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    assert completed.returncode == 2
    assert "expected runner argv must be JSON" in completed.stderr
    assert not report.exists()


def test_source_gate_cli_requires_and_binds_runtime_scratch(source_gate_case):
    case = source_gate_case
    _write_trace(case, _source_gate_success_lines(case))
    missing_report = case["report_path"].with_name(
        "source-gate-cli-missing-scratch.json"
    )
    missing = subprocess.run(
        _cli_command(case, missing_report),
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )

    assert missing.returncode == 2
    assert _codes(json.loads(missing_report.read_text("ascii"))) == [
        "configuration_error"
    ]

    report = case["report_path"].with_name("source-gate-cli.json")
    command = _cli_command(case, report) + [
        "--runtime-scratch-dir", str(case["runtime_scratch_dir"]),
    ]
    completed = subprocess.run(
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(report.read_text("ascii"))
    assert payload["pass"] is True
    assert payload["inputs"]["runtime_scratch_dir"] == str(
        case["runtime_scratch_dir"]
    )


@pytest.fixture
def residual_case(case, monkeypatch):
    base_cache = case["data_root"] / "rec_reranker" / "e71_top16" / "train"
    geometry_cache = (
        case["data_root"] / "rec_reranker" / "e71_top16" / "geometry_train"
    )
    base_cache.mkdir(parents=True)
    geometry_cache.mkdir(parents=True)
    (base_cache / "manifest.json").write_text('{"split":"train"}', encoding="ascii")
    (geometry_cache / "manifest.json").write_text(
        '{"split":"train"}', encoding="ascii"
    )
    output_dir = case["data_root"].parent / "residual-experiment"
    output_dir.mkdir()
    receipt = output_dir / "result-receipt.json"
    receipt.write_text(
        '{"selected":"baseline","validation_data_accessed":false}',
        encoding="ascii",
    )
    receipt.chmod(0o444)
    runner = (
        Path(__file__).parents[1] / "scripts"
        / "train_scanrefer_rec_selective_residual.py"
    )
    expected_sha256 = {}
    for label, path in zip(
            ("backbone", "parent", "geometry"), case["input_artifacts"]):
        path.chmod(0o444)
        expected_sha256[label] = hashlib.sha256(path.read_bytes()).hexdigest()
    monkeypatch.setattr(
        access_audit,
        "AUTHORITATIVE_RESIDUAL_ARTIFACT_SHA256",
        expected_sha256,
        raising=False,
    )
    case.update({
        "mode": "residual",
        "base_cache": base_cache,
        "geometry_cache": geometry_cache,
        "output_dir": output_dir,
        "receipt_path": receipt,
        "expected_initial_runner_argv": [
            os.path.abspath(sys.executable),
            str(runner),
            "--base-cache", str(base_cache),
            "--geometry-cache", str(geometry_cache),
            "--parent-artifact", str(case["input_artifacts"][1]),
            "--geometry-artifact", str(case["input_artifacts"][2]),
            "--output-dir", str(output_dir),
            "--device", "cuda:0",
        ],
    })
    return case


def _residual_success_lines(case):
    reads = [
        case["expected_initial_runner_argv"][1],
        case["base_cache"] / "manifest.json",
        case["geometry_cache"] / "manifest.json",
    ] + list(case["input_artifacts"])
    lines = [
        _openat(
            case["initial_cwd"], path,
            "{}<{}>".format(index + 3, path),
            "1710000100.{:06d}".format(index + 1),
        )
        for index, path in enumerate(reads)
    ]
    pending_receipt = case["output_dir"] / ".result-receipt.json.pending"
    lines.extend([
        _openat(
            case["initial_cwd"], pending_receipt,
            "20<{}>".format(pending_receipt),
            "1710000101.000001", "O_WRONLY|O_CREAT|O_EXCL|O_CLOEXEC",
        ),
        "1710000101.000002 link({}, {}) = 0".format(
            _c_string(pending_receipt), _c_string(case["receipt_path"])
        ),
        "1710000101.000003 unlink({}) = 0".format(
            _c_string(pending_receipt)
        ),
        _openat(
            case["initial_cwd"], case["receipt_path"],
            "21<{}>".format(case["receipt_path"]), "1710000101.000004",
        ),
    ])
    return lines


def test_residual_profile_uses_explicit_non_percent_file_selector():
    assert access_audit.STRACE_FILE_ACCESS_SELECTOR == (
        "open,openat,creat,open_by_handle_at,chdir,fchdir,execve,execveat,"
        "io_uring_setup,rename,renameat,renameat2,unlink,unlinkat,link,"
        "linkat,symlink,symlinkat,chmod,fchmodat,truncate,437"
    )
    assert "%file" not in access_audit.STRACE_FILE_ACCESS_SELECTOR


def test_residual_audit_allows_only_bound_train_caches_and_records_digest(
        residual_case):
    case = residual_case
    _write_trace(case, _residual_success_lines(case))

    status, report = _run(case)

    assert status == 0
    assert report["pass"] is True
    assert report["validation_data_accessed"] is False
    assert len(report["opened_path_sha256"]) == 64
    assert report["opened_path_count"] >= 8
    assert report["inputs"]["receipt"]["validation_data_accessed"] is False
    assert report["inputs"]["base_cache"] == str(case["base_cache"].resolve())
    assert report["inputs"]["geometry_cache"] == str(
        case["geometry_cache"].resolve()
    )
    assert report["destructive_path_call_count"] == 2
    assert len(report["destructive_path_call_sha256"]) == 64
    assert stat.S_IMODE(case["report_path"].stat().st_mode) == 0o444


@pytest.mark.parametrize("location", [
    "backbone", "parent", "geometry", "base_cache", "geometry_cache",
    "output_dir", "trace_prefix", "runner_exit", "existing_report",
])
def test_residual_rejects_unsafe_report_destination_without_modification(
        residual_case, location):
    case = residual_case
    _write_trace(case, _residual_success_lines(case))
    targets = {
        "backbone": case["input_artifacts"][0],
        "parent": case["input_artifacts"][1],
        "geometry": case["input_artifacts"][2],
        "base_cache": case["base_cache"],
        "geometry_cache": case["geometry_cache"],
        "output_dir": case["output_dir"],
        "trace_prefix": case["trace_prefix"],
        "runner_exit": case["runner_exit_code_file"],
        "existing_report": case["report_path"],
    }
    target = Path(targets[location])
    if location == "existing_report":
        target.write_bytes(b"existing audit report")
        target.chmod(0o444)
    case["report_path"] = target
    snapshot_root = case["data_root"].parent
    before = _tree_snapshot(snapshot_root)

    status = audit(**case)

    assert status == 2
    assert _tree_snapshot(snapshot_root) == before


def test_audit_report_publication_never_overwrites_racing_target(
        monkeypatch, case):
    _write_trace(case, [
        _openat(
            case["initial_cwd"], case["input_artifacts"][0],
            "3<{}>".format(case["input_artifacts"][0]),
        )
    ])
    original = access_audit._normalize_config
    racing_bytes = b"racing audit target"

    def normalize(*args, **kwargs):
        config = original(*args, **kwargs)
        Path(config["report_path"]).write_bytes(racing_bytes)
        return config

    monkeypatch.setattr(access_audit, "_normalize_config", normalize)

    with pytest.raises(FileExistsError):
        audit(**case)

    assert case["report_path"].read_bytes() == racing_bytes


@pytest.mark.parametrize("artifact_index,label", [
    (0, "backbone"), (1, "parent"), (2, "geometry"),
])
@pytest.mark.parametrize("mutation", ["sha256", "mode"])
def test_residual_rejects_non_authoritative_protected_artifact(
        residual_case, artifact_index, label, mutation):
    case = residual_case
    artifact = case["input_artifacts"][artifact_index]
    expected_sha256 = access_audit.AUTHORITATIVE_RESIDUAL_ARTIFACT_SHA256[
        label
    ]
    if mutation == "sha256":
        artifact.chmod(0o644)
        artifact.write_bytes(b"replaced protected artifact")
        artifact.chmod(0o444)
    else:
        artifact.chmod(0o644)
    _write_trace(case, _residual_success_lines(case))

    status, report = _run(case)

    assert status == 2
    assert "protected_artifact_mismatch" in _codes(report)
    record = next(
        item for item in report["inputs"]["input_artifacts"]
        if item["label"] == label
    )
    assert record["expected_sha256"] == expected_sha256
    assert record["expected_mode"] == 0o444
    assert record["sha256_matches"] is (mutation != "sha256")
    assert record["mode_matches"] is (mutation != "mode")


def _destructive_syscall_line(
    syscall, protected, output, protected_as_destination=False):
    protected = str(protected)
    safe_path = str(Path(output) / ".result-receipt.json.pending")
    source = safe_path if protected_as_destination else protected
    destination = protected if protected_as_destination else safe_path
    cwd = str(Path(output).parent)
    if syscall == "rename":
        arguments = (_c_string(source), _c_string(destination))
    elif syscall in ("renameat", "renameat2"):
        arguments = (
            "AT_FDCWD<{}>".format(cwd), _c_string(source),
            "AT_FDCWD<{}>".format(cwd), _c_string(destination),
        )
        if syscall == "renameat2":
            arguments += ("RENAME_NOREPLACE",)
    elif syscall == "unlink":
        arguments = (_c_string(protected),)
    elif syscall == "unlinkat":
        arguments = (
            "AT_FDCWD<{}>".format(cwd), _c_string(protected), "0",
        )
    elif syscall == "link":
        arguments = (_c_string(source), _c_string(destination))
    elif syscall == "linkat":
        arguments = (
            "AT_FDCWD<{}>".format(cwd), _c_string(source),
            "AT_FDCWD<{}>".format(cwd), _c_string(destination), "0",
        )
    elif syscall == "symlink":
        arguments = (_c_string("source"), _c_string(protected))
    elif syscall == "symlinkat":
        arguments = (
            _c_string("source"), "AT_FDCWD<{}>".format(cwd),
            _c_string(protected),
        )
    elif syscall == "chmod":
        arguments = (_c_string(protected), "0600")
    elif syscall == "fchmodat":
        arguments = (
            "AT_FDCWD<{}>".format(cwd), _c_string(protected), "0600", "0",
        )
    elif syscall == "truncate":
        arguments = (_c_string(protected), "0")
    else:
        raise AssertionError("unsupported test syscall")
    return "1710000104.000001 {}({}) = 0".format(
        syscall, ", ".join(arguments)
    )


@pytest.mark.parametrize("artifact_index", [0, 1, 2])
@pytest.mark.parametrize("syscall", [
    "rename", "renameat", "renameat2", "unlink", "unlinkat", "link",
    "linkat", "symlink", "symlinkat", "chmod", "fchmodat", "truncate",
])
def test_residual_rejects_successful_destructive_syscall_on_protected_artifact(
        residual_case, artifact_index, syscall):
    case = residual_case
    protected = case["input_artifacts"][artifact_index]
    lines = _residual_success_lines(case) + [
        _destructive_syscall_line(syscall, protected, case["output_dir"])
    ]
    _write_trace(case, lines)

    status, report = _run(case)

    assert status == 1
    violations = [
        item for item in report["violations"]
        if item["code"] == "protected_path_mutation"
    ]
    assert len(violations) == 1
    assert violations[0]["syscall"] == syscall
    assert violations[0]["success"] is True
    assert violations[0]["candidate_path"] == str(protected)


@pytest.mark.parametrize("syscall", [
    "rename", "renameat", "renameat2", "link", "linkat",
])
def test_residual_checks_destination_of_two_path_destructive_syscall(
        residual_case, syscall):
    case = residual_case
    protected = case["input_artifacts"][1]
    line = _destructive_syscall_line(
        syscall,
        protected,
        case["output_dir"],
        protected_as_destination=True,
    )
    _write_trace(case, _residual_success_lines(case) + [line])

    status, report = _run(case)

    assert status == 1
    violation = next(
        item for item in report["violations"]
        if item["code"] == "protected_path_mutation"
    )
    assert violation["candidate_path"] == str(protected)


@pytest.mark.parametrize("syscall", [
    "rename", "renameat", "renameat2", "unlink", "unlinkat", "link",
    "linkat", "symlink", "symlinkat", "chmod", "fchmodat", "truncate",
])
def test_failed_destructive_syscall_is_diagnostic_not_a_mutation(
        residual_case, syscall):
    case = residual_case
    protected = case["input_artifacts"][0]
    failed_line = _destructive_syscall_line(
        syscall, protected, case["output_dir"]
    ).replace(" = 0", " = -1 EPERM (Operation not permitted)")
    _write_trace(case, _residual_success_lines(case) + [failed_line])

    status, report = _run(case)

    assert status == 0
    assert report["pass"] is True
    assert report["destructive_path_call_count"] == 3
    assert "protected_path_mutation" not in _codes(report)


def test_residual_allows_destructive_syscalls_within_bound_scratch(
        residual_case):
    case = residual_case
    scratch = case["data_root"].parent / "bound-residual-scratch"
    scratch.mkdir(mode=0o700)
    scratch.chmod(0o700)
    case["runtime_scratch_dir"] = scratch
    source = scratch / "source.tmp"
    destination = scratch / "destination.tmp"
    lines = _residual_success_lines(case) + [
        "1710000105.000001 rename({}, {}) = 0".format(
            _c_string(source), _c_string(destination)
        ),
        "1710000105.000002 chmod({}, 0600) = 0".format(
            _c_string(destination)
        ),
        "1710000105.000003 unlink({}) = 0".format(
            _c_string(destination)
        ),
    ]
    _write_trace(case, lines)

    status, report = _run(case)

    assert status == 0
    assert report["pass"] is True
    assert report["destructive_path_call_count"] == 5


@pytest.mark.parametrize("blocked", [
    "val/annotation.json",
    "geometry_val/manifest.json",
    "official_result.json",
    "prior.claim.json",
    "prior.receipt.json",
])
def test_residual_audit_rejects_validation_official_claim_and_prior_receipt(
        residual_case, blocked):
    case = residual_case
    denied = case["data_root"] / blocked
    lines = _residual_success_lines(case)
    lines.append(_openat(
        case["initial_cwd"], denied, "31<{}>".format(denied),
        "1710000102.000001",
    ))
    _write_trace(case, lines)

    status, report = _run(case)

    assert status == 1
    assert report["pass"] is False
    assert "deny_path" in _codes(report)
    assert report["validation_data_accessed"] is True


def test_residual_audit_rejects_receipt_claiming_validation_access(
        residual_case):
    case = residual_case
    case["receipt_path"].chmod(0o644)
    case["receipt_path"].write_text(
        '{"selected":"baseline","validation_data_accessed":true}',
        encoding="ascii",
    )
    case["receipt_path"].chmod(0o444)
    _write_trace(case, _residual_success_lines(case))

    status, report = _run(case)

    assert status == 1
    assert "receipt_invalid" in _codes(report)


def test_residual_audit_allows_only_pid_bound_gpu_runtime_writes(residual_case):
    case = residual_case
    scratch = case["data_root"].parent / "residual-runtime-scratch"
    scratch.mkdir(mode=0o700)
    scratch.chmod(0o700)
    case["runtime_scratch_dir"] = scratch
    runtime_lines = [
        _openat(
            case["initial_cwd"], "/dev/null", "40</dev/null<char 1:3>>",
            "1710000103.000001", "O_RDWR|O_CLOEXEC",
        ),
        _openat(
            case["initial_cwd"], "/dev/nvidia0",
            "41</dev/nvidia0<char 195:0>>", "1710000103.000002",
            "O_RDWR|O_CLOEXEC",
        ),
        _openat(
            case["initial_cwd"], "/proc/self/task/101/comm",
            "42</proc/101/task/101/comm>", "1710000103.000003",
            "O_WRONLY|O_CLOEXEC",
        ),
        _openat(
            case["initial_cwd"], "/dev/shm/torch_101_42",
            "43</dev/shm/torch_101_42>", "1710000103.000004",
            "O_RDWR|O_CREAT|O_EXCL|O_CLOEXEC",
        ),
        _openat(
            case["initial_cwd"], scratch / "tmp123",
            "44<{}>".format(scratch / "tmp123"),
            "1710000103.000005",
            "O_RDWR|O_CREAT|O_EXCL|O_CLOEXEC",
        ),
    ]
    _write_trace(case, _residual_success_lines(case) + runtime_lines)

    status, report = _run(case)

    assert status == 0
    assert report["pass"] is True


@pytest.fixture
def hierarchical_case(residual_case):
    case = residual_case
    case["mode"] = "hierarchical"
    case["receipt_path"].chmod(0o644)
    case["receipt_path"].write_text(
        json.dumps({
            "schema": "rec-hierarchical-result-receipt-v1",
            "selected": "baseline",
            "validation_data_accessed": False,
        }),
        encoding="ascii",
    )
    case["receipt_path"].chmod(0o444)
    runner = (
        Path(__file__).parents[1] / "scripts"
        / "train_scanrefer_rec_hierarchical_reranker.py"
    )
    case["expected_initial_runner_argv"] = [
        os.path.abspath(sys.executable),
        str(runner),
        "--base-cache", str(case["base_cache"]),
        "--geometry-cache", str(case["geometry_cache"]),
        "--parent-artifact", str(case["input_artifacts"][1]),
        "--geometry-artifact", str(case["input_artifacts"][2]),
        "--output-dir", str(case["output_dir"]),
        "--device", "cuda:0",
    ]
    return case


def test_hierarchical_training_argv_is_exact_and_train_only(
        hierarchical_case):
    case = hierarchical_case

    argv = access_audit.build_hierarchical_training_argv(
        sys.executable,
        case["base_cache"],
        case["geometry_cache"],
        case["input_artifacts"][1],
        case["input_artifacts"][2],
        case["output_dir"],
    )

    assert argv == case["expected_initial_runner_argv"]
    assert argv[-2:] == ["--device", "cuda:0"]
    assert not any(
        token in argument.lower()
        for argument in argv
        for token in ("validation", "official", "claim")
    )


def test_hierarchical_audit_accepts_bound_train_inputs_and_publication(
        hierarchical_case):
    case = hierarchical_case
    staged = case["output_dir"] / "selected_hierarchical.pth"
    staged.write_bytes(b"staged hierarchy")
    staged.chmod(0o444)
    lines = _residual_success_lines(case) + [
        _openat(
            case["initial_cwd"],
            staged,
            "30<{}>".format(staged),
            "1710000101.000005",
            "O_WRONLY|O_CREAT|O_EXCL|O_CLOEXEC",
        )
    ]
    _write_trace(case, lines)

    status, report = _run(case)

    assert status == 0
    assert report["pass"] is True
    assert report["validation_data_accessed"] is False
    assert report["inputs"]["receipt"][
        "validation_data_accessed"
    ] is False
    assert report["inputs"]["base_cache"] == str(
        case["base_cache"].resolve()
    )
    assert report["inputs"]["geometry_cache"] == str(
        case["geometry_cache"].resolve()
    )
    assert all(
        item["mode_matches"] and item["sha256_matches"]
        for item in report["inputs"]["input_artifacts"]
    )


@pytest.mark.parametrize("blocked", (
    "val/annotation.json",
    "official_result.json",
    "prior.claim.json",
    "prior.receipt.json",
))
def test_hierarchical_audit_rejects_validation_and_prior_evidence(
        hierarchical_case, blocked):
    case = hierarchical_case
    denied = case["data_root"] / blocked
    lines = _residual_success_lines(case) + [
        _openat(
            case["initial_cwd"],
            denied,
            "31<{}>".format(denied),
            "1710000102.000001",
        )
    ]
    _write_trace(case, lines)

    status, report = _run(case)

    assert status == 1
    assert report["pass"] is False
    assert "deny_path" in _codes(report)
    assert report["validation_data_accessed"] is True


def test_hierarchical_audit_rejects_command_drift(hierarchical_case):
    case = hierarchical_case
    case["expected_initial_runner_argv"] += ["--hidden-dim", "64"]
    _write_trace(case, _residual_success_lines(case))

    status, report = _run(case)

    assert status == 2
    assert _codes(report) == ["configuration_error"]


def test_hierarchical_audit_rejects_protected_artifact_mutation(
        hierarchical_case):
    case = hierarchical_case
    protected = case["input_artifacts"][2]
    lines = _residual_success_lines(case) + [
        _destructive_syscall_line(
            "unlink", protected, case["output_dir"]
        )
    ]
    _write_trace(case, lines)

    status, report = _run(case)

    assert status == 1
    violation = next(
        item for item in report["violations"]
        if item["code"] == "protected_path_mutation"
    )
    assert violation["candidate_path"] == str(protected)
