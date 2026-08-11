#!/usr/bin/env python3
"""Fail-closed audit of REC finetuning file access captured by strace."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import sys


SCHEMA = "rec-finetune-file-access-audit-v1"
AUTHORITATIVE_RESIDUAL_ARTIFACT_SHA256 = {
    "backbone": (
        "3e44f4bdad3bd66ad82102032e1cb0241de57d147c0aa1d3eff9736926ef2208"
    ),
    "parent": (
        "f06f8972fdcfbbdcb799df267864ab2ebc9ca8403ff92576e2bbdb0a8c17269b"
    ),
    "geometry": (
        "835c25be4717dfcbb324e0c4c5b9d1d3f3e2b90a4dbcb4d4ebe79f215f263b6f"
    ),
}
# This selector is intentionally explicit.  ``%file`` also follows unrelated
# kernel activity and cannot prove that the residual trainer stayed train-only.
STRACE_FILE_ACCESS_SELECTOR = (
    "open,openat,creat,open_by_handle_at,chdir,fchdir,execve,execveat,"
    "io_uring_setup,rename,renameat,renameat2,unlink,unlinkat,link,"
    "linkat,symlink,symlinkat,chmod,fchmodat,truncate,437"
)
DESTRUCTIVE_PATH_SYSCALLS = frozenset((
    "rename", "renameat", "renameat2", "unlink", "unlinkat", "link",
    "linkat", "symlink", "symlinkat", "chmod", "fchmodat", "truncate",
))
SUPPORTED_SYSCALLS = frozenset((
    "open", "openat", "openat2", "creat", "open_by_handle_at",
    "437", "chdir", "fchdir", "execve", "execveat", "io_uring_setup",
)) | DESTRUCTIVE_PATH_SYSCALLS
OPEN_SYSCALLS = frozenset(("open", "openat", "openat2", "437", "creat"))
PRODUCTION_FILES = frozenset((
    "backbone.pth", "parent.pth", "geometry.pth", "selection.json",
))
SMOKE_FILES = frozenset(("smoke-receipt.json",))
ROBERTA_FILES = frozenset((
    "config.json", "tokenizer_config.json", "tokenizer.json", "vocab.json",
    "merges.txt", "pytorch_model.bin",
))
PLATFORM_PROBE_SPECS = (
    ("/bin/sh", ("/bin/sh", "-c", "uname -p 2> /dev/null")),
    ("/usr/bin/uname", ("uname", "-p")),
)
VIOLATION_FIELDS = (
    "code", "pid", "trace_file", "entry_line", "resume_line",
    "timestamp", "syscall", "success", "errno", "raw_path",
    "candidate_path", "resolved_path", "rule", "detail",
)

_TIMESTAMP = r"(?P<timestamp>[0-9]+(?:\.[0-9]+)?)"
_COMPLETE_RE = re.compile(
    r"^" + _TIMESTAMP
    + r"\s+(?P<syscall>[a-zA-Z0-9_]+)\((?P<args>.*)\)\s+=\s+"
      r"(?P<result>.+)$"
)
_UNFINISHED_RE = re.compile(
    r"^" + _TIMESTAMP
    + r"\s+(?P<syscall>[a-zA-Z0-9_]+)\((?P<prefix>.*)"
      r"<unfinished \.\.\.>$"
)
_RESUMED_RE = re.compile(
    r"^" + _TIMESTAMP
    + r"\s+<\.\.\. (?P<syscall>[a-zA-Z0-9_]+) resumed>"
      r"(?P<suffix>.*)$"
)
_FAILED_RESULT_RE = re.compile(
    r"^-1\s+(?P<errno>[A-Z][A-Z0-9_]+)(?:\s+\(.*\))?$"
)
_SUCCESS_RESULT_RE = re.compile(
    r"^(?P<value>[0-9]+)(?:<(?P<annotation>.*)>)?$"
)
_SIGNAL_RE = re.compile(
    r"^(?:[0-9]+(?:\.[0-9]+)?\s+)?(?:--- .+ ---|\+\+\+ .+ \+\+\+)$"
)
_SCENE_RE = re.compile(r"^scene[0-9]{4}_[0-9]{2}$")
_SAFE_STAGE_SUFFIX_RE = re.compile(r"^[A-Za-z0-9._-]+$")


class TraceIntegrityError(Exception):
    pass


class UnsafeReportDestinationError(ValueError):
    pass


class TraceParseError(Exception):
    def __init__(self, message, pid=None, trace_file=None, line=None,
                 timestamp=None, syscall=None, resume_line=None):
        super().__init__(message)
        self.pid = pid
        self.trace_file = trace_file
        self.line = line
        self.timestamp = timestamp
        self.syscall = syscall
        self.resume_line = resume_line


def _absolute(path):
    value = os.fspath(path)
    if not isinstance(value, str):
        raise ValueError("public path arguments must resolve to str, not bytes")
    return os.path.abspath(os.path.expanduser(value))


def _safe_absolute(path):
    try:
        decoded = os.fsdecode(os.fspath(path))
        return os.path.abspath(os.path.expanduser(decoded))
    except Exception:
        return None


def _safe_path_list(values):
    if not isinstance(values, (list, tuple)):
        return []
    return [path for path in (_safe_absolute(value) for value in values)
            if path is not None]


def _safe_string_list(values):
    if not isinstance(values, (list, tuple)):
        return []
    return [value for value in values if isinstance(value, str)]


def _reject_symlink_components(path, label, allow_missing=False):
    """Reject symlinks in every existing component of an absolute path."""
    path = Path(_absolute(path))
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current = current / part
        if not os.path.lexists(str(current)):
            if allow_missing:
                break
            raise ValueError("{} does not exist: {}".format(label, current))
        if stat.S_ISLNK(os.lstat(str(current)).st_mode):
            raise ValueError(
                "{} has a symlink path component: {}".format(label, current)
            )


def _identity(value):
    return (
        value.st_dev,
        value.st_ino,
        stat.S_IFMT(value.st_mode),
        value.st_size,
        getattr(value, "st_mtime_ns", int(value.st_mtime * 1000000000)),
        getattr(value, "st_ctime_ns", int(value.st_ctime * 1000000000)),
    )


def _read_fd(fd):
    chunks = []
    while True:
        chunk = os.read(fd, 1024 * 1024)
        if not chunk:
            return b"".join(chunks)
        chunks.append(chunk)


def _stable_regular_snapshot(path, label):
    """Read twice through one no-follow fd and bind bytes to file identity."""
    path = _absolute(path)
    try:
        before = os.lstat(path)
    except OSError as error:
        raise TraceIntegrityError("{} is unavailable: {}".format(label, error))
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise TraceIntegrityError(
            "{} must be a regular non-symlink file".format(label)
        )
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(path, flags)
    except OSError as error:
        raise TraceIntegrityError("cannot open {}: {}".format(label, error))
    try:
        opened = os.fstat(fd)
        if _identity(opened) != _identity(before):
            raise TraceIntegrityError("{} identity changed before read".format(label))
        first = _read_fd(fd)
        middle = os.fstat(fd)
        os.lseek(fd, 0, os.SEEK_SET)
        second = _read_fd(fd)
        after_fd = os.fstat(fd)
    finally:
        os.close(fd)
    try:
        after_path = os.lstat(path)
    except OSError as error:
        raise TraceIntegrityError("{} disappeared after read: {}".format(label, error))
    identities = (
        _identity(before), _identity(opened), _identity(middle),
        _identity(after_fd), _identity(after_path),
    )
    if len(set(identities)) != 1:
        raise TraceIntegrityError("{} identity changed while reading".format(label))
    first_hash = hashlib.sha256(first).hexdigest()
    second_hash = hashlib.sha256(second).hexdigest()
    if first_hash != second_hash or first != second:
        raise TraceIntegrityError("{} content changed while reading".format(label))
    return {
        "path": path,
        "size": len(first),
        "mode": stat.S_IMODE(before.st_mode),
        "sha256": first_hash,
    }, first


def _violation(code, *, pid=None, trace_file=None, entry_line=None,
               resume_line=None, timestamp=None, syscall=None, success=None,
               errno=None, raw_path=None, candidate_path=None,
               resolved_path=None, rule=None, detail=""):
    value = {
        "code": code,
        "pid": pid,
        "trace_file": trace_file,
        "entry_line": entry_line,
        "resume_line": resume_line,
        "timestamp": timestamp,
        "syscall": syscall,
        "success": success,
        "errno": errno,
        "raw_path": raw_path,
        "candidate_path": candidate_path,
        "resolved_path": resolved_path,
        "rule": rule,
        "detail": detail,
    }
    assert set(value) == set(VIOLATION_FIELDS)
    return value


def _sort_violations(violations):
    def key(item):
        return (
            item["pid"] is None,
            -1 if item["pid"] is None else item["pid"],
            item["trace_file"] or "",
            item["entry_line"] is None,
            -1 if item["entry_line"] is None else item["entry_line"],
            item["resume_line"] is None,
            -1 if item["resume_line"] is None else item["resume_line"],
            item["code"],
            item["rule"] or "",
            item["detail"],
        )
    return sorted(violations, key=key)


def _canonical_json_bytes(payload):
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _write_report(path, payload):
    path = Path(_absolute(path))
    if not path.name:
        raise ValueError("report path must name a file")
    _reject_symlink_components(path.parent, "report parent")
    parent_before = os.stat(str(path.parent), follow_symlinks=False)
    if not stat.S_ISDIR(parent_before.st_mode):
        raise ValueError("report parent must be a directory")
    directory_fd = os.open(
        str(path.parent),
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        parent_opened = os.fstat(directory_fd)
        if (int(parent_opened.st_dev) != int(parent_before.st_dev)
                or int(parent_opened.st_ino) != int(parent_before.st_ino)):
            raise RuntimeError("report parent identity changed before write")
        descriptor = os.open(
            path.name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL
            | getattr(os, "O_NOFOLLOW", 0),
            0o600,
            dir_fd=directory_fd,
        )
        encoded = _canonical_json_bytes(payload)
        try:
            offset = 0
            while offset < len(encoded):
                written = os.write(descriptor, encoded[offset:])
                if written <= 0:
                    raise OSError("audit report write made no progress")
                offset += written
            os.fsync(descriptor)
            os.fchmod(descriptor, 0o444)
            os.fsync(descriptor)
            opened = os.fstat(descriptor)
            linked = os.stat(
                path.name, dir_fd=directory_fd, follow_symlinks=False
            )
            if (not stat.S_ISREG(opened.st_mode)
                    or int(opened.st_dev) != int(linked.st_dev)
                    or int(opened.st_ino) != int(linked.st_ino)
                    or int(opened.st_size) != len(encoded)
                    or stat.S_IMODE(opened.st_mode) != 0o444):
                raise RuntimeError("audit report pathname identity changed")
        finally:
            os.close(descriptor)
        os.fsync(directory_fd)
        parent_live = os.stat(str(path.parent), follow_symlinks=False)
        if (int(parent_live.st_dev) != int(parent_opened.st_dev)
                or int(parent_live.st_ino) != int(parent_opened.st_ino)):
            raise RuntimeError("report parent identity changed after write")
    finally:
        os.close(directory_fd)


def _decode_c_fragment(value):
    output = []
    index = 0
    simple = {
        "a": "\a", "b": "\b", "f": "\f", "n": "\n", "r": "\r",
        "t": "\t", "v": "\v", "\\": "\\", "\"": "\"", "'": "'",
    }
    while index < len(value):
        character = value[index]
        if character != "\\":
            output.append(character)
            index += 1
            continue
        index += 1
        if index >= len(value):
            raise ValueError("incomplete C escape")
        escaped = value[index]
        if escaped in simple:
            output.append(simple[escaped])
            index += 1
        elif escaped == "x":
            digits = value[index + 1:index + 3]
            if len(digits) != 2 or not re.match(r"^[0-9A-Fa-f]{2}$", digits):
                raise ValueError("invalid hexadecimal C escape")
            output.append(chr(int(digits, 16)))
            index += 3
        elif escaped in "01234567":
            end = index + 1
            while end < len(value) and end < index + 3 and value[end] in "01234567":
                end += 1
            output.append(chr(int(value[index:end], 8)))
            index = end
        else:
            raise ValueError("unknown C escape")
    decoded = "".join(output)
    if "\x00" in decoded:
        raise ValueError("NUL is not valid in a filesystem path")
    return decoded


def _decode_path_token(token):
    token = token.strip()
    match = re.match(r'^"((?:[^"\\]|\\.)*)"$', token)
    if match is None:
        if token.startswith("0x") or token == "NULL":
            raise ValueError("path was rendered as a C pointer")
        if token.startswith('"') or token.endswith("..."):
            raise ValueError("path string was truncated or malformed")
        raise ValueError("path argument is not a complete C string")
    return _decode_c_fragment(match.group(1))


def _parse_c_string_array(token):
    token = token.strip()
    if not token.startswith("[") or not token.endswith("]"):
        raise ValueError("argv is not a complete C-string array")
    inner = token[1:-1].strip()
    if not inner:
        return tuple()
    return tuple(_decode_path_token(part) for part in _split_arguments(inner))


def _decode_annotation(value):
    try:
        return _decode_c_fragment(value)
    except ValueError as error:
        raise ValueError("invalid fd path annotation: {}".format(error))


def _split_arguments(value):
    parts = []
    start = 0
    depths = {"(": 0, "[": 0, "{": 0, "<": 0}
    closing = {")": "(", "]": "[", "}": "{", ">": "<"}
    in_string = False
    escaped = False
    for index, character in enumerate(value):
        if in_string:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
            continue
        if character == '"':
            in_string = True
        elif character in depths:
            depths[character] += 1
        elif character in closing:
            opener = closing[character]
            depths[opener] -= 1
            if depths[opener] < 0:
                raise ValueError("unbalanced syscall arguments")
        elif character == "," and not any(depths.values()):
            parts.append(value[start:index].strip())
            start = index + 1
    if in_string or escaped or any(depths.values()):
        raise ValueError("truncated or unbalanced syscall arguments")
    parts.append(value[start:].strip())
    return parts


def _parse_result(value):
    value = value.strip()
    if value.startswith("?"):
        raise ValueError("syscall result is unknown")
    failed = _FAILED_RESULT_RE.match(value)
    if failed is not None:
        return False, -1, failed.group("errno"), None
    succeeded = _SUCCESS_RESULT_RE.match(value)
    if succeeded is None:
        raise ValueError("unrecognized syscall result")
    annotation = succeeded.group("annotation")
    if annotation is not None:
        annotation = _decode_annotation(annotation)
    return True, int(succeeded.group("value")), None, annotation


def _parse_annotated_fd(token):
    token = token.strip()
    match = re.match(r"^(AT_FDCWD|-?[0-9]+)(?:<(.*)>)?$", token)
    if match is None:
        return None, None
    annotation = match.group(2)
    if annotation is not None:
        annotation = _decode_annotation(annotation)
    return match.group(1), annotation


def _normalize_candidate(raw_path, base):
    if raw_path == "":
        return os.path.normpath(base) if base else None
    if os.path.isabs(raw_path):
        return os.path.normpath(raw_path)
    if base is None or not os.path.isabs(base):
        return None
    return os.path.normpath(os.path.join(base, raw_path))


def _dirfd_base(token, cwd):
    descriptor, annotation = _parse_annotated_fd(token)
    if annotation is not None and os.path.isabs(annotation):
        return os.path.normpath(annotation)
    if descriptor == "AT_FDCWD":
        return cwd
    return None


def _components(path):
    return [part for part in path.replace("\\", "/").split("/") if part not in ("", ".")]


def _has_sequence(parts, sequence):
    width = len(sequence)
    return any(tuple(parts[index:index + width]) == tuple(sequence)
               for index in range(len(parts) - width + 1))


def _deny_rule(path, include_generic_tokens=False):
    if path is None:
        return None
    parts = _components(path)
    if any(part.startswith("ScanRefer_filtered_val") for part in parts):
        return "ScanRefer_filtered_val*"
    if "val_v3scans.pkl" in parts:
        return "val_v3scans.pkl"
    if _has_sequence(parts, ("superpoints", "val")):
        return "superpoints/val"
    if "group_free_pred_bboxes_val" in parts:
        return "group_free_pred_bboxes_val"
    if "scanrefer_pred_spans_val" in parts:
        return "scanrefer_pred_spans_val"
    for leaf in ("val", "geometry_val", "geometry_official_val"):
        if _has_sequence(parts, ("e71_top16", leaf)):
            return "e71_top16/{}".format(leaf)
    if any(part == "official_result" or part.startswith("official_result.")
           for part in parts):
        return "official_result"
    if len(parts) >= 3 and parts[-3:] == [
            "data", "meta_data", "scannetv2_val.txt"]:
        return "repo-data/meta_data/scannetv2_val.txt"
    if include_generic_tokens:
        for part in parts:
            tokens = [
                token.lower() for token in re.split(r"[-_.]+", part) if token
            ]
            for forbidden in ("val", "validation", "official"):
                if forbidden in tokens:
                    return "independent-output-token:{}".format(forbidden)
    return None


def _is_train_only_reranker_profile(config):
    return bool(
        config.get("residual_profile")
        or config.get("hierarchical_profile")
    )


def _residual_forbidden_dependency_rule(path, config):
    """Reject validation/claim namespaces for train-only reranker profiles.

    The output receipt is explicitly exempted because it is the audit target;
    every other pathname is checked in both its raw and resolved forms by the
    caller.  Keeping this rule profile-scoped preserves the older finetune
    audit's deliberately narrower compatibility policy.
    """
    if (not _is_train_only_reranker_profile(config) or path is None
            or _is_allowed_output(path, config)):
        return None
    for part in Path(path).parts:
        token = part.lower()
        if token in ("val", "validation", "official", "claim", "receipt"):
            return "residual-forbidden-component:{}".format(token)
        if token.startswith("geometry_val") or token.startswith("official_result"):
            return "residual-forbidden-component:{}".format(token)
        if token.endswith(".claim") or token.endswith(".claim.json"):
            return "residual-forbidden-component:claim"
        if token.endswith(".receipt") or token.endswith(".receipt.json"):
            return "residual-forbidden-component:receipt"
    return None


def _within(path, root):
    path = os.path.normpath(path)
    root = os.path.normpath(root)
    return path == root or path.startswith(root + os.sep)


def _paths_overlap(first, second):
    return _within(first, second) or _within(second, first)


def _relative_parts(path, root):
    relative = os.path.relpath(path, root)
    return tuple(part for part in relative.split(os.sep) if part not in ("", "."))


def _is_generic_token_scope(path, config):
    if path is None or not os.path.isabs(path):
        return False
    if (_within(path, config["data_root"])
            or _within(path, config["output_dir"])):
        return True
    output = config["output_dir"]
    parent = os.path.dirname(output)
    if not _within(path, parent):
        return False
    parts = _relative_parts(path, parent)
    stage_prefix = ".{}.staging-".format(os.path.basename(output))
    return bool(parts and parts[0].startswith(stage_prefix))


def _is_allowed_source_gate_named_code_dependency(path, config):
    if (path is None or not os.path.isabs(path)
            or not _is_exact_source_gate_smoke(config)
            or not _within(path, config["initial_cwd"])):
        return False
    relative = _relative_parts(path, config["initial_cwd"])
    source_files = (
        ("scripts", "cache_scanrefer_rec_candidates.py"),
        ("scripts", "rec_geometry_cache.py"),
    )
    if relative in source_files:
        return True
    if len(relative) != 3 or relative[:2] != ("scripts", "__pycache__"):
        return False
    return any(
        re.match(
            r"^{}\.cpython-37(?:\.opt-[12])?\.pyc$".format(
                re.escape(Path(*source_file).stem)
            ),
            relative[2],
        ) is not None
        for source_file in source_files
    )


def _source_gate_forbidden_dependency_rule(path, config):
    if (path is None or not os.path.isabs(path)
            or not _is_exact_source_gate_smoke(config)):
        return None
    if _is_allowed_source_gate_named_code_dependency(path, config):
        return None
    for part in Path(path).parts:
        if part.lower() in (".cache", "cache", "validation", "val", "test"):
            return "source-gate-dependency-directory:{}".format(
                part.lower()
            )
    relative = None
    for root in (config["initial_cwd"], config["data_root"]):
        if _within(path, root):
            relative = _relative_parts(path, root)
            break
    if relative is None:
        return None
    for part in relative:
        tokens = [
            token.lower() for token in re.split(r"[-_.]+", part) if token
        ]
        for forbidden in ("validation", "val", "test", "cache"):
            if forbidden in tokens:
                return "source-gate-dependency-token:{}".format(forbidden)
    return None


def _is_allowed_output(path, config):
    output = config["output_dir"]
    parent = os.path.dirname(output)
    if _is_train_only_reranker_profile(config):
        artifact_name = (
            "selected_hierarchical.pth"
            if config.get("hierarchical_profile")
            else "selected_residual.pth"
        )
        if path in (parent, output):
            return True
        if os.path.dirname(path) == output and os.path.basename(path) in (
                ".result-receipt.json.pending",
                "result-receipt.json",
                artifact_name,
        ):
            return True
        return False
    expected = SMOKE_FILES if config["mode"] == "smoke" else PRODUCTION_FILES
    if path in (parent, output):
        return True
    if os.path.dirname(path) == output and os.path.basename(path) in expected:
        return True
    if not _within(path, parent):
        return False
    parts = _relative_parts(path, parent)
    if not parts:
        return True
    stage_prefix = ".{}.staging-".format(os.path.basename(output))
    if not parts[0].startswith(stage_prefix):
        return False
    suffix = parts[0][len(stage_prefix):]
    if not suffix or _SAFE_STAGE_SUFFIX_RE.match(suffix) is None:
        return False
    if len(parts) == 1:
        return True
    if len(parts) != 2:
        return False
    name = parts[1]
    if name in expected:
        return True
    if any(re.match(
        r"^\.{}\.[A-Za-z0-9._-]+\.tmp$".format(re.escape(final_name)),
        name,
    ) is not None for final_name in expected):
        return True
    return (config["mode"] == "production"
            and any(re.match(
                r"^{}\.tmp\.[A-Za-z0-9._-]+$".format(
                    re.escape(final_name)
                ),
                name,
            ) is not None for final_name in ("parent.pth", "geometry.pth")))


def _is_allowed_data_root(path, config):
    if path in config["input_artifacts"]:
        return True
    if _is_allowed_output(path, config):
        return True
    for cache_root in config.get("residual_cache_roots", ()):
        if _within(path, cache_root):
            return True
    parts = _relative_parts(path, config["data_root"])
    if parts == ("train_v3scans.pkl",):
        return True
    if (len(parts) == 2 and parts[0] in ("scanrefer", "ScanRefer")
            and parts[1] in (
                "ScanRefer_filtered_train.txt",
                "ScanRefer_filtered_train.json",
            )):
        return True
    if (len(parts) == 3 and parts[:2] == ("superpoints", "train")
            and parts[2].endswith("_superpoint.pth")
            and _SCENE_RE.match(parts[2][:-len("_superpoint.pth")])):
        return True
    if (len(parts) == 3
            and parts[:2] == (
                "group_free_pred_bboxes", "group_free_pred_bboxes_train"
            )
            and parts[2].endswith(".npy")
            and _SCENE_RE.match(parts[2][:-len(".npy")])):
        return True
    if len(parts) == 2 and parts[0] == "roberta-base" and parts[1] in ROBERTA_FILES:
        return True
    return False


def _open_has_write_intent(syscall, arguments):
    if syscall == "creat":
        return True
    flags_index = 1 if syscall == "open" else 2
    if len(arguments) <= flags_index:
        raise ValueError("open syscall has too few arguments for flags")
    flags = arguments[flags_index]
    if syscall in ("openat2", "437"):
        match = re.search(r"(?:^|[,{])\s*flags=([^,}]+)", flags)
        if match is None:
            raise ValueError("openat2 flags are unavailable")
        flags = match.group(1).strip()
    return any(flag in flags.split("|") for flag in (
        "O_WRONLY", "O_RDWR", "O_CREAT", "O_TRUNC", "O_APPEND",
        "O_TMPFILE",
    ))


def _strip_device_annotation(path):
    if path is None:
        return None
    return re.sub(r"<(?:char|block) [0-9]+:[0-9]+>$", "", path)


def _is_allowed_source_gate_runtime_endpoint(path, config):
    if (path is None
            or not (_is_exact_source_gate_smoke(config)
                    or _is_train_only_reranker_profile(config))):
        return False
    if path in ("/dev/null", "/dev/null<char 1:3>"):
        return True
    path = _strip_device_annotation(path)
    if path == "/dev/null":
        return False
    if re.match(
            r"^/dev/nvidia(?:[0-9]+|ctl|-modeset|-uvm-tools|-uvm)$",
            path):
        return True
    torch_shm = re.match(r"^/dev/shm/torch_([0-9]+)_[0-9]+$", path)
    if (torch_shm is not None
            and int(torch_shm.group(1)) in config.get("trace_pids", ())):
        return True
    if re.match(r"^/dev/shm/[A-Za-z0-9]{6}$", path):
        return True
    scratch = config.get("runtime_scratch_dir")
    return (
        scratch is not None and path != scratch and _within(path, scratch)
    )


def _is_allowed_proc_comm_write(candidate, resolved, config):
    candidate_match = re.match(
        r"^/proc/self/task/([0-9]+)/comm$", candidate or ""
    )
    resolved_match = re.match(
        r"^/proc/([0-9]+)/task/([0-9]+)/comm$", resolved or ""
    )
    if candidate_match is None or resolved_match is None:
        return False
    task_pid = int(candidate_match.group(1))
    return (
        task_pid in config.get("trace_pids", ())
        and int(resolved_match.group(1)) == config.get("root_pid")
        and int(resolved_match.group(2)) == task_pid
    )


def _is_bound_output_alias(candidate, resolved, config):
    if resolved is None or not _is_allowed_output(resolved, config):
        return False
    match = re.match(r"^/proc/self/fd/[0-9]+/(.+)$", candidate or "")
    if match is None:
        return False
    relative = os.path.relpath(resolved, os.path.dirname(config["output_dir"]))
    return relative != ".." and match.group(1) == relative


def _is_exact_source_gate_publication_write(candidate, resolved, config):
    if not _is_bound_output_alias(candidate, resolved, config):
        return False
    parent = os.path.dirname(config["output_dir"])
    parts = _relative_parts(resolved, parent)
    if len(parts) != 2:
        return False
    stage_prefix = ".{}.staging-".format(
        os.path.basename(config["output_dir"])
    )
    stage_suffix = (
        parts[0][len(stage_prefix):]
        if parts[0].startswith(stage_prefix) else ""
    )
    return (
        bool(stage_suffix)
        and _SAFE_STAGE_SUFFIX_RE.match(stage_suffix) is not None
        and re.match(
            r"^\.smoke-receipt\.json\.[A-Za-z0-9._-]+\.tmp$",
            parts[1],
        ) is not None
    )


def _classify_successful_write_open(candidate, resolved, config):
    if (_is_allowed_output(candidate, config)
            and _is_allowed_output(resolved, config)):
        return "publication"
    if _is_bound_output_alias(candidate, resolved, config):
        return "publication"
    if _is_allowed_proc_comm_write(candidate, resolved, config):
        return "runtime"
    if (_is_allowed_source_gate_runtime_endpoint(candidate, config)
            and _is_allowed_source_gate_runtime_endpoint(resolved, config)):
        return "runtime"
    return None


def _base_event(record, result):
    return {
        "pid": record["pid"],
        "trace_file": record["trace_file"],
        "entry_line": record["entry_line"],
        "resume_line": record["resume_line"],
        "timestamp": record["timestamp"],
        "syscall": record["syscall"],
        "success": result[0],
        "errno": result[2],
        "raw_path": None,
        "candidate_path": None,
        "resolved_path": None,
    }


def _policy_violation(event, config, uncertain=None, forced=None):
    generic_token_scope = any(
        _is_generic_token_scope(value, config)
        for value in (
            event["raw_path"], event["candidate_path"],
            event["resolved_path"],
        )
    )
    matches = []
    for label, value in (
            ("raw", event["raw_path"]),
            ("candidate", event["candidate_path"]),
            ("resolved", event["resolved_path"])):
        rule = _deny_rule(
            value, include_generic_tokens=generic_token_scope
        )
        if rule is None:
            rule = _source_gate_forbidden_dependency_rule(value, config)
        if rule is None:
            rule = _residual_forbidden_dependency_rule(value, config)
        if rule is not None:
            matches.append((label, rule))
    common = dict(event)
    if matches:
        detail = "denied representations: " + ", ".join(
            "{}={}".format(label, rule) for label, rule in matches
        )
        return _violation(
            "deny_path", rule=matches[0][1], detail=detail, **common
        )
    if forced is not None:
        return _violation(
            forced[0], rule=forced[1], detail=forced[2], **common
        )
    if uncertain is not None:
        return _violation(
            "uncertain_path", rule="fail-closed-path-resolution",
            detail=uncertain, **common
        )
    if (event["success"] and event["syscall"] in OPEN_SYSCALLS
            and event["candidate_path"] is not None
            and _within(event["candidate_path"], config["data_root"])
            and (event["resolved_path"] is None
                 or not _within(
                     event["resolved_path"], config["data_root"]
                 ))):
        return _violation(
            "data_root_allow_miss",
            rule="data-root-resolution-membership",
            detail=(
                "successful open candidate is inside DATA_ROOT but its "
                "authoritative resolved path is outside DATA_ROOT"
            ),
            **common
        )
    if event["success"]:
        for label, value in (
                ("candidate", event["candidate_path"]),
                ("resolved", event["resolved_path"])):
            if (value is not None and _within(value, config["data_root"])
                    and not _is_allowed_data_root(value, config)):
                return _violation(
                    "data_root_allow_miss",
                    rule="data-root-success-allowlist",
                    detail=(
                        "successful DATA_ROOT {} path is outside the allowlist"
                    ).format(label),
                    **common
                )
    return None


def _record_error(record, message):
    raise TraceParseError(
        message,
        pid=record.get("pid"),
        trace_file=record.get("trace_file"),
        line=record.get("entry_line"),
        timestamp=record.get("timestamp"),
        syscall=record.get("syscall"),
        resume_line=record.get("resume_line"),
    )


def _destructive_path_operands(syscall, arguments, cwd):
    def direct(index, role):
        if len(arguments) <= index:
            raise ValueError("{} has too few arguments".format(syscall))
        raw = _decode_path_token(arguments[index])
        return {
            "role": role,
            "raw_path": raw,
            "candidate_path": _normalize_candidate(raw, cwd),
        }

    def relative_to(dirfd_index, path_index, role):
        if len(arguments) <= max(dirfd_index, path_index):
            raise ValueError("{} has too few arguments".format(syscall))
        base = _dirfd_base(arguments[dirfd_index], cwd)
        raw = _decode_path_token(arguments[path_index])
        return {
            "role": role,
            "raw_path": raw,
            "candidate_path": _normalize_candidate(raw, base),
        }

    if syscall == "rename":
        return [direct(0, "source"), direct(1, "destination")]
    if syscall in ("renameat", "renameat2"):
        return [
            relative_to(0, 1, "source"),
            relative_to(2, 3, "destination"),
        ]
    if syscall == "unlink":
        return [direct(0, "target")]
    if syscall == "unlinkat":
        return [relative_to(0, 1, "target")]
    if syscall == "link":
        return [direct(0, "source"), direct(1, "destination")]
    if syscall == "linkat":
        return [
            relative_to(0, 1, "source"),
            relative_to(2, 3, "destination"),
        ]
    if syscall == "symlink":
        return [direct(1, "destination")]
    if syscall == "symlinkat":
        return [relative_to(1, 2, "destination")]
    if syscall in ("chmod", "truncate"):
        return [direct(0, "target")]
    if syscall == "fchmodat":
        return [relative_to(0, 1, "target")]
    raise ValueError("unsupported destructive path syscall")


def _is_allowed_destructive_path(path, config):
    if path is None:
        return False
    if _is_allowed_output(path, config):
        return True
    scratch = config.get("runtime_scratch_dir")
    return scratch is not None and _within(path, scratch)


def _protected_mutation_label(path, config):
    if path in config["input_artifacts"]:
        index = config["input_artifacts"].index(path)
        return ("backbone", "parent", "geometry")[index]
    for label, root in (
            ("base_cache", config.get("base_cache")),
            ("geometry_cache", config.get("geometry_cache")),
            ("data_root", config.get("data_root")),
            ("initial_cwd", config.get("initial_cwd"))):
        if root is not None and _within(path, root):
            return label
    return None


def _event_for_destructive_operand(record, result, operand):
    event = _base_event(record, result)
    event.update({
        "raw_path": operand["raw_path"],
        "candidate_path": operand["candidate_path"],
        "resolved_path": None,
    })
    return event


def _evaluate_record(record, state, config):
    try:
        arguments = _split_arguments(record["args"])
        result = _parse_result(record["result"])
    except ValueError as error:
        _record_error(record, str(error))
    syscall = record["syscall"]
    event = _base_event(record, result)
    uncertain = None
    forced = None

    try:
        if syscall in OPEN_SYSCALLS:
            if syscall in ("open", "creat"):
                path_index = 0
                base = state["cwd"]
            else:
                path_index = 1
                if len(arguments) < 2:
                    raise ValueError("openat syscall has too few arguments")
                base = _dirfd_base(arguments[0], state["cwd"])
            if len(arguments) <= path_index:
                raise ValueError("open syscall has too few arguments")
            raw = _decode_path_token(arguments[path_index])
            candidate = _normalize_candidate(raw, base)
            resolved = result[3] if result[0] else None
            if resolved is not None and not os.path.isabs(resolved):
                resolved = None
            if resolved is not None:
                resolved = os.path.normpath(resolved)
            event.update(raw_path=raw, candidate_path=candidate,
                         resolved_path=resolved)
            if result[0] and resolved is None:
                uncertain = "successful open lacks an absolute -yy returned-fd path"
            elif not result[0] and candidate is None:
                uncertain = "failed relative open uses an unknown dirfd"
            elif result[0] and _open_has_write_intent(syscall, arguments):
                write_class = _classify_successful_write_open(
                    candidate, resolved, config
                )
                if write_class is None:
                    forced = (
                        "write_allow_miss",
                        "successful-write-output-allowlist",
                        "successful write open is outside the mode-specific "
                        "output allowlist",
                    )
                elif (write_class == "publication"
                      and _is_exact_source_gate_smoke(config)):
                    config["source_gate_publication_writes"].append(
                        (candidate, resolved)
                    )

        elif syscall == "open_by_handle_at":
            resolved = result[3] if result[0] else None
            if resolved is not None and os.path.isabs(resolved):
                resolved = os.path.normpath(resolved)
            else:
                resolved = None
            event["resolved_path"] = resolved
            if not result[0]:
                uncertain = "failed open_by_handle_at cannot reveal its target path"
            elif resolved is None:
                uncertain = "successful open_by_handle_at lacks a resolved fd path"
            elif _open_has_write_intent(syscall, arguments):
                forced = (
                    "write_allow_miss",
                    "successful-write-output-allowlist",
                    "write-intent open_by_handle_at has no authoritative "
                    "candidate path",
                )

        elif syscall in ("execve", "execveat"):
            if syscall == "execve":
                if len(arguments) < 2:
                    raise ValueError("execve has too few arguments")
                raw = _decode_path_token(arguments[0])
                base = state["cwd"]
                argv_index = 1
            else:
                if len(arguments) < 3:
                    raise ValueError("execveat has too few arguments")
                base = _dirfd_base(arguments[0], state["cwd"])
                raw = _decode_path_token(arguments[1])
                argv_index = 2
            candidate = _normalize_candidate(raw, base)
            event.update(raw_path=raw, candidate_path=candidate,
                         resolved_path=None)
            argv = None
            if result[0]:
                try:
                    argv = _parse_c_string_array(arguments[argv_index])
                except ValueError:
                    argv = None
            expected_executable = config["expected_initial_runner_argv"][0]
            is_bound_initial_exec = (
                result[0]
                and syscall == "execve"
                and record["pid"] == config.get("root_pid")
                and record["entry_line"] == 1
                and raw == expected_executable
                and candidate == expected_executable
                and argv == config["expected_initial_runner_argv"]
            )
            if is_bound_initial_exec:
                config["initial_interpreter_usage"] += 1
                event["resolved_path"] = config[
                    "initial_interpreter"
                ]["resolved_path"]
            elif result[0]:
                matched_probe = None
                if syscall == "execve" and argv is not None:
                    for probe in config["platform_probe_executables"]:
                        if (raw == probe["logical_path"]
                                and candidate == probe["logical_path"]
                                and argv == tuple(probe["argv"])):
                            usage = config["platform_probe_usage"][
                                probe["logical_path"]
                            ]
                            config["platform_probe_usage"][
                                probe["logical_path"]
                            ] = usage + 1
                            if usage == 0:
                                matched_probe = probe
                            break
                if matched_probe is not None:
                    event["resolved_path"] = matched_probe["resolved_path"]
                else:
                    forced = (
                        "uncertain_path",
                        "successful-exec-without-authoritative-target",
                        "successful exec is not one unused exact bound command",
                    )
            elif candidate is None:
                uncertain = "relative exec path uses an unknown dirfd or cwd"

        elif syscall == "chdir":
            if not arguments:
                raise ValueError("chdir has too few arguments")
            raw = _decode_path_token(arguments[0])
            candidate = _normalize_candidate(raw, state["cwd"])
            event.update(raw_path=raw, candidate_path=candidate,
                         resolved_path=candidate if result[0] else None)
            if result[0]:
                state["cwd"] = candidate
                forced = (
                    "chdir_success", "successful-chdir-forbidden",
                    "successful chdir makes later relative-path provenance unsafe",
                )
            elif candidate is None:
                uncertain = "failed chdir has an unknown relative base"

        elif syscall == "fchdir":
            if not arguments:
                raise ValueError("fchdir has too few arguments")
            _descriptor, annotation = _parse_annotated_fd(arguments[0])
            candidate = (os.path.normpath(annotation)
                         if annotation is not None and os.path.isabs(annotation)
                         else None)
            event.update(candidate_path=candidate,
                         resolved_path=candidate if result[0] else None)
            if result[0]:
                state["cwd"] = candidate
                forced = (
                    "fchdir_success", "successful-fchdir-forbidden",
                    "successful fchdir makes later relative-path provenance unsafe",
                )

        elif syscall == "io_uring_setup":
            if result[0]:
                forced = (
                    "io_uring_setup_success", "successful-io-uring-forbidden",
                    "io_uring can issue file access outside the traced syscall set",
                )

        elif syscall in DESTRUCTIVE_PATH_SYSCALLS:
            operands = _destructive_path_operands(
                syscall, arguments, state["cwd"]
            )
            config.setdefault("destructive_path_calls", []).append({
                "pid": record["pid"],
                "syscall": syscall,
                "success": result[0],
                "errno": result[2],
                "paths": [dict(operand) for operand in operands],
            })
            for operand in operands:
                operand_event = _event_for_destructive_operand(
                    record, result, operand
                )
                violation = _policy_violation(operand_event, config)
                if violation is not None:
                    return violation
            event = _event_for_destructive_operand(record, result, operands[0])
            if result[0]:
                offender = next(
                    (
                        operand for operand in operands
                        if not _is_allowed_destructive_path(
                            operand["candidate_path"], config
                        )
                    ),
                    None,
                )
                if offender is not None:
                    event = _event_for_destructive_operand(
                        record, result, offender
                    )
                    protected_label = _protected_mutation_label(
                        offender["candidate_path"], config
                    )
                    if protected_label is not None:
                        forced = (
                            "protected_path_mutation",
                            "successful-protected-path-mutation",
                            "successful {} {} path targets protected {}"
                            .format(
                                syscall, offender["role"], protected_label
                            ),
                        )
                    elif offender["candidate_path"] is None:
                        uncertain = (
                            "successful {} {} path has an unknown base"
                        ).format(syscall, offender["role"])
                    else:
                        forced = (
                            "path_mutation_allow_miss",
                            "successful-path-mutation-allowlist",
                            "successful {} {} path is outside the "
                            "mode-specific output and scratch allowlists"
                            .format(syscall, offender["role"]),
                        )
        else:
            _record_error(record, "unsupported traced syscall")
    except ValueError as error:
        _record_error(record, str(error))

    if event["success"] and syscall not in DESTRUCTIVE_PATH_SYSCALLS:
        authoritative_path = event["resolved_path"] or event["candidate_path"]
        if authoritative_path is not None:
            config.setdefault("opened_paths", []).append({
                "pid": event["pid"],
                "syscall": event["syscall"],
                "path": authoritative_path,
            })
    return _policy_violation(
        event, config, uncertain=uncertain, forced=forced
    )


def _parse_complete_match(match, *, pid, trace_file, entry_line,
                          resume_line=None, entry_timestamp=None):
    syscall = match.group("syscall")
    if syscall not in SUPPORTED_SYSCALLS:
        raise TraceParseError(
            "unsupported syscall in selected trace: {}".format(syscall),
            pid=pid, trace_file=trace_file, line=entry_line,
            timestamp=entry_timestamp or match.group("timestamp"),
            syscall=syscall, resume_line=resume_line,
        )
    return {
        "pid": pid,
        "trace_file": trace_file,
        "entry_line": entry_line,
        "resume_line": resume_line,
        "timestamp": entry_timestamp or match.group("timestamp"),
        "syscall": syscall,
        "args": match.group("args"),
        "result": match.group("result"),
    }


def _parse_trace(text, pid, trace_file, config):
    state = {"cwd": config["initial_cwd"]}
    pending = None
    violations = []
    syscall_count = 0
    for line_number, line in enumerate(text.splitlines(), 1):
        if _SIGNAL_RE.match(line):
            continue
        resumed = _RESUMED_RE.match(line)
        if resumed is not None:
            if pending is None:
                raise TraceParseError(
                    "orphan resumed syscall", pid=pid, trace_file=trace_file,
                    line=line_number, timestamp=resumed.group("timestamp"),
                    syscall=resumed.group("syscall"), resume_line=line_number,
                )
            if resumed.group("syscall") != pending["syscall"]:
                raise TraceParseError(
                    "resumed syscall does not match pending entry", pid=pid,
                    trace_file=trace_file, line=pending["line"],
                    timestamp=pending["timestamp"], syscall=pending["syscall"],
                    resume_line=line_number,
                )
            reconstructed = (
                pending["timestamp"] + " " + pending["syscall"] + "("
                + pending["prefix"] + resumed.group("suffix")
            )
            complete = _COMPLETE_RE.match(reconstructed)
            if complete is None:
                raise TraceParseError(
                    "resumed syscall could not be reconstructed", pid=pid,
                    trace_file=trace_file, line=pending["line"],
                    timestamp=pending["timestamp"], syscall=pending["syscall"],
                    resume_line=line_number,
                )
            record = _parse_complete_match(
                complete, pid=pid, trace_file=trace_file,
                entry_line=pending["line"], resume_line=line_number,
                entry_timestamp=pending["timestamp"],
            )
            pending = None
        else:
            unfinished = _UNFINISHED_RE.match(line)
            if unfinished is not None:
                if pending is not None:
                    raise TraceParseError(
                        "new unfinished syscall while another is pending",
                        pid=pid, trace_file=trace_file, line=line_number,
                        timestamp=unfinished.group("timestamp"),
                        syscall=unfinished.group("syscall"),
                    )
                if unfinished.group("syscall") not in SUPPORTED_SYSCALLS:
                    raise TraceParseError(
                        "unsupported unfinished syscall", pid=pid,
                        trace_file=trace_file, line=line_number,
                        timestamp=unfinished.group("timestamp"),
                        syscall=unfinished.group("syscall"),
                    )
                pending = {
                    "line": line_number,
                    "timestamp": unfinished.group("timestamp"),
                    "syscall": unfinished.group("syscall"),
                    "prefix": unfinished.group("prefix"),
                }
                continue
            complete = _COMPLETE_RE.match(line)
            if complete is None:
                raise TraceParseError(
                    "unrecognized or truncated trace line", pid=pid,
                    trace_file=trace_file, line=line_number,
                )
            if pending is not None:
                raise TraceParseError(
                    "completed syscall encountered while another is pending",
                    pid=pid, trace_file=trace_file, line=pending["line"],
                    timestamp=pending["timestamp"], syscall=pending["syscall"],
                )
            record = _parse_complete_match(
                complete, pid=pid, trace_file=trace_file,
                entry_line=line_number,
            )
        syscall_count += 1
        violation = _evaluate_record(record, state, config)
        if violation is not None:
            violations.append(violation)
    if pending is not None:
        raise TraceParseError(
            "trace ended with an unfinished syscall", pid=pid,
            trace_file=trace_file, line=pending["line"],
            timestamp=pending["timestamp"], syscall=pending["syscall"],
        )
    return syscall_count, violations


def _discover_traces(prefix):
    prefix = Path(prefix)
    parent = prefix.parent
    name_prefix = prefix.name + "."
    found = []
    try:
        entries = list(parent.iterdir())
    except OSError as error:
        raise TraceIntegrityError("cannot enumerate trace prefix: {}".format(error))
    for path in entries:
        if not path.name.startswith(name_prefix):
            continue
        suffix = path.name[len(name_prefix):]
        if suffix.isdigit():
            found.append((int(suffix), path))
    found.sort(key=lambda item: (item[0], item[1].name))
    if not found:
        raise TraceIntegrityError("trace prefix has no PREFIX.<pid> files")
    return found


def _parse_runner_exit(snapshot):
    try:
        text = snapshot.decode("ascii").strip()
    except UnicodeDecodeError:
        return None
    if re.match(r"^-?[0-9]+$", text) is None:
        return None
    return int(text)


def _initial_report(config):
    return {
        "schema": SCHEMA,
        "pass": False,
        "validation_data_accessed": False,
        "opened_path_count": 0,
        "opened_path_sha256": hashlib.sha256(
            _canonical_json_bytes([])
        ).hexdigest(),
        "destructive_path_call_count": 0,
        "destructive_path_call_sha256": hashlib.sha256(
            _canonical_json_bytes([])
        ).hexdigest(),
        "inputs": {
            "trace_prefix": config["trace_prefix"],
            "data_root": config["data_root"],
            "initial_cwd": config["initial_cwd"],
            "initial_interpreter": config.get("initial_interpreter"),
            "expected_initial_runner_argv": list(
                config.get("expected_initial_runner_argv", ())
            ),
            "platform_probe_executables": config.get(
                "platform_probe_executables", []
            ),
            "root_pid": config.get("root_pid"),
            "input_artifacts": [],
            "base_cache": config.get("base_cache"),
            "geometry_cache": config.get("geometry_cache"),
            "output_dir": config["output_dir"],
            "runtime_scratch_dir": config.get("runtime_scratch_dir"),
            "mode": config["mode"],
            "runner_exit_code_file": {
                "path": config["runner_exit_code_file"],
                "sha256": None,
                "exit_code": None,
            },
            "receipt": {
                "path": config["receipt_path"],
                "sha256": None,
                "size": None,
                "validation_data_accessed": None,
            },
            "traces": [],
        },
        "counts": {
            "trace_files": 0,
            "empty_trace_files": 0,
            "syscalls": 0,
            "violations": 0,
            "deny_path": 0,
            "data_root_allow_miss": 0,
            "uncertain_path": 0,
        },
        "violations": [],
    }


def _source_gate_runner_path():
    return os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "probe_scanrefer_rec_source_gate.py",
    )


def _residual_runner_path():
    return os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "train_scanrefer_rec_selective_residual.py",
    )


def _hierarchical_runner_path():
    return os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "train_scanrefer_rec_hierarchical_reranker.py",
    )


def _argv_names_residual_runner(argv, initial_cwd):
    if len(argv) < 2:
        return False
    if (len(argv) >= 3
            and argv[1:3] == (
                "-m", "scripts.train_scanrefer_rec_selective_residual"
            )):
        return True
    candidate = _normalize_candidate(argv[1], initial_cwd)
    if candidate is None:
        return False
    try:
        return os.path.samefile(candidate, _residual_runner_path())
    except OSError:
        return False


def _argv_names_hierarchical_runner(argv, initial_cwd):
    if len(argv) < 2:
        return False
    if (len(argv) >= 3
            and argv[1:3] == (
                "-m", "scripts.train_scanrefer_rec_hierarchical_reranker"
            )):
        return True
    candidate = _normalize_candidate(argv[1], initial_cwd)
    if candidate is None:
        return False
    try:
        return os.path.samefile(candidate, _hierarchical_runner_path())
    except OSError:
        return False


def build_residual_training_argv(
        interpreter, base_cache, geometry_cache, parent_artifact,
        geometry_artifact, output_dir):
    """Return the only residual training argv accepted by this audit."""
    return [
        _absolute(interpreter),
        _residual_runner_path(),
        "--base-cache", _absolute(base_cache),
        "--geometry-cache", _absolute(geometry_cache),
        "--parent-artifact", _absolute(parent_artifact),
        "--geometry-artifact", _absolute(geometry_artifact),
        "--output-dir", _absolute(output_dir),
        "--device", "cuda:0",
    ]


def build_hierarchical_training_argv(
        interpreter, base_cache, geometry_cache, parent_artifact,
        geometry_artifact, output_dir):
    """Return the only hierarchical training argv accepted by this audit."""
    return [
        _absolute(interpreter),
        _hierarchical_runner_path(),
        "--base-cache", _absolute(base_cache),
        "--geometry-cache", _absolute(geometry_cache),
        "--parent-artifact", _absolute(parent_artifact),
        "--geometry-artifact", _absolute(geometry_artifact),
        "--output-dir", _absolute(output_dir),
        "--device", "cuda:0",
    ]


def _expected_residual_argv(config):
    return tuple(build_residual_training_argv(
        config["initial_interpreter"]["logical_path"],
        config["base_cache"],
        config["geometry_cache"],
        config["input_artifacts"][1],
        config["input_artifacts"][2],
        config["output_dir"],
    ))


def _expected_hierarchical_argv(config):
    return tuple(build_hierarchical_training_argv(
        config["initial_interpreter"]["logical_path"],
        config["base_cache"],
        config["geometry_cache"],
        config["input_artifacts"][1],
        config["input_artifacts"][2],
        config["output_dir"],
    ))


def _validate_residual_command(config):
    names_residual = _argv_names_residual_runner(
        config["expected_initial_runner_argv"], config["initial_cwd"]
    )
    if config.get("residual_profile") != names_residual:
        raise ValueError(
            "residual mode and residual runner command must be selected together"
        )
    if names_residual and config["expected_initial_runner_argv"] != \
            _expected_residual_argv(config):
        raise ValueError("residual runner requires the exact train-only argv")


def _validate_hierarchical_command(config):
    names_hierarchical = _argv_names_hierarchical_runner(
        config["expected_initial_runner_argv"], config["initial_cwd"]
    )
    if config.get("hierarchical_profile") != names_hierarchical:
        raise ValueError(
            "hierarchical mode and runner command must be selected together"
        )
    if (names_hierarchical
            and config["expected_initial_runner_argv"]
            != _expected_hierarchical_argv(config)):
        raise ValueError(
            "hierarchical runner requires the exact train-only argv"
        )


def _argv_names_source_gate_runner(argv, initial_cwd):
    if len(argv) < 2:
        return False
    if (len(argv) >= 3
            and argv[1:3] == (
                "-m", "scripts.probe_scanrefer_rec_source_gate"
            )):
        return True
    candidate = _normalize_candidate(argv[1], initial_cwd)
    if candidate is None:
        return False
    try:
        return os.path.samefile(candidate, _source_gate_runner_path())
    except OSError:
        return False


def _raw_argv_names_source_gate_runner(argv, initial_cwd):
    if not isinstance(argv, (list, tuple)):
        return False
    try:
        return _argv_names_source_gate_runner(
            tuple(argv), _safe_absolute(initial_cwd)
        )
    except (TypeError, ValueError):
        return False


def _expected_source_gate_argv(config, probe_steps):
    expected = (
        config["initial_interpreter"]["logical_path"],
        _source_gate_runner_path(),
        "--data-root", config["data_root"],
        "--backbone-checkpoint", config["input_artifacts"][0],
        "--parent-reranker", config["input_artifacts"][1],
        "--geometry-reranker", config["input_artifacts"][2],
        "--output-dir", config["output_dir"],
        "--device", "cuda:0",
        "--probe-steps", str(probe_steps),
    )
    return expected


def _source_gate_probe_steps(config):
    argv = config["expected_initial_runner_argv"]
    for probe_steps in (1, 306):
        if argv == _expected_source_gate_argv(config, probe_steps):
            return probe_steps
    return None


def _is_exact_source_gate_smoke(config):
    return (
        config["mode"] == "smoke"
        and _source_gate_probe_steps(config) is not None
    )


def _validate_source_gate_command(config):
    argv = config["expected_initial_runner_argv"]
    if not _argv_names_source_gate_runner(argv, config["initial_cwd"]):
        return
    if not _is_exact_source_gate_smoke(config):
        raise ValueError(
            "source-gate runner requires exact smoke argv with probe-steps "
            "1 or 306"
        )


def _normalize_source_gate_control_paths(config):
    report_path = config["report_path"]
    if os.path.lexists(report_path):
        raise UnsafeReportDestinationError(
            "source-gate report_path must be fresh"
        )
    try:
        _reject_symlink_components(
            report_path, "report_path", allow_missing=True
        )
    except ValueError as error:
        raise UnsafeReportDestinationError(str(error))
    _reject_symlink_components(
        config["trace_prefix"], "trace_prefix", allow_missing=True
    )
    _reject_symlink_components(
        config["runner_exit_code_file"], "runner_exit_code_file"
    )
    config["report_path"] = str(Path(report_path).resolve(strict=False))
    config["trace_prefix"] = str(
        Path(config["trace_prefix"]).resolve(strict=False)
    )
    config["runner_exit_code_file"] = str(
        Path(config["runner_exit_code_file"]).resolve(strict=True)
    )


def _has_cache_directory_component(path):
    return any(
        part.lower() in (".cache", "cache", "__pycache__")
        for part in Path(path).parts
    )


def _is_output_namespace(path, config):
    output = config["output_dir"]
    if _paths_overlap(path, output):
        return True
    parent = os.path.dirname(output)
    if not _within(path, parent):
        return False
    parts = _relative_parts(path, parent)
    stage_prefix = ".{}.staging-".format(os.path.basename(output))
    return bool(parts and parts[0].startswith(stage_prefix))


def _paths_overlap_trace_namespace(path, trace_prefix):
    return (
        _paths_overlap(path, trace_prefix)
        or path.startswith(trace_prefix + ".")
    )


def _canonical_fallback_path(value):
    try:
        absolute = _safe_absolute(value)
        if absolute is None:
            return None
        return str(Path(absolute).resolve(strict=False))
    except Exception:
        return None


def _fallback_report_is_safe(
        *, trace_prefix, data_root, initial_cwd, input_artifacts, output_dir,
        runner_exit_code_file, receipt_path, report_path,
        runtime_scratch_dir, base_cache=None, geometry_cache=None):
    try:
        report_absolute = _safe_absolute(report_path)
        if report_absolute is None:
            return False
        if os.path.lexists(report_absolute):
            return False
        _reject_symlink_components(
            report_absolute, "report_path", allow_missing=True
        )
        _reject_symlink_components(
            os.path.dirname(report_absolute), "report parent"
        )
        report = str(Path(report_absolute).resolve(strict=False))
    except Exception:
        return False

    named_paths = (
        trace_prefix, data_root, initial_cwd, output_dir,
        runner_exit_code_file, receipt_path,
    )
    canonical = tuple(_canonical_fallback_path(path) for path in named_paths)
    if canonical[0] is None:
        return False
    (trace, data, project, output, runner_exit, receipt) = canonical

    inputs = tuple(
        path for path in (
            _canonical_fallback_path(value)
            for value in (
                input_artifacts
                if isinstance(input_artifacts, (list, tuple)) else ()
            )
        ) if path is not None
    )
    caches = tuple(
        path for path in (
            _canonical_fallback_path(value)
            for value in (base_cache, geometry_cache)
            if value is not None
        ) if path is not None
    )

    scratch = None
    if runtime_scratch_dir is not None:
        scratch = _canonical_fallback_path(runtime_scratch_dir)
        if scratch is None:
            return False

    protected_trees = tuple(
        path for path in (data, project) if path is not None
    ) + inputs + tuple(
        os.path.dirname(path) for path in inputs
    ) + caches
    if scratch is not None:
        protected_trees += (scratch,)
    if any(_paths_overlap(report, tree) for tree in protected_trees):
        return False
    if output is not None and _is_output_namespace(
            report, {"output_dir": output}):
        return False
    if (runner_exit is not None and _paths_overlap(report, runner_exit)):
        return False
    if receipt is not None and _paths_overlap(report, receipt):
        return False
    if _paths_overlap_trace_namespace(report, trace):
        return False
    return True


def _normalize_audit_report_path(config):
    report_path = config["report_path"]
    if os.path.lexists(report_path):
        raise UnsafeReportDestinationError(
            "report_path must be fresh"
        )
    try:
        _reject_symlink_components(
            report_path, "report_path", allow_missing=True
        )
        _reject_symlink_components(
            os.path.dirname(report_path), "report parent"
        )
    except ValueError as error:
        raise UnsafeReportDestinationError(str(error))
    report = str(Path(report_path).resolve(strict=False))
    protected_trees = (
        config["data_root"],
        config["initial_cwd"],
    ) + tuple(config["input_artifacts"]) + tuple(
        os.path.dirname(path) for path in config["input_artifacts"]
    ) + tuple(config.get("residual_cache_roots", ()))
    scratch = config.get("runtime_scratch_dir")
    if scratch is not None:
        protected_trees += (scratch,)
    for protected in protected_trees:
        if _paths_overlap(report, protected):
            raise UnsafeReportDestinationError(
                "report_path overlaps protected tree: {}".format(protected)
            )
    if _is_output_namespace(report, config):
        raise UnsafeReportDestinationError(
            "report_path overlaps output namespace"
        )
    if _paths_overlap(report, config["runner_exit_code_file"]):
        raise UnsafeReportDestinationError(
            "report_path overlaps runner_exit_code_file"
        )
    if _paths_overlap(report, config["receipt_path"]):
        raise UnsafeReportDestinationError(
            "report_path overlaps receipt_path"
        )
    if _paths_overlap_trace_namespace(report, config["trace_prefix"]):
        raise UnsafeReportDestinationError(
            "report_path overlaps trace namespace"
        )
    config["report_path"] = report


def _validate_exact_source_gate_paths(config):
    if not _is_exact_source_gate_smoke(config):
        return
    scratch = config["runtime_scratch_dir"]
    protected_trees = (
        config["data_root"],
        config["initial_cwd"],
    ) + tuple(config["input_artifacts"]) + tuple(
        os.path.dirname(path) for path in config["input_artifacts"]
    )
    control_paths = (
        ("trace_prefix", config["trace_prefix"]),
        ("runner_exit_code_file", config["runner_exit_code_file"]),
        ("receipt_path", config["receipt_path"]),
        ("report_path", config["report_path"]),
    )
    for label, path in (
            ("source-gate output_dir", config["output_dir"]),
            ("runtime_scratch_dir", scratch)):
        if _has_cache_directory_component(path):
            raise ValueError("{} overlaps a cache tree".format(label))
        for protected in protected_trees:
            if _paths_overlap(path, protected):
                raise ValueError(
                    "{} overlaps protected tree: {}".format(
                        label, protected
                    )
                )
    if _paths_overlap(scratch, config["output_dir"]):
        raise ValueError(
            "runtime_scratch_dir overlaps source-gate output_dir"
        )
    for control_label, control in control_paths:
        if _paths_overlap(scratch, control):
            error_type = (
                UnsafeReportDestinationError
                if control_label == "report_path" else ValueError
            )
            raise error_type(
                "runtime_scratch_dir overlaps control path: {}".format(
                    control
                )
            )
    source_controls = (
        ("trace_prefix", config["trace_prefix"]),
        ("runner_exit_code_file", config["runner_exit_code_file"]),
        ("report_path", config["report_path"]),
    )
    for label, control in source_controls:
        if _is_output_namespace(control, config):
            error_type = (
                UnsafeReportDestinationError
                if label == "report_path" else ValueError
            )
            raise error_type(
                "{} overlaps source-gate output namespace".format(label)
            )
        for protected in protected_trees:
            if _paths_overlap(control, protected):
                error_type = (
                    UnsafeReportDestinationError
                    if label == "report_path" else ValueError
                )
                raise error_type(
                    "{} overlaps protected tree: {}".format(
                        label, protected
                    )
                )
    report_path = config["report_path"]
    if _paths_overlap(report_path, scratch):
        raise UnsafeReportDestinationError(
            "report_path overlaps runtime_scratch_dir"
        )
    if _paths_overlap(report_path, config["runner_exit_code_file"]):
        raise UnsafeReportDestinationError(
            "report_path overlaps runner_exit_code_file"
        )
    if _paths_overlap_trace_namespace(
            report_path, config["trace_prefix"]):
        raise UnsafeReportDestinationError(
            "report_path overlaps trace namespace"
        )


def _normalize_config(trace_prefix, data_root, initial_cwd, input_artifacts,
                      output_dir, mode, runner_exit_code_file, receipt_path,
                      report_path, expected_initial_runner_argv,
                      runtime_scratch_dir=None, base_cache=None,
                      geometry_cache=None):
    artifact_paths = tuple(_absolute(path) for path in input_artifacts)
    if len(artifact_paths) != 3 or len(set(artifact_paths)) != 3:
        raise ValueError("exactly three distinct input artifacts are required")
    if mode not in ("smoke", "production", "residual", "hierarchical"):
        raise ValueError(
            "mode must be smoke, production, residual, or hierarchical"
        )

    data_root = _absolute(data_root)
    initial_cwd = _absolute(initial_cwd)
    output_dir = _absolute(output_dir)
    _reject_symlink_components(data_root, "data_root")
    _reject_symlink_components(initial_cwd, "initial_cwd")
    _reject_symlink_components(output_dir, "output_dir", allow_missing=True)
    if not os.path.isdir(data_root):
        raise ValueError("data_root must be an existing directory")
    if not os.path.isdir(initial_cwd):
        raise ValueError("initial_cwd must be an existing directory")
    if os.path.lexists(output_dir) and not os.path.isdir(output_dir):
        raise ValueError("output_dir must be a directory when it exists")
    data_root = str(Path(data_root).resolve(strict=True))
    initial_cwd = str(Path(initial_cwd).resolve(strict=True))
    output_dir = str(Path(output_dir).resolve(strict=False))

    artifacts = []
    for index, path in enumerate(artifact_paths):
        label = "input_artifact_{}".format(index + 1)
        _reject_symlink_components(path, label)
        artifacts.append(str(Path(path).resolve(strict=True)))
    artifacts = tuple(artifacts)
    if len(set(artifacts)) != 3:
        raise ValueError("input artifacts must have distinct canonical paths")

    interpreter_logical = _absolute(sys.executable)
    interpreter_resolved = str(
        Path(interpreter_logical).resolve(strict=True)
    )
    interpreter_identity, _interpreter_bytes = _stable_regular_snapshot(
        interpreter_resolved, "current Python interpreter"
    )
    if (not isinstance(expected_initial_runner_argv, (list, tuple))
            or not expected_initial_runner_argv
            or any(not isinstance(argument, str) or "\x00" in argument
                   for argument in expected_initial_runner_argv)):
        raise ValueError(
            "expected initial runner argv must be a nonempty string array"
        )
    expected_initial_runner_argv = tuple(expected_initial_runner_argv)
    expected_argv0 = expected_initial_runner_argv[0]
    if (not os.path.isabs(expected_argv0)
            or not os.path.exists(expected_argv0)
            or str(Path(expected_argv0).resolve(strict=True))
            != interpreter_resolved):
        raise ValueError(
            "expected initial runner argv[0] must resolve to this interpreter"
        )
    platform_probes = []
    for logical_path, argv in PLATFORM_PROBE_SPECS:
        resolved_path = str(Path(logical_path).resolve(strict=True))
        identity, _binary = _stable_regular_snapshot(
            resolved_path, "platform probe executable {}".format(logical_path)
        )
        platform_probes.append({
            "logical_path": logical_path,
            "resolved_path": resolved_path,
            "argv": list(argv),
            "sha256": identity["sha256"],
            "size": identity["size"],
        })
    residual_profile = mode == "residual" or _argv_names_residual_runner(
        expected_initial_runner_argv, initial_cwd
    )
    hierarchical_profile = (
        mode == "hierarchical" or _argv_names_hierarchical_runner(
            expected_initial_runner_argv, initial_cwd
        )
    )
    if residual_profile and hierarchical_profile:
        raise ValueError("residual and hierarchical audit profiles conflict")
    train_only_profile = residual_profile or hierarchical_profile
    expected_residual_sha256 = {}
    if train_only_profile:
        expected_residual_sha256 = dict(
            AUTHORITATIVE_RESIDUAL_ARTIFACT_SHA256
        )
        if (set(expected_residual_sha256)
                != {"backbone", "parent", "geometry"}
                or any(
                    not isinstance(value, str)
                    or re.match(r"^[0-9a-f]{64}$", value) is None
                    for value in expected_residual_sha256.values()
                )):
            raise ValueError(
                "authoritative reranker artifact SHA-256 contract is invalid"
            )
        if base_cache is None or geometry_cache is None:
            raise ValueError(
                "train-only reranker audit requires both train cache roots"
            )
        cache_roots = []
        for label, value in (
                ("base_cache", base_cache),
                ("geometry_cache", geometry_cache)):
            cache = _absolute(value)
            _reject_symlink_components(cache, label)
            if not os.path.isdir(cache):
                raise ValueError("{} must be an existing directory".format(label))
            cache = str(Path(cache).resolve(strict=True))
            if _deny_rule(cache, include_generic_tokens=True) is not None:
                raise ValueError("{} contains a forbidden validation component".format(label))
            cache_roots.append(cache)
        base_cache = cache_roots[0]
        geometry_cache = cache_roots[1]
    config = {
        "trace_prefix": _absolute(trace_prefix),
        "data_root": data_root,
        "initial_cwd": initial_cwd,
        "initial_interpreter": {
            "logical_path": interpreter_logical,
            "resolved_path": interpreter_resolved,
            "sha256": interpreter_identity["sha256"],
            "size": interpreter_identity["size"],
        },
        "initial_interpreter_usage": 0,
        "expected_initial_runner_argv": expected_initial_runner_argv,
        "platform_probe_executables": platform_probes,
        "platform_probe_usage": {
            logical_path: 0 for logical_path, _argv in PLATFORM_PROBE_SPECS
        },
        "source_gate_publication_writes": [],
        "opened_paths": [],
        "trace_pids": frozenset(),
        "input_artifacts": artifacts,
        "output_dir": output_dir,
        "runtime_scratch_dir": None,
        "mode": mode,
        "runner_exit_code_file": _absolute(runner_exit_code_file),
        "receipt_path": str(Path(_absolute(receipt_path)).resolve(strict=False)),
        "report_path": _absolute(report_path),
        "residual_profile": residual_profile,
        "hierarchical_profile": hierarchical_profile,
        "expected_residual_artifact_sha256": expected_residual_sha256,
        "base_cache": base_cache,
        "geometry_cache": geometry_cache,
        "residual_cache_roots": tuple(
            () if not train_only_profile else (base_cache, geometry_cache)
        ),
    }
    if _argv_names_source_gate_runner(
            config["expected_initial_runner_argv"], config["initial_cwd"]):
        _normalize_source_gate_control_paths(config)
    _validate_source_gate_command(config)
    _validate_residual_command(config)
    _validate_hierarchical_command(config)
    if runtime_scratch_dir is not None:
        scratch = _absolute(runtime_scratch_dir)
        _reject_symlink_components(scratch, "runtime_scratch_dir")
        if not os.path.isdir(scratch):
            raise ValueError(
                "runtime_scratch_dir must be an existing directory"
            )
        scratch = str(Path(scratch).resolve(strict=True))
        if stat.S_IMODE(os.lstat(scratch).st_mode) != 0o700:
            raise ValueError("runtime_scratch_dir mode must be 0700")
        if os.listdir(scratch):
            raise ValueError("runtime_scratch_dir must be empty")
        config["runtime_scratch_dir"] = scratch
    if (_is_exact_source_gate_smoke(config)
            and config["runtime_scratch_dir"] is None):
        raise ValueError(
            "exact source-gate smoke requires runtime_scratch_dir"
        )
    expected_name = (
        "smoke-receipt.json" if mode == "smoke"
        else "result-receipt.json"
        if mode in ("residual", "hierarchical")
        else "selection.json"
    )
    expected_receipt = os.path.join(config["output_dir"], expected_name)
    if config["receipt_path"] != expected_receipt:
        raise ValueError(
            "receipt/selection path must be the mode-specific output artifact"
        )
    _normalize_audit_report_path(config)
    _validate_exact_source_gate_paths(config)
    return config


def _finish_report(report):
    report["violations"] = _sort_violations(report["violations"])
    counts = report["counts"]
    counts["violations"] = len(report["violations"])
    for code in ("deny_path", "data_root_allow_miss", "uncertain_path"):
        counts[code] = sum(
            violation["code"] == code for violation in report["violations"]
        )


def audit(*, trace_prefix, data_root, initial_cwd, input_artifacts, output_dir,
          mode, runner_exit_code_file, receipt_path, report_path,
          expected_initial_runner_argv, runtime_scratch_dir=None,
          base_cache=None, geometry_cache=None):
    """Audit one completed runner trace and return the documented exit status."""
    fallback_report_safe = _fallback_report_is_safe(
        trace_prefix=trace_prefix,
        data_root=data_root,
        initial_cwd=initial_cwd,
        input_artifacts=input_artifacts,
        output_dir=output_dir,
        runner_exit_code_file=runner_exit_code_file,
        receipt_path=receipt_path,
        report_path=report_path,
        runtime_scratch_dir=runtime_scratch_dir,
        base_cache=base_cache,
        geometry_cache=geometry_cache,
    )
    report_path_fallback = _safe_absolute(report_path)
    fallback = {
        "trace_prefix": _safe_absolute(trace_prefix),
        "data_root": _safe_absolute(data_root),
        "initial_cwd": _safe_absolute(initial_cwd),
        "input_artifacts": _safe_path_list(input_artifacts),
        "output_dir": _safe_absolute(output_dir),
        "runtime_scratch_dir": _safe_absolute(runtime_scratch_dir),
        "mode": mode if isinstance(mode, str) else None,
        "base_cache": _safe_absolute(base_cache),
        "geometry_cache": _safe_absolute(geometry_cache),
        "runner_exit_code_file": _safe_absolute(runner_exit_code_file),
        "receipt_path": _safe_absolute(receipt_path),
        "report_path": report_path_fallback,
        "expected_initial_runner_argv": _safe_string_list(
            expected_initial_runner_argv
        ),
    }
    fatal = False
    try:
        config = _normalize_config(
            trace_prefix, data_root, initial_cwd, input_artifacts, output_dir,
            mode, runner_exit_code_file, receipt_path, report_path,
            expected_initial_runner_argv, runtime_scratch_dir,
            base_cache, geometry_cache,
        )
    except Exception as error:
        config = fallback
        report = _initial_report(config)
        report["violations"].append(_violation(
            "configuration_error", rule="cli-contract",
            detail=str(error),
        ))
        _finish_report(report)
        if (report_path_fallback is not None
                and fallback_report_safe
                and not isinstance(error, UnsafeReportDestinationError)):
            _write_report(report_path_fallback, report)
        return 2

    report = _initial_report(config)

    for label, path in zip(
            ("backbone", "parent", "geometry"), config["input_artifacts"]):
        try:
            identity, _content = _stable_regular_snapshot(
                path, "{} input artifact".format(label)
            )
            identity["label"] = label
            if _is_train_only_reranker_profile(config):
                expected_sha256 = config[
                    "expected_residual_artifact_sha256"
                ][label]
                identity.update({
                    "expected_sha256": expected_sha256,
                    "expected_mode": 0o444,
                    "sha256_matches": identity["sha256"] == expected_sha256,
                    "mode_matches": identity["mode"] == 0o444,
                })
                if (not identity["sha256_matches"]
                        or not identity["mode_matches"]):
                    fatal = True
                    report["violations"].append(_violation(
                        "protected_artifact_mismatch",
                        rule="authoritative-train-only-reranker-artifact",
                        detail=(
                            "{} expected sha256={} mode=0444; observed "
                            "sha256={} mode={:04o}"
                        ).format(
                            label,
                            expected_sha256,
                            identity["sha256"],
                            identity["mode"],
                        ),
                    ))
            report["inputs"]["input_artifacts"].append(identity)
        except TraceIntegrityError as error:
            fatal = True
            report["violations"].append(_violation(
                "input_integrity_error", rule="stable-input-artifact",
                detail=str(error),
            ))

    try:
        exit_identity, exit_content = _stable_regular_snapshot(
            config["runner_exit_code_file"], "runner exit-code file"
        )
        runner_exit = _parse_runner_exit(exit_content)
        report["inputs"]["runner_exit_code_file"].update({
            "sha256": exit_identity["sha256"],
            "exit_code": runner_exit,
        })
        if runner_exit is None:
            report["violations"].append(_violation(
                "runner_exit_invalid", rule="runner-exit-must-be-zero",
                detail="runner exit-code file is not one decimal integer",
            ))
        elif runner_exit != 0:
            report["violations"].append(_violation(
                "runner_exit_nonzero", rule="runner-exit-must-be-zero",
                detail="runner exited with status {}".format(runner_exit),
            ))
    except TraceIntegrityError as error:
        report["violations"].append(_violation(
            "runner_exit_invalid", rule="runner-exit-must-be-zero",
            detail=str(error),
        ))

    if not os.path.lexists(config["receipt_path"]):
        report["violations"].append(_violation(
            "receipt_missing", rule="receipt-exists-and-is-hashed",
            detail="mode-specific receipt/selection file does not exist",
        ))
    else:
        try:
            receipt_identity, _receipt = _stable_regular_snapshot(
                config["receipt_path"], "receipt/selection"
            )
            report["inputs"]["receipt"].update({
                "sha256": receipt_identity["sha256"],
                "size": receipt_identity["size"],
            })
            if _is_train_only_reranker_profile(config):
                try:
                    receipt_value = json.loads(_receipt.decode("utf-8"))
                except (UnicodeDecodeError, ValueError) as error:
                    report["violations"].append(_violation(
                        "receipt_invalid",
                        rule="train-only-receipt-no-validation",
                        detail=(
                            "train-only receipt is invalid JSON: {}"
                        ).format(error),
                    ))
                else:
                    validation_access = (
                        receipt_value.get("validation_data_accessed")
                        if isinstance(receipt_value, dict) else None
                    )
                    report["inputs"]["receipt"][
                        "validation_data_accessed"
                    ] = validation_access
                    if validation_access is not False:
                        report["validation_data_accessed"] = True
                        report["violations"].append(_violation(
                            "receipt_invalid",
                            rule="train-only-receipt-no-validation",
                            detail=(
                                "train-only receipt must state "
                                "validation_data_accessed=false"
                            ),
                        ))
                    if (config.get("hierarchical_profile")
                            and (not isinstance(receipt_value, dict)
                                 or receipt_value.get("schema")
                                 != "rec-hierarchical-result-receipt-v1")):
                        report["violations"].append(_violation(
                            "receipt_invalid",
                            rule="hierarchical-receipt-schema",
                            detail=(
                                "hierarchical completion receipt schema "
                                "is invalid"
                            ),
                        ))
            if _is_exact_source_gate_smoke(config):
                publication_errors = []
                try:
                    entries = sorted(os.listdir(config["output_dir"]))
                    parent_entries = sorted(os.listdir(
                        os.path.dirname(config["output_dir"])
                    ))
                    receipt_mode = stat.S_IMODE(os.lstat(
                        config["receipt_path"]
                    ).st_mode)
                except OSError as error:
                    publication_errors.append(
                        "cannot inspect source-gate output: {}".format(error)
                    )
                else:
                    if entries != ["smoke-receipt.json"]:
                        publication_errors.append(
                            "source-gate output directory is not receipt-only"
                        )
                    if receipt_mode != 0o444:
                        publication_errors.append(
                            "source-gate smoke receipt mode is not 0444"
                        )
                    stage_prefix = ".{}.staging-".format(
                        os.path.basename(config["output_dir"])
                    )
                    if any(
                            name.startswith(stage_prefix)
                            for name in parent_entries):
                        publication_errors.append(
                            "source-gate staging residue is present"
                        )
                if publication_errors:
                    report["violations"].append(_violation(
                        "receipt_invalid",
                        rule="source-gate-receipt-publication",
                        detail="; ".join(publication_errors),
                    ))
        except TraceIntegrityError as error:
            report["violations"].append(_violation(
                "receipt_invalid", rule="receipt-exists-and-is-hashed",
                detail=str(error),
            ))

    try:
        traces = _discover_traces(config["trace_prefix"])
        config["root_pid"] = traces[0][0]
        config["trace_pids"] = frozenset(pid for pid, _path in traces)
        report["inputs"]["root_pid"] = config["root_pid"]
        report["counts"]["trace_files"] = len(traces)
        for pid, path in traces:
            identity, content = _stable_regular_snapshot(
                path, "trace {}".format(path.name)
            )
            trace_input = {
                "path": identity["path"],
                "pid": pid,
                "sha256": identity["sha256"],
                "size": identity["size"],
            }
            report["inputs"]["traces"].append(trace_input)
            if not content:
                report["counts"]["empty_trace_files"] += 1
                continue
            try:
                text = content.decode("utf-8")
            except UnicodeDecodeError as error:
                raise TraceParseError(
                    "trace is not valid UTF-8: {}".format(error), pid=pid,
                    trace_file=str(path),
                )
            count, violations = _parse_trace(
                text, pid, str(path), config
            )
            report["counts"]["syscalls"] += count
            report["violations"].extend(violations)
        if report["counts"]["syscalls"] == 0:
            raise TraceParseError(
                "trace set contains no parseable selected syscall"
            )
        exec_contract = [
            (
                "initial Python interpreter",
                config["initial_interpreter_usage"],
            ),
            (
                "/bin/sh platform probe",
                config["platform_probe_usage"]["/bin/sh"],
            ),
            (
                "/usr/bin/uname platform probe",
                config["platform_probe_usage"]["/usr/bin/uname"],
            ),
        ]
        invalid_exec_contract = [
            "{} observed {} times".format(label, count)
            for label, count in exec_contract if count != 1
        ]
        if invalid_exec_contract:
            fatal = True
            report["violations"].append(_violation(
                "trace_parse_error",
                rule="exec-contract-exactly-once",
                detail=(
                    "required successful exec contract is incomplete: "
                    + "; ".join(invalid_exec_contract)
                ),
            ))
        if _is_exact_source_gate_smoke(config):
            writes = config["source_gate_publication_writes"]
            if (len(writes) != 1
                    or not _is_exact_source_gate_publication_write(
                        writes[0][0], writes[0][1], config
                    )):
                report["violations"].append(_violation(
                    "receipt_invalid",
                    rule="source-gate-publication-write-exactly-once",
                    detail=(
                        "source-gate publication must contain exactly one "
                        "bound staging receipt tempfile write"
                    ),
                ))
    except TraceIntegrityError as error:
        fatal = True
        report["violations"].append(_violation(
            "trace_integrity_error", rule="stable-regular-trace",
            detail=str(error),
        ))
    except TraceParseError as error:
        fatal = True
        report["violations"].append(_violation(
            "trace_parse_error", pid=error.pid,
            trace_file=error.trace_file, entry_line=error.line,
            resume_line=error.resume_line, timestamp=error.timestamp,
            syscall=error.syscall, rule="fail-closed-strace-parser",
            detail=str(error),
        ))

    opened_paths = config.get("opened_paths", [])
    report["opened_path_count"] = len(opened_paths)
    report["opened_path_sha256"] = hashlib.sha256(
        _canonical_json_bytes(opened_paths)
    ).hexdigest()
    destructive_path_calls = config.get("destructive_path_calls", [])
    report["destructive_path_call_count"] = len(destructive_path_calls)
    report["destructive_path_call_sha256"] = hashlib.sha256(
        _canonical_json_bytes(destructive_path_calls)
    ).hexdigest()
    if _is_train_only_reranker_profile(config) and any(
            violation.get("code") == "deny_path"
            for violation in report["violations"]):
        report["validation_data_accessed"] = True
    _finish_report(report)
    report["pass"] = not report["violations"]
    _write_report(config["report_path"], report)
    if fatal:
        return 2
    return 0 if report["pass"] else 1


def _json_string_array(value):
    try:
        parsed = json.loads(value)
    except (TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError(
            "expected runner argv must be JSON: {}".format(error)
        )
    if (not isinstance(parsed, list) or not parsed
            or any(not isinstance(argument, str) for argument in parsed)):
        raise argparse.ArgumentTypeError(
            "expected runner argv must be a nonempty JSON string array"
        )
    return parsed


def audit_residual_training_file_access(
        *, trace_prefix, data_root, initial_cwd, input_artifacts,
        base_cache, geometry_cache, output_dir, runner_exit_code_file,
        receipt_path, report_path, expected_initial_runner_argv,
        runtime_scratch_dir=None):
    """Audit the fixed train-only residual runner profile.

    This named wrapper keeps callers from accidentally selecting the legacy
    production allowlist while retaining one parser and one receipt schema.
    """
    return audit(
        trace_prefix=trace_prefix,
        data_root=data_root,
        initial_cwd=initial_cwd,
        input_artifacts=input_artifacts,
        output_dir=output_dir,
        mode="residual",
        runner_exit_code_file=runner_exit_code_file,
        receipt_path=receipt_path,
        report_path=report_path,
        expected_initial_runner_argv=expected_initial_runner_argv,
        runtime_scratch_dir=runtime_scratch_dir,
        base_cache=base_cache,
        geometry_cache=geometry_cache,
    )


def audit_hierarchical_training_file_access(
        *, trace_prefix, data_root, initial_cwd, input_artifacts,
        base_cache, geometry_cache, output_dir, runner_exit_code_file,
        receipt_path, report_path, expected_initial_runner_argv,
        runtime_scratch_dir=None):
    """Audit the fixed train-only hierarchical runner profile."""
    return audit(
        trace_prefix=trace_prefix,
        data_root=data_root,
        initial_cwd=initial_cwd,
        input_artifacts=input_artifacts,
        output_dir=output_dir,
        mode="hierarchical",
        runner_exit_code_file=runner_exit_code_file,
        receipt_path=receipt_path,
        report_path=report_path,
        expected_initial_runner_argv=expected_initial_runner_argv,
        runtime_scratch_dir=runtime_scratch_dir,
        base_cache=base_cache,
        geometry_cache=geometry_cache,
    )


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Audit REC finetuning strace file access without training."
    )
    parser.add_argument("--trace-prefix", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--initial-cwd", required=True)
    parser.add_argument(
        "--backbone-checkpoint", "--input-backbone", dest="backbone",
        required=True,
    )
    parser.add_argument(
        "--parent-reranker", "--input-parent", dest="parent", required=True,
    )
    parser.add_argument(
        "--geometry-reranker", "--input-geometry", dest="geometry",
        required=True,
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--runtime-scratch-dir")
    parser.add_argument(
        "--mode",
        choices=("smoke", "production", "residual", "hierarchical"),
        required=True,
    )
    parser.add_argument("--base-cache")
    parser.add_argument("--geometry-cache")
    parser.add_argument("--runner-exit-code-file", required=True)
    parser.add_argument(
        "--receipt-path", "--selection-path", "--receipt-or-selection-path",
        dest="receipt_path", required=True,
    )
    parser.add_argument(
        "--expected-initial-runner-argv-json",
        "--expected-initial-runner-argv",
        dest="expected_initial_runner_argv",
        type=_json_string_array,
        required=True,
    )
    parser.add_argument("--report-path", required=True)
    return parser.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    return audit(
        trace_prefix=args.trace_prefix,
        data_root=args.data_root,
        initial_cwd=args.initial_cwd,
        input_artifacts=(args.backbone, args.parent, args.geometry),
        output_dir=args.output_dir,
        runtime_scratch_dir=args.runtime_scratch_dir,
        mode=args.mode,
        runner_exit_code_file=args.runner_exit_code_file,
        receipt_path=args.receipt_path,
        report_path=args.report_path,
        expected_initial_runner_argv=args.expected_initial_runner_argv,
        base_cache=args.base_cache,
        geometry_cache=args.geometry_cache,
    )


if __name__ == "__main__":
    raise SystemExit(main())
