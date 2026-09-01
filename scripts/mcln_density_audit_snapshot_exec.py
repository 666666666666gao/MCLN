"""Drop Linux capabilities, enter the reviewed snapshot, then exec audit."""

from __future__ import print_function

import argparse
import ctypes
import errno
import os
import pathlib
import stat
import sys


PR_CAPBSET_DROP = 24
PR_CAP_AMBIENT = 47
PR_CAP_AMBIENT_CLEAR_ALL = 4
PR_SET_NO_NEW_PRIVS = 38
PR_SET_SECUREBITS = 28
LANDLOCK_CREATE_RULESET_VERSION = 1
LANDLOCK_RULE_PATH_BENEATH = 1
SECBIT_NOROOT = 1
SECBIT_NOROOT_LOCKED = 2
SECBIT_NO_SETUID_FIXUP = 4
SECBIT_NO_SETUID_FIXUP_LOCKED = 8
LINUX_CAPABILITY_VERSION_3 = 0x20080522
SNAPSHOT_OWNER_UID = 65532
SNAPSHOT_OWNER_GID = 65532
SYS_LANDLOCK_CREATE_RULESET = 444
SYS_LANDLOCK_ADD_RULE = 445
SYS_LANDLOCK_RESTRICT_SELF = 446
ACCESS_FS_WRITE_FILE = 1 << 1
ACCESS_FS_REMOVE_DIR = 1 << 4
ACCESS_FS_REMOVE_FILE = 1 << 5
ACCESS_FS_MAKE_CHAR = 1 << 6
ACCESS_FS_MAKE_DIR = 1 << 7
ACCESS_FS_MAKE_REG = 1 << 8
ACCESS_FS_MAKE_SOCK = 1 << 9
ACCESS_FS_MAKE_FIFO = 1 << 10
ACCESS_FS_MAKE_BLOCK = 1 << 11
ACCESS_FS_MAKE_SYM = 1 << 12
HANDLED_WRITE_ACCESS = (
    ACCESS_FS_WRITE_FILE
    | ACCESS_FS_REMOVE_DIR
    | ACCESS_FS_REMOVE_FILE
    | ACCESS_FS_MAKE_CHAR
    | ACCESS_FS_MAKE_DIR
    | ACCESS_FS_MAKE_REG
    | ACCESS_FS_MAKE_SOCK
    | ACCESS_FS_MAKE_FIFO
    | ACCESS_FS_MAKE_BLOCK
    | ACCESS_FS_MAKE_SYM
)


class RulesetAttr(ctypes.Structure):
    _fields_ = [("handled_access_fs", ctypes.c_uint64)]


class PathBeneathAttr(ctypes.Structure):
    _fields_ = [
        ("allowed_access", ctypes.c_uint64),
        ("parent_fd", ctypes.c_int32),
    ]


class CapHeader(ctypes.Structure):
    _fields_ = [
        ("version", ctypes.c_uint32),
        ("pid", ctypes.c_int),
    ]


class CapData(ctypes.Structure):
    _fields_ = [
        ("effective", ctypes.c_uint32),
        ("permitted", ctypes.c_uint32),
        ("inheritable", ctypes.c_uint32),
    ]


LIBC = ctypes.CDLL(None, use_errno=True)
LIBC.prctl.argtypes = [
    ctypes.c_int,
    ctypes.c_ulong,
    ctypes.c_ulong,
    ctypes.c_ulong,
    ctypes.c_ulong,
]
LIBC.prctl.restype = ctypes.c_int
LIBC.syscall.restype = ctypes.c_long
LIBC.capset.argtypes = [
    ctypes.POINTER(CapHeader),
    ctypes.c_void_p,
]
LIBC.capset.restype = ctypes.c_int


def checked_prctl(option, argument):
    result = LIBC.prctl(option, argument, 0, 0, 0)
    if result != 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))


def checked_syscall(number, *arguments):
    result = LIBC.syscall(number, *arguments)
    if result < 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))
    return result


def restrict_writes(allowed_paths):
    abi = checked_syscall(
        SYS_LANDLOCK_CREATE_RULESET, 0, 0, LANDLOCK_CREATE_RULESET_VERSION
    )
    if abi < 1:
        raise RuntimeError("Landlock ABI v1 is unavailable")
    ruleset_attr = RulesetAttr(HANDLED_WRITE_ACCESS)
    ruleset_fd = checked_syscall(
        SYS_LANDLOCK_CREATE_RULESET,
        ctypes.byref(ruleset_attr),
        ctypes.sizeof(ruleset_attr),
        0,
    )
    try:
        for path in allowed_paths:
            resolved = pathlib.Path(path).resolve()
            if not resolved.is_dir():
                raise ValueError(
                    "allowed write root is not a directory: {}".format(resolved)
                )
            parent_fd = os.open(
                str(resolved), os.O_PATH | os.O_CLOEXEC | os.O_DIRECTORY
            )
            try:
                rule = PathBeneathAttr(HANDLED_WRITE_ACCESS, parent_fd)
                checked_syscall(
                    SYS_LANDLOCK_ADD_RULE,
                    ruleset_fd,
                    LANDLOCK_RULE_PATH_BENEATH,
                    ctypes.byref(rule),
                    0,
                )
            finally:
                os.close(parent_fd)
        checked_prctl(PR_SET_NO_NEW_PRIVS, 1)
        checked_syscall(SYS_LANDLOCK_RESTRICT_SELF, ruleset_fd, 0)
    finally:
        os.close(ruleset_fd)


def drop_all_capabilities():
    securebits = (
        SECBIT_NOROOT
        | SECBIT_NOROOT_LOCKED
        | SECBIT_NO_SETUID_FIXUP
        | SECBIT_NO_SETUID_FIXUP_LOCKED
    )
    checked_prctl(PR_SET_SECUREBITS, securebits)
    last_capability = int(
        pathlib.Path("/proc/sys/kernel/cap_last_cap").read_text(
            encoding="ascii"
        ).strip()
    )
    for capability in range(last_capability + 1):
        checked_prctl(PR_CAPBSET_DROP, capability)
    try:
        checked_prctl(PR_CAP_AMBIENT, PR_CAP_AMBIENT_CLEAR_ALL)
    except OSError as error:
        if error.errno != errno.EINVAL:
            raise
    header = CapHeader(version=LINUX_CAPABILITY_VERSION_3, pid=0)
    data = (CapData * 2)()
    if LIBC.capset(ctypes.byref(header), ctypes.byref(data)) != 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))
    checked_prctl(PR_SET_NO_NEW_PRIVS, 1)


def capability_status():
    observed = {}
    with open("/proc/self/status", "r", encoding="ascii") as handle:
        for line in handle:
            name, separator, value = line.partition(":")
            if separator and name in {
                    "CapInh", "CapPrm", "CapEff", "CapBnd", "CapAmb",
                    "NoNewPrivs"}:
                observed[name] = value.strip()
    return observed


def verify_capability_status():
    observed = capability_status()
    expected_zero = "0000000000000000"
    for name in ("CapInh", "CapPrm", "CapEff", "CapBnd", "CapAmb"):
        if observed.get(name) != expected_zero:
            raise RuntimeError("capability drop failed for {}".format(name))
    if observed.get("NoNewPrivs") != "1":
        raise RuntimeError("no-new-privileges was not enabled")


def close_nonstandard_file_descriptors():
    descriptors = []
    for name in os.listdir("/proc/self/fd"):
        try:
            descriptor = int(name)
        except ValueError:
            continue
        if descriptor >= 3:
            descriptors.append(descriptor)
    for descriptor in descriptors:
        try:
            os.close(descriptor)
        except OSError:
            pass


def validate_roots(code_root, write_root):
    raw_code_root = pathlib.Path(code_root)
    raw_write_root = pathlib.Path(write_root)
    if raw_code_root.is_symlink() or raw_write_root.is_symlink():
        raise ValueError("snapshot executor roots must not be symlinks")
    code_root = raw_code_root.resolve()
    write_root = raw_write_root.resolve()
    if not code_root.is_dir():
        raise ValueError("code snapshot root is not a real directory")
    if not write_root.is_dir():
        raise ValueError("runtime output root is not a real directory")
    entrypoint = code_root / "train_dist_mod.py"
    info = entrypoint.lstat()
    if not stat.S_ISREG(info.st_mode) or entrypoint.is_symlink():
        raise ValueError("snapshot training entrypoint is not regular")
    if stat.S_IMODE(info.st_mode) != 0o444:
        raise ValueError("snapshot training entrypoint is not mode 0444")
    root_info = code_root.lstat()
    if (
            root_info.st_uid != SNAPSHOT_OWNER_UID
            or root_info.st_gid != SNAPSHOT_OWNER_GID
            or info.st_uid != SNAPSHOT_OWNER_UID
            or info.st_gid != SNAPSHOT_OWNER_GID):
        raise ValueError("code snapshot is not owned by the isolated owner")
    return code_root, write_root, entrypoint


def run_verify_only(code_root, entrypoint, write_root):
    try:
        descriptor = os.open(str(entrypoint), os.O_WRONLY | os.O_APPEND)
    except PermissionError:
        pass
    else:
        os.close(descriptor)
        raise RuntimeError("capability-free process can write code snapshot")
    try:
        entrypoint.chmod(0o644)
    except PermissionError:
        pass
    else:
        entrypoint.chmod(0o444)
        raise RuntimeError("capability-free process can chmod code snapshot")
    directory_probe = code_root / ".density_directory_probe"
    try:
        directory_probe.mkdir()
    except PermissionError:
        pass
    else:
        directory_probe.rmdir()
        raise RuntimeError("capability-free process can create in code snapshot")
    parent_probe = code_root.with_name(code_root.name + ".density_parent_probe")
    if parent_probe.exists():
        raise RuntimeError("parent rename probe already exists")
    try:
        code_root.rename(parent_probe)
    except PermissionError:
        pass
    else:
        parent_probe.rename(code_root)
        raise RuntimeError("restricted process can rename code snapshot")
    probe = write_root / ".density_capability_probe"
    descriptor = os.open(
        str(probe), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600
    )
    try:
        os.write(descriptor, b"capability_drop_verified\n")
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    probe.unlink()
    print("capability_drop=pass snapshot_write=denied runtime_write=allowed")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--code-root", required=True)
    parser.add_argument("--write-root", required=True)
    parser.add_argument("--allow-write", action="append", default=[])
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    code_root, write_root, entrypoint = validate_roots(
        args.code_root, args.write_root
    )
    os.chdir(str(code_root))
    close_nonstandard_file_descriptors()
    allowed_write_roots = [write_root]
    allowed_write_roots.extend(pathlib.Path(path) for path in args.allow_write)
    allowed_write_roots.extend(
        pathlib.Path(path) for path in ("/tmp", "/dev", "/proc")
    )
    restrict_writes(allowed_write_roots)
    drop_all_capabilities()
    verify_capability_status()
    if args.verify_only:
        if args.command:
            raise ValueError("verify-only mode does not accept a command")
        run_verify_only(code_root, entrypoint, write_root)
        return
    command = list(args.command)
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        raise ValueError("formal audit command is missing")
    os.execv(command[0], command)


if __name__ == "__main__":
    main()
