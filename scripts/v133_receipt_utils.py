"""Small POSIX helpers for immutable, no-clobber V133 receipts."""

from __future__ import print_function

import json
import os
import stat
import tempfile
from pathlib import Path


def fsync_directory(path):
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(str(path), flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_write_new_json(payload, output):
    """Publish one read-only JSON inode without replacing an existing path."""
    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    fsync_directory(output.parent)
    descriptor, temporary = tempfile.mkstemp(
        prefix=output.name + ".tmp.", dir=str(output.parent)
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fchmod(
                handle.fileno(),
                stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH,
            )
            os.fsync(handle.fileno())
        # A same-directory hard link is an atomic create-if-absent operation.
        # Unlike os.replace, it cannot overwrite a concurrent receipt.
        os.link(temporary, str(output))
        fsync_directory(output.parent)
        os.unlink(temporary)
        fsync_directory(output.parent)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
            fsync_directory(output.parent)
    return output
