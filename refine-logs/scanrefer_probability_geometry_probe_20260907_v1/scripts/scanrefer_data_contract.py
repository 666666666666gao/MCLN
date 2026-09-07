"""Bind current ScanRefer runs to the explicitly supplied superpoint inputs."""

import hashlib
from pathlib import Path


def set_scanrefer_data_root(command, data_root):
    # Joint3DDataset concatenates data_path + 'roberta-base/' directly.
    assert str(data_root).endswith('/')
    result = list(command)
    result[result.index('--data_root') + 1] = str(data_root)
    return result


def verify_scanrefer_superpoints(data_root, split, expected_files):
    # The historical command still points to mixed superpoints; 206/312 val
    # files differ from the mesh-derived inputs used by protected V99.
    assert len(expected_files) == {'train': 1201, 'val': 312}[split]
    directory = Path(data_root) / 'superpoints' / split
    for name, expected_sha in expected_files.items():
        path = directory / name
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected_sha, str(path)
    return {'split': split, 'files_verified': len(expected_files),
            'resolved_directory': str(directory.resolve())}
