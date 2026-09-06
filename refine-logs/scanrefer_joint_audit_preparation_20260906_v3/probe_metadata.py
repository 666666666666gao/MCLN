import copy, json, pytest
from scripts.audit_scanrefer_joint_readout_pair import assert_metadata_equal
import torch
from pathlib import Path
manifest = json.loads(Path('/root/autodl-tmp/mcln_scanrefer_joint_readout_pair_20260906_v1/input_manifest.json').read_text())
summary = {}
for name, item in manifest['artifacts'].items():
    if name == 'backbone':
        continue
    artifact = torch.load(item['path'], map_location='cpu')
    metadata = {key: value for key, value in artifact.items() if key != 'model_state_dict'}
    summary[name] = {key: type(value).__name__ for key, value in metadata.items()}
    assert_metadata_equal(copy.deepcopy(metadata), metadata)
    changed = copy.deepcopy(metadata)
    if name == 'v99':
        changed['normalization']['groups']['query_features']['mean'][0] += 1
    else:
        changed['feature_mean'][0] += 1
    with pytest.raises(AssertionError):
        assert_metadata_equal(changed, metadata)
print(json.dumps(summary, sort_keys=True))
