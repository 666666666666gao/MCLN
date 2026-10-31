"""Exercise the actual replacement backbone and its cross-encoder layout."""
import json
from pathlib import Path

import torch
from models.backbone_module import Pointnet2Backbone

torch.manual_seed(111)
model = Pointnet2Backbone(input_feature_dim=3, output_dim=288).cuda().eval()
with torch.no_grad():
    result = model(torch.randn(2, 50000, 6, device="cuda"))
features = result["fp2_features"].transpose(1, 2).contiguous()
assert features.shape == (2, 1024, 288)
assert result["fp2_xyz"].shape == (2, 1024, 3)
assert torch.isfinite(features).all().item()
witness = {
    "seed": 111, "input_shape": [2, 50000, 6],
    "pointnet_feature_shape": list(result["fp2_features"].shape),
    "cross_encoder_feature_shape": list(features.shape),
    "finite": True, "gpu": torch.cuda.get_device_name(0),
    "peak_bytes": torch.cuda.max_memory_allocated(),
}
Path("backbone_kernel_witness.json").write_text(json.dumps(witness, indent=2) + "\n")
print(json.dumps(witness))
