"""Apply the paper Table 4 no-PVI/no-TGS ablation to an isolated author tree."""
import argparse
import hashlib
import json
from pathlib import Path


def replace_once(text, old, new):
    assert text.count(old) == 1, old[:100]
    return text.replace(old, new, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    path = args.root / "models/pv_ground.py"
    text = path.read_text(encoding="utf-8")
    original_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    text = replace_once(text, "from .pv_backbone import pv_backbone\n", "")
    text = replace_once(
        text, "self.backbone_net = pv_backbone(model_cfg)",
        "self.backbone_net = Pointnet2Backbone(input_feature_dim=input_feature_dim, output_dim=d_model)",
    )
    text = replace_once(
        text, "self.gumbel = GumbelSampling(288, 256)",
        "self.points_obj_cls = PointsObjClsModule(d_model)\n"
        "        self.gsample_module = GeneralSamplingModule()",
    )
    start = text.index("        voxel_batch = self.backbone_net(inputs)")
    end = text.index("        # step 2. Text encoder", start)
    text = text[:start] + (
        "        point_batch = self.backbone_net(inputs['point_clouds'])\n"
        "        end_points['seed_inds'] = point_batch['fp2_inds']\n"
        "        end_points['seed_xyz'] = point_batch['fp2_xyz']\n"
        "        end_points['seed_features'] = point_batch['fp2_features'].transpose(1, 2).contiguous()\n\n"
    ) + text[end:]
    text = replace_once(text, "inputs['points'].device", "inputs['point_clouds'].device")
    point_expression = "inputs['points'][..., 1:4].view(inputs['batch_size'], -1, 3).contiguous()"
    assert text.count(point_expression) == 2
    text = text.replace(point_expression, "inputs['point_clouds'][..., :3].contiguous()")
    start = text.index("        # features: [B, V, F]")
    end = text.index("        return end_points", start)
    text = text[:start] + (
        "        logits = self.points_obj_cls(features)\n"
        "        end_points['seeds_obj_cls_logits'] = logits\n"
        "        indices = torch.topk(logits.sigmoid().squeeze(1), self.num_queries)[1].int()\n"
        "        xyz, features, indices = self.gsample_module(xyz, features, indices)\n"
        "        end_points['query_points_xyz'] = xyz\n"
        "        end_points['query_points_feature'] = features\n"
        "        end_points['query_points_sample_inds'] = indices\n"
    ) + text[end:]
    text = text[:text.index("\nclass GumbelSampling")]
    path.write_text(text, encoding="utf-8")
    # PV-Ground computed this inherited KPS loss but omitted it when using TGS.
    # Restore the MCLN/EDA coefficient for the reinstated Top-K classifier.
    loss_path = args.root / "models/losses.py"
    loss = loss_path.read_text(encoding="utf-8")
    loss = replace_once(loss, "    loss = (\n        1.0 / (num_decoder_layers + 1)",
                        "    loss = (\n        8 * query_points_generation_loss\n"
                        "        + 1.0 / (num_decoder_layers + 1)")
    loss_path.write_text(loss, encoding="utf-8")
    print(json.dumps({"original_model_sha256": original_hash,
           "clean_model_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
           "removed": ["sparse voxel backbone", "VSA", "Gumbel TGS"],
           "replacement": ["PointNet++", "supervised Top-K"],
           "retained": "author MCLN decoder, box/mask heads, remaining losses"}))


if __name__ == "__main__":
    main()
