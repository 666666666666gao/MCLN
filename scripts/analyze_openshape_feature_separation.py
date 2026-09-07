"""Audit repeated detector-slot separation; this is not semantic or REC accuracy."""
import argparse
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    root = parser.parse_args().root
    receipt = json.loads((root / 'probe_receipt.json').read_bytes())
    features = np.load(root / 'object_embeddings.npz')['features']
    features = features / np.linalg.norm(features, axis=1, keepdims=True)
    first = {}
    repeated = []
    for item in receipt['objects']:
        if not item['encoded']:
            continue
        key = (item['scan_id'], item['slot'])
        if key in first:
            repeated.append(item)
        else:
            first[key] = item
    positive, negative, same_class, correct = [], [], [], []
    for item in repeated:
        key = (item['scan_id'], item['slot'])
        references = [v for k, v in first.items() if k[0] == key[0]]
        indices = [v['embedding_index'] for v in references]
        similarities = features[indices] @ features[item['embedding_index']]
        # Compare indices in the same matvec. Recomputing the self dot product
        # separately can create an incorrect rank through float32 roundoff.
        winner = references[int(np.argmax(similarities))]
        correct.append(winner['slot'] == item['slot'])
        positive.append(float(similarities[indices.index(first[key]['embedding_index'])]))
    values = list(first.values())
    for i, left in enumerate(values):
        for right in values[i + 1:]:
            if left['scan_id'] != right['scan_id']:
                continue
            cosine = float(features[left['embedding_index']] @ features[right['embedding_index']])
            negative.append(cosine)
            if left['detector_class'] == right['detector_class']:
                same_class.append(cosine)

    def stats(values):
        return dict(n=len(values), minimum=float(np.min(values)),
                    median=float(np.median(values)), maximum=float(np.max(values)))

    result = dict(same_slot_cosine=stats(positive), distinct_same_scene_slot_cosine=stats(negative),
                  distinct_same_scene_same_detector_class_cosine=stats(same_class),
                  repeat_nearest_same_scene_correct=sum(correct), repeat_count=len(correct),
                  unique_encoded_slots=len(first), new_rec_metrics=False,
                  interpretation='Same-slot sampling stability and distinct-slot separation only. Detector slots can duplicate physical objects. No language, GT identity or REC accuracy is measured.')
    (root / 'feature_separation_analysis.json').write_bytes(json.dumps(result, indent=2).encode() + b'\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
