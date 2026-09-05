"""Frozen train-scene identity and mechanical decision for the G0 pair."""

import hashlib
import json
import re


SALT = "MCLN-NR3D-VIEW-AUG-PAIR-V1-20260901"
CHECKPOINT_SHA = "76aa6cd49ca20a34e78509465f1185b1b9040e60807ad327d6b0876aeb6edba1"
VIEW_WORDS = frozenset((
    "front", "behind", "back", "left", "right", "facing", "leftmost",
    "rightmost", "looking", "across",
))
EXPECTED = {
    "fit": (404, 25768, 9589,
            "1cd8a48e901d5e4a67ba82185c576e4639697d86cd26d6665e6de698ea4f16ff"),
    "holdout": (107, 7151, 2718,
                "8ea5315099343e93a0513eecf4ff18c1f62f788153f1aa9c1962f320c8231967"),
}


def digest_ids(ids, ordered=False):
    values = list(ids)
    if len(values) != len(set(values)):
        raise ValueError("repeated source-row identity")
    payload = values if ordered else sorted(values)
    return hashlib.sha256(json.dumps(
        payload, separators=(",", ":")
    ).encode("utf-8")).hexdigest()


def is_view_dependent(text):
    return bool(VIEW_WORDS.intersection(re.findall(r"[a-z]+", text.lower())))


def split_rows(rows):
    partitions = {"fit": [], "holdout": []}
    for index, row in enumerate(rows):
        fold = int(hashlib.sha256(
            (SALT + "\0" + row["scan_id"]).encode("utf-8")
        ).hexdigest()[:8], 16) % 5
        partitions["holdout" if fold == 0 else "fit"].append(index)
    return partitions


def validate_census(rows, partitions):
    receipt = {}
    scene_sets = {}
    for name, ids in partitions.items():
        scenes = {rows[index]["scan_id"] for index in ids}
        scene_sets[name] = scenes
        view_count = sum(is_view_dependent(rows[index]["utterance"]) for index in ids)
        actual = (len(scenes), len(ids), view_count, digest_ids(ids))
        if actual != EXPECTED[name]:
            raise ValueError("{} census drift: {}".format(name, actual))
        receipt[name] = dict(zip(
            ("scenes", "rows", "view_dependent_rows", "identity_sha256"), actual
        ))
    if scene_sets["fit"] & scene_sets["holdout"] or len(rows) != 32919:
        raise ValueError("train scene split is not an exact disjoint partition")
    return receipt


def compare_rows(old, fixed):
    if [row["id"] for row in old] != [row["id"] for row in fixed]:
        raise ValueError("evaluation row order differs")
    if [row["view_dependent"] for row in old] != [row["view_dependent"] for row in fixed]:
        raise ValueError("evaluation view groups differ")
    results = {}
    for group in ("overall", "view_dependent", "view_independent"):
        indices = [i for i, row in enumerate(old) if group == "overall" or
                   row["view_dependent"] == (group == "view_dependent")]
        results[group] = {"rows": len(indices)}
        for threshold in ("025", "050"):
            key = "hit" + threshold
            fixes = sum(not old[i][key] and fixed[i][key] for i in indices)
            breaks = sum(old[i][key] and not fixed[i][key] for i in indices)
            results[group][threshold] = {
                "old_hits": sum(old[i][key] for i in indices),
                "fixed_hits": sum(fixed[i][key] for i in indices),
                "fixes": fixes, "breaks": breaks, "delta_hits": fixes - breaks,
            }
    passed = (results["overall"]["025"]["delta_hits"] > 0 and
              results["overall"]["050"]["delta_hits"] >= 0 and
              results["view_dependent"]["025"]["delta_hits"] > 0)
    return {"metrics": results, "scientific_gate_pass": passed}
