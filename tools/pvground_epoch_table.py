"""Build a provenance-preserving table from verified remote metric exports."""

import argparse
import csv
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshots", type=Path, nargs="+", required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    args = parser.parse_args()
    records = []
    sources = []
    seen = set()
    pending = []
    selected_rows = []
    for path in args.snapshots:
        raw = path.read_bytes()
        snapshot = json.loads(raw)
        sources.append((path.name, snapshot["observed_at"],
                        hashlib.sha256(raw).hexdigest()))
        for run in snapshot["runs"]:
            dataset, run_id = run["dataset"], run["run"]
            assert (dataset, run_id) not in seen
            seen.add((dataset, run_id))
            metrics = run["metrics"]
            assert len({row["epoch"] for row in metrics}) == len(metrics)
            trained = [row for row in metrics if row["epoch"] > 0]
            priority = ("hits050", "hits025") if dataset == "scanrefer" else (
                "hits025", "hits050")
            selected = max(trained, key=lambda row: tuple(row[key] for key in priority)) if trained else None
            if not metrics:
                pending.append(f"{dataset}/{run_id}: no complete evaluation.")
            if run["current_run"]:
                selected_rows.append((dataset, run_id, selected))
            for row in sorted(metrics, key=lambda row: row["epoch"]):
                assert row["rows"] == {
                    "scanrefer": 9508, "nr3d": 7899, "sr3d": 17726}[dataset]
                assert row["primary_score"] == "last/bbf" and row["same_complete_model"]
                records.append(dict(
                    dataset=dataset, run=run_id,
                    scope="current" if run["current_run"] else "historical",
                    epoch=row["epoch"], trained=row["epoch"] > 0,
                    rows=row["rows"], hits025=row["hits025"], hits050=row["hits050"],
                    acc025_percent=f'{100 * row["hits025"] / row["rows"]:.6f}',
                    acc050_percent=f'{100 * row["hits050"] / row["rows"]:.6f}',
                    selected_current=bool(run["current_run"] and selected == row),
                    primary_score=row["primary_score"],
                    same_complete_model=row["same_complete_model"],
                    source=run["source"], source_sha256=run["source_sha256"],
                    observed_at=snapshot["observed_at"],
                ))
    order = {"scanrefer": 0, "nr3d": 1, "sr3d": 2}
    records.sort(key=lambda row: (order[row["dataset"]], row["run"], row["epoch"]))
    output_csv = args.output_prefix.with_suffix(".csv")
    with output_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    text = [
        "# PV-Ground Complete Epoch Metrics", "",
        "Only full native last/bbf semantic-alignment evaluations are included.",
        "Epoch0 is untrained initialization. Historical runs are separate experiments;",
        "interrupted epochs have no result. Same-row accuracies use the same model.",
        "ScanRefer selects trained Acc050 first; NR3D/SR3D select trained Acc025 first.",
        "The other threshold breaks ties. Selection here does not prove weight retention.",
        "Author-baseline reproduction results are documented separately in",
        "PV_GROUND_MODULE_FT_2026-10-30.md.", "",
        "## Current Trained Selection", "",
        "| Dataset | Run | Epoch | Acc025 | Acc050 |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for dataset, run_id, selected in sorted(selected_rows, key=lambda item: order[item[0]]):
        if selected is None:
            text.append(f"| {dataset} | {run_id} | pending | - | - |")
        else:
            text.append(
                f'| {dataset} | {run_id} | {selected["epoch"]} | '
                f'{100 * selected["hits025"] / selected["rows"]:.6f}% | '
                f'{100 * selected["hits050"] / selected["rows"]:.6f}% |')
    text.extend([
        "", "## All Complete Epochs", "",
        "| Dataset | Run | Scope | Epoch | Rows | Hits025 | Hits050 | Acc025 | Acc050 |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for row in records:
        epoch = f'{row["epoch"]}' if row["trained"] else "0 (init)"
        text.append(
            f'| {row["dataset"]} | {row["run"]} | {row["scope"]} | {epoch} | '
            f'{row["rows"]} | {row["hits025"]} | {row["hits050"]} | '
            f'{row["acc025_percent"]}% | {row["acc050_percent"]}% |')
    text.extend(["", "## Source Snapshots", "",
                 "| File | Observed UTC | SHA256 |", "| --- | --- | --- |"])
    text.extend(f"| {name} | {observed} | {digest} |" for name, observed, digest in sources)
    text.extend(["", *pending, ""])
    args.output_prefix.with_suffix(".md").write_text("\n".join(text), encoding="utf-8")
    print(json.dumps(dict(rows=len(records), runs=len(seen), output=str(output_csv))))


if __name__ == "__main__":
    main()
