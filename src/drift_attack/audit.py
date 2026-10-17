"""Reproducible, standard-library-only audit of the official UCI 270 archive."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

from .dataset import GASES, load_archive
from .proxy import CUTOFF_QUANTILES, MIN_GROUP_SIZE, fit_reference


SOURCE_PAGE = "https://archive.ics.uci.edu/dataset/270/gas+sensor+array+drift+dataset+at+different+concentrations"
ARCHIVE_URL = "https://archive.ics.uci.edu/static/public/270/gas%2Bsensor%2Barray%2Bdrift%2Bdataset%2Bat%2Bdifferent%2Bconcentrations.zip"
APPROVED_ARCHIVE_SHA256 = "98fe3a30981a222dd4518fbcc3dddd45d5c0ce9b03ef6dc6fe5cf7a04cfbff5e"
MONTHS = {
    1: "1, 2", 2: "3, 4, 8, 9, 10", 3: "11, 12, 13", 4: "14, 15",
    5: "16", 6: "17, 18, 19, 20", 7: "21", 8: "22, 23",
    9: "24, 30", 10: "36",
}
DOCUMENTED_RANGES = {
    1: "10-600", 2: "10-300", 3: "50-1000", 4: "5-500",
    5: "12-1000", 6: "10-100",
}
PDF_NAMES = (
    "Drift or Attack AIot.pdf",
    "drift_or_attack_IEEE_report.pdf",
    "drift_or_attack_IEEE_turnitin.pdf",
)
SPLITS = {1: "reference", 2: "train", 3: "train", 4: "train",
          5: "validation", 6: "validation", 7: "test", 8: "test",
          9: "test", 10: "test"}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def audit(archive: Path, output: Path, project_root: Path) -> dict:
    archive_hash = file_sha256(archive)
    if archive_hash != APPROVED_ARCHIVE_SHA256:
        raise ValueError("archive SHA-256 differs from the inspected UCI download; inspect it before parsing")
    batches = load_archive(archive)
    reference = fit_reference(batches[1])
    summary = []
    by_gas = []
    all_concentrations = {gas: [] for gas in GASES}
    for batch in range(1, 11):
        rows = batches[batch]
        statuses = Counter(reference.status(row) for row in rows)
        scores = [score for row in rows if (score := reference.score(row)) is not None]
        record = {
            "batch": batch,
            "split": SPLITS[batch],
            "rows": len(rows),
            "distinct_concentrations": len({row.concentration for row in rows}),
            "min_concentration": min(row.concentration for row in rows),
            "max_concentration": max(row.concentration for row in rows),
            "eligible": statuses["eligible"],
            "no_exact_reference": statuses["no_exact_reference"],
            "sparse_reference": statuses["sparse_reference"],
        }
        for q in CUTOFF_QUANTILES:
            suffix = str(int(q * 100))
            record[f"within_q{suffix}"] = sum(score <= reference.cutoffs[q] for score in scores)
            record[f"outside_q{suffix}"] = sum(score > reference.cutoffs[q] for score in scores)
        summary.append(record)
        for gas, name in GASES.items():
            selected = [row for row in rows if row.gas == gas]
            all_concentrations[gas].extend(row.concentration for row in selected)
            by_gas.append({
                "batch": batch, "split": SPLITS[batch], "gas_code": gas,
                "gas": name, "rows": len(selected),
                "distinct_concentrations": len({row.concentration for row in selected}),
                "eligible": sum(reference.status(row) == "eligible" for row in selected),
                "no_exact_reference": sum(reference.status(row) == "no_exact_reference" for row in selected),
                "sparse_reference": sum(reference.status(row) == "sparse_reference" for row in selected),
            })
    pdf_hashes = {name: file_sha256(project_root / name) for name in PDF_NAMES}
    manifest = {
        "source_page": SOURCE_PAGE,
        "archive_url": ARCHIVE_URL,
        "archive_sha256": archive_hash,
        "archive_bytes": archive.stat().st_size,
        "zip_members": [f"batch{i}.dat" for i in range(1, 11)],
        "pdf_sha256": pdf_hashes,
        "feature_count_per_row": 128,
        "total_rows": sum(len(rows) for rows in batches.values()),
        "reference_groups_all": len(reference.group_sizes),
        "reference_groups_eligible": len(reference.centers),
        "reference_rows_for_leave_one_out": len(reference.leave_one_out_scores),
        "cutoffs": {str(q): reference.cutoffs[q] for q in CUTOFF_QUANTILES},
        "concentration_by_gas": {
            str(gas): {"gas": GASES[gas], "min": min(values), "max": max(values),
                       "distinct": len(set(values))}
            for gas, values in all_concentrations.items()
        },
        "splits": {str(batch): split for batch, split in SPLITS.items()},
    }
    output.mkdir(parents=True, exist_ok=True)
    _write_csv(output / "batch_summary.csv", summary)
    _write_csv(output / "batch_gas_summary.csv", by_gas)
    (output / "source_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    (output / "data_audit.md").write_text(_report(manifest, summary, by_gas))
    return manifest


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _report(manifest: dict, summary: list[dict], by_gas: list[dict]) -> str:
    test = [row for row in summary if row["split"] == "test"]
    test_total = sum(row["rows"] for row in test)
    test_eligible = sum(row["eligible"] for row in test)
    gas4_eligible = sum(row["eligible"] for row in by_gas if row["gas_code"] == 4)
    lines = [
        "# UCI 270 data audit",
        "",
        "## Source and observed structure",
        "",
        f"- Official source page: {SOURCE_PAGE}",
        f"- Download URL: {ARCHIVE_URL}",
        f"- Downloaded archive SHA-256: `{manifest['archive_sha256']}`; bytes: {manifest['archive_bytes']:,}. The audit checks this recorded digest before parsing. UCI does not publish an independent digest on the source page.",
        "- Archive validation: exactly `batch1.dat` through `batch10.dat`, no extra or unsafe paths, ZIP CRC check passed. Members were read directly; no extraction occurred.",
        f"- Observed rows: {manifest['total_rows']:,}. Every row has a gas code, concentration, and exactly 128 consecutive indexed, finite numeric features.",
        "- Observed gas codes: 1 Ethanol, 2 Ethylene, 3 Ammonia, 4 Acetaldehyde, 5 Acetone, 6 Toluene (names from UCI documentation).",
        "- The archive verifies 128 feature values per row. The interpretation as eight features from each of 16 physical sensors comes from UCI documentation, not a separate sensor ID field in the files.",
        "- File line number is available, but no per-record timestamp, attack onset, or verified drift/attack label was found. Do not infer a time interval or true within-batch chronology from line order.",
        "- UCI documents the following batch-level month groups: " + "; ".join(f"B{b}: {MONTHS[b]}" for b in range(1, 11)) + ". These months are metadata, not row-level fields.",
        "",
        "## Fixed split and reference-envelope feasibility rule",
        "",
        "- B1 is the reference; B2-4 training; B5-6 validation; B7-10 held-out test. No random split or model fitting was performed.",
        f"- Match only exact `(gas code, concentration)` pairs represented by at least {MIN_GROUP_SIZE} B1 records. A different concentration is unmatched; groups with fewer than {MIN_GROUP_SIZE} B1 records are reported as sparse. No nearest-neighbor imputation is used.",
        "- For each feature, compute the sample standard deviation on all B1 records only. Floor a zero/near-zero scale at `1e-12`. For each eligible pair, compute the per-feature median on B1 records in that pair.",
        "- Distance is the mean of 128 absolute feature deviations from the pair median, each divided by its B1 feature scale. Gas and concentration are matching keys, not distance inputs.",
        "- For each eligible B1 record, recompute its pair median without that record and calculate a leave-one-out distance. Pool these B1-only distances and take linearly interpolated 90th, 95th, and 99th percentiles. These are prespecified sensitivity alternatives; no validation or test data choose a cutoff.",
        "- Counts below show `within` and `outside` the B1 reference envelope. They are candidate operational states, **not verified normal or sensor drift labels**. B1 counts use full reference medians and are in-sample, so they must not be read as calibrated false-alarm estimates.",
        "- Assumptions: B1 is a usable clean reference; exact gas/concentration identity would be known at decision time; within-pair B1 spread is representative; and a feature-space departure may reflect drift. These assumptions are unverified. Gas and concentration can be known in the laboratory but might not be known in deployment.",
        "",
        "## Counts by batch",
        "",
        "| Batch | Split | Rows | Concentrations: distinct (min-max ppmv) | Eligible | No exact B1 pair | Sparse B1 pair | Within/Outside q90 | Within/Outside q95 | Within/Outside q99 |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        concentration = f"{row['distinct_concentrations']} ({row['min_concentration']:g}-{row['max_concentration']:g})"
        lines.append(f"| {row['batch']} | {row['split']} | {row['rows']} | {concentration} | {row['eligible']} | {row['no_exact_reference']} | {row['sparse_reference']} | {row['within_q90']}/{row['outside_q90']} | {row['within_q95']}/{row['outside_q95']} | {row['within_q99']}/{row['outside_q99']} |")
    lines += [
        "",
        "These are observed record counts, not class-label counts. `batch_summary.csv` and `batch_gas_summary.csv` give machine-readable summaries. Neither file assigns row-level proxy labels.",
        "",
        "### Gas records by batch (observed)",
        "",
        "| Batch | Ethanol | Ethylene | Ammonia | Acetaldehyde | Acetone | Toluene |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for batch in range(1, 11):
        counts = [next(row["rows"] for row in by_gas if row["batch"] == batch and row["gas_code"] == gas)
                  for gas in GASES]
        lines.append(f"| {batch} | " + " | ".join(str(value) for value in counts) + " |")
    lines += [
        "",
        "## Cutoffs and concentration coverage",
        "",
        f"- Eligible B1 groups: {manifest['reference_groups_eligible']} of {manifest['reference_groups_all']}; B1 leave-one-out scores: {manifest['reference_rows_for_leave_one_out']}.",
        "- B1-only cutoffs: " + ", ".join(f"q{int(float(q)*100)} = {value:.6g}" for q, value in manifest["cutoffs"].items()) + ".",
        "",
        "| Gas | UCI page's stated interval (ppmv) | Observed min-max (ppmv) | Distinct observed levels |",
        "|---|---:|---:|---:|",
    ]
    for gas in GASES:
        item = manifest["concentration_by_gas"][str(gas)]
        lines.append(f"| {item['gas']} | {DOCUMENTED_RANGES[gas]} | {item['min']:g}-{item['max']:g} | {item['distinct']} |")
    lines += [
        "",
        "Several observed extremes fall outside the intervals stated on the UCI page. This is a documentation/data discrepancy, not a reason to discard those rows; the parser uses the observed values and reports them without correction.",
        "",
        "## Feasibility decision and limits",
        "",
        f"- Strict B1 reference coverage in held-out B7-10 is {test_eligible:,}/{test_total:,} ({test_eligible/test_total:.1%}); {test_total-test_eligible:,} records cannot receive either reference-envelope state under this rule.",
        f"- Acetaldehyde (gas 4) has {gas4_eligible} eligible records across all batches because its B1 exact concentration groups have too few replicates.",
        "- Therefore this rule does **not** support a full six-gas, all-record three-class evaluation. Even for eligible records, outside-envelope status does not establish sensor aging; changes in measurement conditions and other causes remain possible. No normal/drift/attack labels or attack records were produced.",
        "- Honest alternatives for a later approved phase: evaluate reference departure only on explicitly covered gas/concentration strata, while reporting exclusions; obtain independent drift annotations or controlled recalibration data; or build a clearly synthetic three-way simulation and state that its drift class is simulated rather than natural.",
        "- Detection delay in elapsed time is unsupported: the archive has no per-record timestamps or verified attack start. ZIP file order and line numbers are not a measured stream.",
        "",
        "## Submitted PDF integrity",
        "",
    ]
    for name, digest in manifest["pdf_sha256"].items():
        lines.append(f"- `{name}`: `{digest}`")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = audit(args.archive, args.output, Path.cwd())
    print(f"Audited {manifest['total_rows']} rows; wrote {args.output / 'data_audit.md'}")


if __name__ == "__main__":
    main()
