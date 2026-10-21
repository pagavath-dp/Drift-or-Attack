"""Read-only checks of recorded inputs, model reports, and final-result artifacts."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

from scripts.build_final_results import (EXTERNAL_INPUTS, INPUTS, derive_tables,
                                         input_hash_status, inspect_source_hashes,
                                         load_and_validate)


def verify(reports: Path) -> int:
    try:
        statuses = inspect_source_hashes(reports)
    except (OSError, KeyError, ValueError) as error:
        print(f"Saved provenance unavailable: {error}", file=sys.stderr)
        return 1
    internal = set(INPUTS) | {"experimental_protocol.md"}
    failed = False
    for name, status in statuses.items():
        print(f"{status}: {name}")
        if status == "mismatch" or (name in internal and status == "missing"):
            failed = True
    if failed:
        print("Verification failed: a recorded input is missing or differs from its historical hash.",
              file=sys.stderr)
        return 1
    try:
        frozen = json.loads((reports.parent / "docs/frozen_report_hashes.json").read_text(encoding="utf-8"))
        for relative, expected in frozen.items():
            path = reports / Path(relative).relative_to("reports")
            status = input_hash_status(path, expected)
            if status not in ("exact", "crlf_checkout"):
                raise ValueError(f"frozen artifact {relative}: {status}")
        actual_files = {"reports/" + path.relative_to(reports).as_posix()
                        for path in reports.rglob("*") if path.is_file()}
        if actual_files != set(frozen):
            raise ValueError("frozen artifact inventory differs: " +
                             ", ".join(sorted(actual_files ^ set(frozen))))
        data, batches, provenance = load_and_validate(reports, require_external_inputs=False)
        output = reports / "final_results"
        if json.loads((output / "source_hashes.json").read_text(encoding="utf-8")) != provenance:
            raise ValueError("saved provenance differs from validated inputs")
        for name, rows in derive_tables(data, batches).items():
            with (output / "tables" / f"{name}.csv").open(newline="", encoding="utf-8") as stream:
                actual = list(csv.DictReader(stream))
            expected = [{key: str(value) for key, value in row.items()} for row in rows]
            if actual != expected:
                raise ValueError(f"saved table differs from frozen reports: {name}")
        figures = list((output / "figures").glob("*.svg"))
        if len(figures) != 7:
            raise ValueError(f"expected seven saved SVG figures, found {len(figures)}")
        for figure in figures:
            if ET.parse(figure).getroot().tag != "{http://www.w3.org/2000/svg}svg":
                raise ValueError(f"invalid SVG root: {figure.name}")
        if not (output / "final_results.md").is_file():
            raise ValueError("missing final_results.md")
    except (OSError, ValueError, KeyError, ET.ParseError) as error:
        print(f"Saved-artifact verification failed: {error}", file=sys.stderr)
        return 1
    print(f"Saved reports ({len(frozen)} pinned file hashes), derived tables, figures, and provenance: verified.")
    missing = [name for name in EXTERNAL_INPUTS if statuses[name] == "missing"]
    if missing:
        print("External inputs unavailable; end-to-end reproduction is blocked: " + ", ".join(missing))
        return 2
    print("All recorded external inputs: exact byte hashes verified.")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports", type=Path, default=Path("reports"))
    args = parser.parse_args()
    raise SystemExit(verify(args.reports))


if __name__ == "__main__":
    main()
