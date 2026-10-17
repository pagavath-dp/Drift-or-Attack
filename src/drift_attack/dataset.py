"""Strict parser for the inspected UCI dataset 270 ZIP layout."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from zipfile import BadZipFile, ZipFile
import math


GASES = {
    1: "Ethanol",
    2: "Ethylene",
    3: "Ammonia",
    4: "Acetaldehyde",
    5: "Acetone",
    6: "Toluene",
}
EXPECTED_NAMES = {f"batch{i}.dat" for i in range(1, 11)}
FEATURE_COUNT = 128
MAX_MEMBER_BYTES = 20_000_000


@dataclass(frozen=True)
class Measurement:
    batch: int
    line: int
    gas: int
    concentration: float
    features: tuple[float, ...]


def parse_line(text: str, batch: int, line: int) -> Measurement:
    fields = text.strip().split()
    if len(fields) != FEATURE_COUNT + 1:
        raise ValueError(f"batch{batch} line {line}: expected 128 features, got {len(fields)-1}")
    label = fields[0].split(";")
    if len(label) != 2:
        raise ValueError(f"batch{batch} line {line}: invalid gas/concentration label")
    try:
        gas = int(label[0])
        concentration = float(label[1])
    except ValueError as exc:
        raise ValueError(f"batch{batch} line {line}: invalid gas/concentration") from exc
    if gas not in GASES or not math.isfinite(concentration) or concentration <= 0:
        raise ValueError(f"batch{batch} line {line}: invalid gas/concentration value")
    values = []
    for expected, field in enumerate(fields[1:], 1):
        key, sep, value = field.partition(":")
        if not sep or key != str(expected):
            raise ValueError(f"batch{batch} line {line}: expected feature index {expected}")
        try:
            number = float(value)
        except ValueError as exc:
            raise ValueError(f"batch{batch} line {line}: invalid feature value") from exc
        if not math.isfinite(number):
            raise ValueError(f"batch{batch} line {line}: non-finite feature")
        values.append(number)
    return Measurement(batch, line, gas, concentration, tuple(values))


def load_archive(path: str | Path) -> dict[int, list[Measurement]]:
    """Validate the whole archive before reading data; never extract members."""
    with ZipFile(path) as archive:
        members = archive.infolist()
        names = [info.filename for info in members]
        if len(names) != 10 or set(names) != EXPECTED_NAMES:
            raise ValueError(f"unexpected ZIP entries: {names}")
        for info in members:
            name = info.filename
            parts = PurePosixPath(name).parts
            if (info.is_dir() or name.startswith("/") or "\\" in name
                    or ".." in parts or info.file_size > MAX_MEMBER_BYTES):
                raise ValueError(f"unsafe ZIP member: {name}")
        damaged = archive.testzip()
        if damaged is not None:
            raise BadZipFile(f"CRC failure: {damaged}")
        batches: dict[int, list[Measurement]] = {}
        for batch in range(1, 11):
            name = f"batch{batch}.dat"
            rows = []
            with archive.open(name) as member:
                for line, raw in enumerate(member, 1):
                    try:
                        decoded = raw.decode("ascii")
                    except UnicodeDecodeError as exc:
                        raise ValueError(f"{name} line {line}: not ASCII") from exc
                    rows.append(parse_line(decoded, batch, line))
            if not rows:
                raise ValueError(f"empty ZIP member: {name}")
            batches[batch] = rows
    return batches
