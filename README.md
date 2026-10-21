# Drift-or-Attack: controlled synthetic intervention study

This repository contains a saved audit and model comparison using the UCI Gas Sensor Array Drift at Different Concentrations dataset. It classifies **assigned feature-space simulations**, not verified natural sensor aging or real cyberattacks. Batches 7–10 are exploratory because their audit distributions were inspected before the protocol was frozen. The protocol and saved results are in `docs/experimental_protocol.md` and `reports/`.

The checkout does **not** contain the raw UCI archive, the three root-level PDFs recorded in `reports/source_manifest.json`, or a `.venv`. They must be supplied separately for end-to-end reproduction. Do not replace the recorded hashes to accommodate a different download or document.

## Safe Windows verification

From PowerShell in the repository root, use Python 3.13 or another Python 3.10+ interpreter for the standard-library-only verifier:

```powershell
py -3.13 -B -m scripts.verify_frozen --reports reports
```

The command only reads files; `-B` also prevents Python bytecode cache writes. It checks recorded input hashes, all 34 checked-in report files against `docs/frozen_report_hashes.json` (digests of the original Git commit), saved report arithmetic and provenance, derived CSV tables, and the seven SVG files. Exit code `0` means all recorded inputs and saved artifacts passed; `2` means the saved artifacts passed but one or more external inputs are missing; `1` means an available input or saved artifact failed verification. Its output names each missing file or mismatch separately.

The historical hashes are SHA-256 hashes of the original committed bytes; line endings vary by file. Six recorded text inputs have LF in Git and CRLF in this Windows checkout. Verification first checks the exact byte hash. For text artifacts (`.csv`, `.json`, `.md`, and `.svg`) only, it then accepts CRLF-to-LF conversion **when that conversion reproduces the recorded hash** and prints `crlf_checkout` for affected source inputs. CSV files whose original committed bytes already use CRLF pass the exact-byte check. Archive and PDF hashes must match byte for byte. Verification never changes a historical hash or rewrites a file; changed text content still fails.

The external inputs expected at the recorded paths are:

| Path | Required provenance |
| --- | --- |
| `data/raw/uci-270.zip` | UCI 270 archive; SHA-256 `98fe3a30981a222dd4518fbcc3dddd45d5c0ce9b03ef6dc6fe5cf7a04cfbff5e` |
| `Drift or Attack AIot.pdf` | Proposal PDF; SHA-256 recorded in `reports/source_manifest.json` |
| `drift_or_attack_IEEE_report.pdf` | Earlier report PDF; SHA-256 recorded in the manifest |
| `drift_or_attack_IEEE_turnitin.pdf` | Earlier similarity PDF; SHA-256 recorded in the manifest |

The UCI dataset page is <https://archive.ics.uci.edu/dataset/270/gas+sensor+array+drift+dataset+at+different+concentrations>. The archive digest pins the version used by this study; a later official download may differ and requires investigation rather than a silent pin change.

For the full test suite, create a local environment after Python 3.13 is installed:

```powershell
py -3.13 -m venv .venv
& .\.venv\Scripts\python.exe -m pip install -r requirements-ml.txt
$env:PYTHONPATH = 'src'
& .\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

The tests that generate final figures and tables write only to a temporary directory; the integration preflight test skips when the raw archive is absent. The AE training runner additionally requires the packages pinned in `requirements-ae.lock`. Package installation and an equivalent training environment have not been verified in this checkout.

## Reproduction and write paths

The audit, episode audit, preflight, baseline, AE, and final-results builder are **generation commands**. Their `--output` or `--reports` arguments write files. Do not point them at checked-in `reports/`. The final-results builder rejects any output path inside the supplied reports directory and requires an empty output directory. Use a separate work directory after the original external inputs and environment pass verification. The saved reports remain authoritative; a new run must compare its confusion matrices with them before any per-episode export is treated as reproduced evidence.

The fixed split is Batch 1 reference, Batches 2–4 training, Batches 5–6 validation, and Batches 7–10 exploratory test. The synthetic families are bias, scaling, noise, freeze, and replay. Freeze and replay are exact-copy controls with deterministic signatures. Binary intervention detection and three-class background/drift/attack classification are separate tasks. The matched-primary exploratory three-class macro-F1 is 0.3873 for RF and 0.3694 for AE-gated RF; their background intervention flags are 724/909 and 752/909, respectively. These are frozen results, not deployment performance.

For the 24 October course submission, use `docs/submission_note.md` for methodology, the implemented/proposed architecture boundary, and results wording; `docs/course_submission_checklist.md` for required sections and team-supplied files; and `docs/future-paper-gap.md` only as an after-submission research handoff. The opening status line in the frozen experimental protocol describes an earlier packaging state; this README states what is present in the current checkout.
