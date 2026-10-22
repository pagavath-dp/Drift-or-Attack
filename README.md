# Drift-or-Attack: controlled synthetic intervention study

This repository contains a saved audit and model comparison using the UCI Gas Sensor Array Drift at Different Concentrations dataset. It classifies **assigned feature-space simulations**, not verified natural sensor aging or real cyberattacks. Batches 7–10 are exploratory because their audit distributions were inspected before the protocol was frozen. The protocol and saved results are in `docs/experimental_protocol.md` and `reports/`.

This local checkout contains the pinned UCI archive and all three root-level PDFs recorded in `reports/source_manifest.json`; `scripts.verify_frozen` verifies their exact hashes and all 34 frozen reports. They are external inputs and are not part of the intended code commit. The PDFs are required for strict historical provenance verification, but are not model inputs for a new isolated scientific run. A repository-root `.venv` exists on the current macOS machine; it is ignored by Git and is not distributed with the repository. Do not replace a recorded hash to accommodate a different download or document.

## Safe Windows verification

From PowerShell in this checkout, use the locally established Python 3.13 environment for the standard-library-only verifier (`py -3.13` is also suitable on a machine with the Windows Python launcher):

```powershell
& .\reproductions\toolchain\python-embed-3.13.16\python.exe -B -m scripts.verify_frozen --reports reports
```

The command only reads files; `-B` also prevents Python bytecode cache writes. It checks recorded input hashes, all 34 checked-in report files against `docs/frozen_report_hashes.json` (digests of the original Git commit), saved report arithmetic and provenance, derived CSV tables, and the seven SVG files. Inventory checks remain strict outside the separately managed `reports/evaluation2/` tree and incidental files named `.DS_Store`; an unexpected file elsewhere under `reports/` fails verification. The Evaluation 2 paper is not a pinned historical artifact and is not content-verified by this command. Exit code `0` means all recorded inputs and saved artifacts passed; `2` means the saved artifacts passed but one or more external inputs are missing; `1` means an available input or saved artifact failed verification. Its output names each missing file or mismatch separately.

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

The tests that generate final figures and tables write only to a temporary directory; the integration preflight test uses a temporary output directory when the raw archive is present. The AE training runner additionally requires the packages pinned in `requirements-ae.lock`. A local Windows Python 3.13.16 environment with those exact package versions was established under ignored `reproductions/toolchain/`; `scripts.reproduce_models --check-only` passed there. Equivalent generated model evidence has **not** been established.

On this macOS checkout, the following safe checks were run on 22 October 2026 from the repository root. All exited `0`; the test suite ran 56 tests and used temporary outputs without changing frozen reports. The verifier and reproduction preflight only read project files. The reproduction command checks prerequisites; it does not fit models or validate new predictions.

```sh
PYTHONPATH=src .venv/bin/python -B -m unittest discover -s tests -q
.venv/bin/python -B -m scripts.verify_frozen --reports reports
PYTHONPATH=src .venv/bin/python -B -m scripts.reproduce_models --check-only --archive data/raw/uci-270.zip
```

## Reproduction and write paths

The audit, episode audit, preflight, baseline, AE, and final-results builder are **generation commands**. Their `--output` or `--reports` arguments write files. They reject targets inside this checkout's frozen `reports/` tree. The final-results builder also rejects an output path inside its supplied reports directory and requires an empty output directory. Historical verification with `scripts.verify_frozen` remains strict about all recorded PDFs.

For a **new isolated model reproduction**, `scripts.reproduce_models` checks the pinned archive, Python 3.13, every version in `requirements-ae.lock`, and all frozen report hashes before generating anything. It copies only the frozen calibration/preflight input evidence to a new output directory; the old PDFs are not read. It fits the existing baseline and AE runners unchanged in their choices, compares generated validation and exploratory matrices and selected thresholds with the frozen JSON, then reconstructs those matrices from per-episode rows. It writes `episode_predictions.csv` and `matrix_comparison.json` only after every comparison passes. An incomplete or mismatched run leaves only isolated diagnostic outputs, never a validated prediction export. The CSV carries episode ID, split, condition, batch, gas/concentration as audit metadata, true label, RF prediction and probabilities, AE score, AE binary decision, and AE-gated prediction. Gas and concentration remain excluded from model inputs.

```powershell
$python = '.\reproductions\toolchain\python-embed-3.13.16\python.exe'
& $python -B -m scripts.reproduce_models --check-only --archive data/raw/uci-270.zip
# Only after the check passes, choose a new output directory for an isolated run:
& $python -B -m scripts.reproduce_models --archive data/raw/uci-270.zip --output reproductions/run-new
```

On a new Windows checkout, install Python 3.13 and the exact `requirements-ae.lock` packages in a separate environment before using these commands; the ignored `reproductions/toolchain/` directory is local and is not distributed with the repository. The already attempted Windows run did not pass the exact comparison gate, so a new run may stop before export as well.

The output path must be new and outside frozen `reports/`; use a new name for another attempt. The archive must match SHA-256 `98fe3a30981a222dd4518fbcc3dddd45d5c0ce9b03ef6dc6fe5cf7a04cfbff5e`. The saved reports remain authoritative if a reproduction differs.

**Windows checkpoint, 21 October 2026:** One isolated run at `reproductions/run-2026-10-21-windows/` stopped before AE fitting and prediction export because two generated preflight validation noise medians differ from the frozen JSON by approximately `1.1e-16`. The already generated baseline diagnostics also selected RF `min_samples_leaf=5` instead of the frozen `1`, with different RF confusion matrices; the Isolation Forest threshold and matrices matched. Do not treat this directory as a reproduced result or use it for gas-specific error rates. The ignored environment and diagnostics may be absent in another checkout.

The fixed split is Batch 1 reference, Batches 2–4 training, Batches 5–6 validation, and Batches 7–10 exploratory test. The synthetic families are bias, scaling, noise, freeze, and replay. Freeze and replay are exact-copy controls with deterministic signatures. Binary intervention detection and three-class background/drift/attack classification are separate tasks. The matched-primary exploratory three-class macro-F1 is 0.3873 for RF and 0.3694 for AE-gated RF; their background intervention flags are 724/909 and 752/909, respectively. These are frozen results, not deployment performance.

For the 24 October course submission, use `docs/submission_note.md` for methodology, the implemented/proposed architecture boundary, and results wording; `docs/course_submission_checklist.md` for required sections and team-supplied files; and `docs/future-paper-gap.md` for the active research-validation tasks that may continue after upload. The opening status line in the frozen experimental protocol describes an earlier packaging state; this README states what is present in the current checkout.
