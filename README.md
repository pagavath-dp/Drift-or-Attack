# Drift-or-Attack data audit

This repository preserves the submitted project PDFs and audits the UCI **Gas Sensor Array Drift at Different Concentrations** dataset (dataset 270, DOI `10.24432/C5MK6M`). The audit does not train a model or assert that UCI supplies drift or cyberattack labels.

Source: https://archive.ics.uci.edu/dataset/270/gas+sensor+array+drift+dataset+at+different+concentrations

Use a project-local Python 3.13 environment. No third-party Python package is needed for this milestone. The raw archive remains in ignored `data/raw/`.

```sh
/Library/Frameworks/Python.framework/Versions/3.13/bin/python3 -m venv .venv
mkdir -p data/raw
curl --fail --location --output data/raw/uci-270.zip 'https://archive.ics.uci.edu/static/public/270/gas%2Bsensor%2Barray%2Bdrift%2Bdataset%2Bat%2Bdifferent%2Bconcentrations.zip'
shasum -a 256 data/raw/uci-270.zip
unzip -l data/raw/uci-270.zip
PYTHONPATH=src .venv/bin/python -m drift_attack.audit --archive data/raw/uci-270.zip --output reports
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -v
```

The audit pins the SHA-256 of the archive inspected for this milestone and checks it before parsing. The parser reads ZIP members directly and rejects unexpected names, unsafe paths, malformed rows, missing features, and non-finite values. It does not extract archive paths. UCI does not publish an independent archive digest on its dataset page; a changed official download must be reviewed before changing the pin.

The fixed split follows the submitted proposal: Batch 1 reference, Batches 2-4 training, Batches 5-6 validation, and Batches 7-10 held-out test. The audit only assesses whether a strict reference-envelope proxy might be feasible. It never treats later batches as verified drift. See `reports/data_audit.md` for the precise rule, coverage, counts, and limitations.

## Stage 1 synthetic episode checks

The approved Design C protocol is in `docs/experimental_protocol.md`. Its episodes are artificial orderings, and its intervention labels describe assigned simulations only. The original Stage 1 generator diagnostics and manifest remain under `reports/`; run the named conditions below for current diagnostics.

```sh
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -v
```

The diagnostics and source-episode manifest are written under `reports/`. The manifest's gas, concentration, batch, source IDs, family, and sensor fields are audit metadata and must not become model inputs. The generator stops on duplicate source vectors, source-row reuse, cross-episode generated-vector duplicates, partition mismatches, unexpected intervention positions, or non-finite values.

## Training-only magnitude calibration

The original Stage 1 bias grid `0.5, 1, 2` and its `reports/episode_diagnostics.*` and `reports/episode_manifest.csv` are preserved. Run the two explicitly named conditions to create separate outputs without replacing those files:

```sh
PYTHONPATH=src .venv/bin/python -m drift_attack.episode_audit --archive data/raw/uci-270.zip --output reports --condition original_stress
PYTHONPATH=src .venv/bin/python -m drift_attack.episode_audit --archive data/raw/uci-270.zip --output reports --condition matched_primary
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -v
```

`reports/magnitude_calibration.json` records the Batches 1–4 scale provenance, all Batch 2–4 source-episode IDs and training drift magnitudes, the frozen pools, and their digest. The `matched_primary` bias draws only from these frozen pools. Each condition has its own diagnostics and manifest filename. Magnitude matching is distributional over the training episodes; the reports show residual paired gaps and per-split changes. Validation and test do not recalibrate the generator. No model is trained by these commands.

## Anomaly-only and Random Forest stage

The approved baseline environment is the existing project-local Python 3.13 `.venv`, with exact installed versions in `requirements-ml.txt`. TensorFlow and XGBoost are not installed for this stage. Run the pretraining gates before fitting; the model runner repeats them automatically:

```sh
PYTHONPATH=src .venv/bin/python -m drift_attack.model_preflight --archive data/raw/uci-270.zip --reports reports
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -v
PYTHONPATH=src .venv/bin/python -m drift_attack.ml_baselines --archive data/raw/uci-270.zip --reports reports
```

The exact 768 numeric inputs, training/validation split, threshold and model-selection rules, and metric denominators are in `docs/experimental_protocol.md`. `reports/ml_preflight.json` records finite-pool reuse and paired magnitude mismatch by split, family, and severity. `reports/ml_baselines.md` summarizes results; the JSON contains confusion matrices, per-class metrics, batch breakdowns, and denominators. The original-stress condition is evaluated separately with the primary-trained models. Batches 7–10 are exploratory; artificial episode results do not demonstrate real sensor aging or cyberattack detection.

## Exact-copy diagnostic and autoencoder comparison

The original preflight is retained as `reports/ml_preflight_before_exact_copy.json`. The current preflight additionally counts exact-zero freeze and exact-repeat replay sensor-block signatures by split and family. Freeze/replay are exact-copy controls; their results do not establish general attack detection. The primary 768-feature representation and baseline result files are unchanged.

TensorFlow 2.21.0 was installed only in the project-local Python 3.13 `.venv`; `requirements-ae.lock` records the complete installed package versions. XGBoost was not installed. Reproduce the approved autoencoder comparison with:

```sh
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -v
PYTHONPATH=src .venv/bin/python -m drift_attack.autoencoder_stage --archive data/raw/uci-270.zip --reports reports
```

`reports/ml_autoencoder.md` summarizes the comparison and `reports/ml_autoencoder.json` contains full binary and three-class metrics, per-batch and family/severity results, denominators, provenance, and limitations. The runner repeats preflight gates, confirms the saved forest validation confusion, and uses Batches 5–6 only for autoencoder threshold selection. Batches 7–10 remain exploratory.
