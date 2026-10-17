# UCI 270 data audit

## Source and observed structure

- Official source page: https://archive.ics.uci.edu/dataset/270/gas+sensor+array+drift+dataset+at+different+concentrations
- Download URL: https://archive.ics.uci.edu/static/public/270/gas%2Bsensor%2Barray%2Bdrift%2Bdataset%2Bat%2Bdifferent%2Bconcentrations.zip
- Downloaded archive SHA-256: `98fe3a30981a222dd4518fbcc3dddd45d5c0ce9b03ef6dc6fe5cf7a04cfbff5e`; bytes: 10,118,308. The audit checks this recorded digest before parsing. UCI does not publish an independent digest on the source page.
- Archive validation: exactly `batch1.dat` through `batch10.dat`, no extra or unsafe paths, ZIP CRC check passed. Members were read directly; no extraction occurred.
- Observed rows: 13,910. Every row has a gas code, concentration, and exactly 128 consecutive indexed, finite numeric features.
- Observed gas codes: 1 Ethanol, 2 Ethylene, 3 Ammonia, 4 Acetaldehyde, 5 Acetone, 6 Toluene (names from UCI documentation).
- The archive verifies 128 feature values per row. The interpretation as eight features from each of 16 physical sensors comes from UCI documentation, not a separate sensor ID field in the files.
- File line number is available, but no per-record timestamp, attack onset, or verified drift/attack label was found. Do not infer a time interval or true within-batch chronology from line order.
- UCI documents the following batch-level month groups: B1: 1, 2; B2: 3, 4, 8, 9, 10; B3: 11, 12, 13; B4: 14, 15; B5: 16; B6: 17, 18, 19, 20; B7: 21; B8: 22, 23; B9: 24, 30; B10: 36. These months are metadata, not row-level fields.

## Fixed split and reference-envelope feasibility rule

- B1 is the reference; B2-4 training; B5-6 validation; B7-10 held-out test. No random split or model fitting was performed.
- Match only exact `(gas code, concentration)` pairs represented by at least 5 B1 records. A different concentration is unmatched; groups with fewer than 5 B1 records are reported as sparse. No nearest-neighbor imputation is used.
- For each feature, compute the sample standard deviation on all B1 records only. Floor a zero/near-zero scale at `1e-12`. For each eligible pair, compute the per-feature median on B1 records in that pair.
- Distance is the mean of 128 absolute feature deviations from the pair median, each divided by its B1 feature scale. Gas and concentration are matching keys, not distance inputs.
- For each eligible B1 record, recompute its pair median without that record and calculate a leave-one-out distance. Pool these B1-only distances and take linearly interpolated 90th, 95th, and 99th percentiles. These are prespecified sensitivity alternatives; no validation or test data choose a cutoff.
- Counts below show `within` and `outside` the B1 reference envelope. They are candidate operational states, **not verified normal or sensor drift labels**. B1 counts use full reference medians and are in-sample, so they must not be read as calibrated false-alarm estimates.
- Assumptions: B1 is a usable clean reference; exact gas/concentration identity would be known at decision time; within-pair B1 spread is representative; and a feature-space departure may reflect drift. These assumptions are unverified. Gas and concentration can be known in the laboratory but might not be known in deployment.

## Counts by batch

| Batch | Split | Rows | Concentrations: distinct (min-max ppmv) | Eligible | No exact B1 pair | Sparse B1 pair | Within/Outside q90 | Within/Outside q95 | Within/Outside q99 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | reference | 445 | 33 (5-1000) | 395 | 0 | 50 | 363/32 | 375/20 | 391/4 |
| 2 | train | 1244 | 23 (10-600) | 955 | 44 | 245 | 150/805 | 444/511 | 823/132 |
| 3 | train | 1586 | 13 (10-300) | 977 | 262 | 347 | 66/911 | 166/811 | 334/643 |
| 4 | train | 161 | 5 (50-250) | 131 | 0 | 30 | 0/131 | 12/119 | 14/117 |
| 5 | validation | 197 | 7 (50-500) | 131 | 0 | 66 | 0/131 | 0/131 | 27/104 |
| 6 | validation | 2300 | 17 (10-250) | 2137 | 86 | 77 | 0/2137 | 493/1644 | 1313/824 |
| 7 | test | 3613 | 43 (1-300) | 1118 | 2082 | 413 | 46/1072 | 302/816 | 823/295 |
| 8 | test | 294 | 5 (10-500) | 137 | 42 | 115 | 0/137 | 76/61 | 113/24 |
| 9 | test | 470 | 3 (10-120) | 219 | 98 | 153 | 0/219 | 57/162 | 217/2 |
| 10 | test | 3600 | 16 (1-1000) | 2360 | 640 | 600 | 192/2168 | 1104/1256 | 2158/202 |

These are observed record counts, not class-label counts. `batch_summary.csv` and `batch_gas_summary.csv` give machine-readable summaries. Neither file assigns row-level proxy labels.

### Gas records by batch (observed)

| Batch | Ethanol | Ethylene | Ammonia | Acetaldehyde | Acetone | Toluene |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 90 | 98 | 83 | 30 | 70 | 74 |
| 2 | 164 | 334 | 100 | 109 | 532 | 5 |
| 3 | 365 | 490 | 216 | 240 | 275 | 0 |
| 4 | 64 | 43 | 12 | 30 | 12 | 0 |
| 5 | 28 | 40 | 20 | 46 | 63 | 0 |
| 6 | 514 | 574 | 110 | 29 | 606 | 467 |
| 7 | 649 | 662 | 360 | 744 | 630 | 568 |
| 8 | 30 | 30 | 40 | 33 | 143 | 18 |
| 9 | 61 | 55 | 100 | 75 | 78 | 101 |
| 10 | 600 | 600 | 600 | 600 | 600 | 600 |

## Cutoffs and concentration coverage

- Eligible B1 groups: 67 of 87; B1 leave-one-out scores: 395.
- B1-only cutoffs: q90 = 0.253605, q95 = 0.4814, q99 = 0.959129.

| Gas | UCI page's stated interval (ppmv) | Observed min-max (ppmv) | Distinct observed levels |
|---|---:|---:|---:|
| Ethanol | 10-600 | 2.5-600 | 35 |
| Ethylene | 10-300 | 2.5-300 | 38 |
| Ammonia | 50-1000 | 2.5-1000 | 44 |
| Acetaldehyde | 5-500 | 2.5-300 | 37 |
| Acetone | 12-1000 | 10-1000 | 47 |
| Toluene | 10-100 | 1-230 | 32 |

Several observed extremes fall outside the intervals stated on the UCI page. This is a documentation/data discrepancy, not a reason to discard those rows; the parser uses the observed values and reports them without correction.

## Feasibility decision and limits

- Strict B1 reference coverage in held-out B7-10 is 3,834/7,977 (48.1%); 4,143 records cannot receive either reference-envelope state under this rule.
- Acetaldehyde (gas 4) has 0 eligible records across all batches because its B1 exact concentration groups have too few replicates.
- Therefore this rule does **not** support a full six-gas, all-record three-class evaluation. Even for eligible records, outside-envelope status does not establish sensor aging; changes in measurement conditions and other causes remain possible. No normal/drift/attack labels or attack records were produced.
- Honest alternatives for a later approved phase: evaluate reference departure only on explicitly covered gas/concentration strata, while reporting exclusions; obtain independent drift annotations or controlled recalibration data; or build a clearly synthetic three-way simulation and state that its drift class is simulated rather than natural.
- Detection delay in elapsed time is unsupported: the archive has no per-record timestamps or verified attack start. ZIP file order and line numbers are not a measured stream.

## Submitted PDF integrity

- `Drift or Attack AIot.pdf`: `0f8b3ea3c49ae8a92b210baefc65f4d0970d4f267ea7d0cd693629c32e3d40aa`
- `drift_or_attack_IEEE_report.pdf`: `eab289504443e08d22ec36cbc1f7a5f88bdbeac911fd6bdf9fa000b3b822cab0`
- `drift_or_attack_IEEE_turnitin.pdf`: `1e2cbf94c22e36911e7f29e1bfd59177044df440908efb6507cc927e94491232`
