# Stage 1 synthetic episode diagnostics — original_stress

These are generator checks, not classifier results or real sensor-health labels. Episodes are artificial seeded orderings within one batch, gas, and concentration. Gas/concentration are laboratory grouping metadata only.

- Archive SHA-256: `98fe3a30981a222dd4518fbcc3dddd45d5c0ce9b03ef6dc6fe5cf7a04cfbff5e`; master seed: 1729.
- Original approved bias grid: [0.5, 1.0, 2.0]; calibration digest: `None`.
- Source and generated-vector duplicates across base episodes, source-row reuse, cross-partition paired variants, and padding/duration mismatches: all zero (generation stops if any occur).
- Every variant has eight positions; intervention positions are 5–7 (zero-based). Background positions are unmodified. Each intervention changes only one shared selected eight-feature sensor block.
- Generator scales use Batches 1–4 only. Min/median/max: 0.587922/7.35187/107254.

| Batch | Split | Source rows | Eight-row episodes | Used rows | Excluded rows |
|---:|---|---:|---:|---:|---:|
| 1 | reference | 445 | 6 | 48 | 397 |
| 2 | train | 1244 | 135 | 1080 | 164 |
| 3 | train | 1586 | 192 | 1536 | 50 |
| 4 | train | 161 | 15 | 120 | 41 |
| 5 | validation | 197 | 20 | 160 | 37 |
| 6 | validation | 2300 | 262 | 2096 | 204 |
| 7 | test | 3613 | 391 | 3128 | 485 |
| 8 | test | 294 | 31 | 248 | 46 |
| 9 | test | 470 | 55 | 440 | 30 |
| 10 | test | 3600 | 432 | 3456 | 144 |

Batch 1 episodes are counted for source feasibility only; Track B variants are generated for Batches 2–10.

| Split | Background | Synthetic drift | Synthetic attack |
|---|---:|---:|---:|
| train | 342 | 342 | 342 |
| validation | 282 | 282 | 282 |
| test | 909 | 909 | 909 |

| Split | Bias | Scaling | Noise | Freeze | Replay |
|---|---:|---:|---:|---:|---:|
| train | 69 | 69 | 68 | 68 | 68 |
| validation | 57 | 57 | 56 | 56 | 56 |
| test | 184 | 182 | 181 | 181 | 181 |

## Intervention diagnostics

| Family | Episodes | Median mean absolute standardized change | Induced values beyond training range / affected values |
|---|---:|---:|---:|
| bias | 310 | 1 | 1844/7440 (24.8%) |
| freeze | 305 | 0.115 | 5/7320 (0.1%) |
| gradual_gain | 1533 | 0.0691 | 28/36792 (0.1%) |
| noise | 305 | 0.414 | 653/7320 (8.9%) |
| replay | 305 | 0.119 | 4/7320 (0.1%) |
| scaling | 308 | 0.102 | 9/7392 (0.1%) |

## Per-split, per-family and severity magnitudes

| Split | Family and severity | Episodes | Median mean standardized change | Range |
|---|---|---:|---:|---:|
| train | bias:0.5 | 23 | 0.5 | 0.5–0.5 |
| train | bias:1.0 | 23 | 1 | 1–1 |
| train | bias:2.0 | 23 | 2 | 2–2 |
| train | freeze:None | 68 | 0.069878 | 0.0020797–1.2603 |
| train | gradual_gain:0.05 | 114 | 0.050217 | 0.0037377–0.33978 |
| train | gradual_gain:0.1 | 114 | 0.09558 | 0.0035811–0.27524 |
| train | gradual_gain:0.2 | 114 | 0.21542 | 0.0085974–0.57062 |
| train | noise:0.25 | 22 | 0.20569 | 0.15174–0.25642 |
| train | noise:0.5 | 23 | 0.42582 | 0.29631–0.58588 |
| train | noise:1.0 | 23 | 0.78526 | 0.62554–1.0157 |
| train | replay:None | 68 | 0.10203 | 0.0011936–16.947 |
| train | scaling:0.05 | 23 | 0.081039 | 0.0077114–0.20365 |
| train | scaling:0.1 | 23 | 0.085175 | 0.011294–0.29153 |
| train | scaling:0.2 | 23 | 0.3119 | 0.012948–0.61539 |
| validation | bias:0.5 | 20 | 0.5 | 0.5–0.5 |
| validation | bias:1.0 | 18 | 1 | 1–1 |
| validation | bias:2.0 | 19 | 2 | 2–2 |
| validation | freeze:None | 56 | 0.13672 | 0.013246–0.81226 |
| validation | gradual_gain:0.05 | 95 | 0.035813 | 0.0031171–0.11379 |
| validation | gradual_gain:0.1 | 94 | 0.08809 | 0.007177–0.2055 |
| validation | gradual_gain:0.2 | 93 | 0.12977 | 0.01102–0.40447 |
| validation | noise:0.25 | 18 | 0.21448 | 0.16454–0.27296 |
| validation | noise:0.5 | 18 | 0.40883 | 0.32199–0.53232 |
| validation | noise:1.0 | 20 | 0.72404 | 0.50847–0.98886 |
| validation | replay:None | 56 | 0.10835 | 0.0078243–0.69446 |
| validation | scaling:0.05 | 19 | 0.073167 | 0.0061379–0.17062 |
| validation | scaling:0.1 | 20 | 0.10289 | 0.01054–0.31423 |
| validation | scaling:0.2 | 18 | 0.13882 | 0.023798–0.46877 |
| test | bias:0.5 | 63 | 0.5 | 0.5–0.5 |
| test | bias:1.0 | 60 | 1 | 1–1 |
| test | bias:2.0 | 61 | 2 | 2–2 |
| test | freeze:None | 181 | 0.13119 | 0.0070633–3.3037 |
| test | gradual_gain:0.05 | 305 | 0.031592 | 0.0016183–0.21868 |
| test | gradual_gain:0.1 | 302 | 0.065385 | 0.0025115–0.46126 |
| test | gradual_gain:0.2 | 302 | 0.12146 | 0.0076017–0.80708 |
| test | noise:0.25 | 59 | 0.20357 | 0.12198–0.27983 |
| test | noise:0.5 | 61 | 0.40473 | 0.27695–0.52812 |
| test | noise:1.0 | 61 | 0.81951 | 0.54043–1.0684 |
| test | replay:None | 181 | 0.13364 | 0.0063869–2.3363 |
| test | scaling:0.05 | 61 | 0.054795 | 0.0082201–0.23517 |
| test | scaling:0.1 | 61 | 0.1131 | 0.0051219–0.34422 |
| test | scaling:0.2 | 60 | 0.20503 | 0.034251–0.96777 |

## Paired bias-versus-drift magnitude comparison

Both variants use the same base episode and gain-level index. The median absolute paired gap shows residual magnitude mismatch; distribution matching does not guarantee equal magnitude within a pair.

| Split | Gain level | Bias episodes | Bias median | Paired drift median | Median absolute paired gap |
|---|---:|---:|---:|---:|---:|
| train | 0.05 | 23 | 0.5 | 0.033216 | 0.46678 |
| train | 0.1 | 23 | 1 | 0.10099 | 0.89901 |
| train | 0.2 | 23 | 2 | 0.20838 | 1.7916 |
| validation | 0.05 | 20 | 0.5 | 0.025764 | 0.47424 |
| validation | 0.1 | 18 | 1 | 0.079867 | 0.92013 |
| validation | 0.2 | 19 | 2 | 0.16713 | 1.8329 |
| test | 0.05 | 63 | 0.5 | 0.031399 | 0.4686 |
| test | 0.1 | 60 | 1 | 0.0547 | 0.9453 |
| test | 0.2 | 61 | 2 | 0.10293 | 1.8971 |

Across all splits and levels, bias median is 1 and gradual-gain median is 0.0691 (14.47×). This pooled ratio mixes severity levels and is descriptive only. Bias induces 1844 values beyond B1-4 feature ranges. No severity used model outcomes.

The condition-specific JSON contains exact family/severity counts by split and family counts by batch. The condition-specific manifest records source and grouping metadata for auditing; those columns must never be passed as model features.

## Unresolved scientific limitations

- Source order is artificial; a ramp versus an abrupt step is an injected signature. Equal active duration removes a trivial duration cue but cannot make intent causally identifiable. A gradual attacker can mimic synthetic drift.
- The primary bias uses exact values from finite frozen training pools. Their reuse may be a learnable simulation signature. Static pairing and duration checks cannot rule out every future model-input shortcut.
- All interventions edit extracted feature blocks rather than raw sensor waveforms. Numerical validity does not establish physical plausibility or security realism.
- Background is unmodified, not verified healthy. The generator does not infer natural drift or real cyberattack labels.
- Batches 7–10 are exploratory held-out evaluation because their audit coverage and score distributions were inspected before protocol freeze. No model was trained or evaluated here.
- Model-input shortcut tests, operating-threshold selection, and statistical uncertainty checks remain for later approved stages.
