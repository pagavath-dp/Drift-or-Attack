# Stage 1 synthetic episode diagnostics

These are generator checks, not classifier results or real sensor-health labels. Episodes are artificial seeded orderings within one batch, gas, and concentration. Gas/concentration are laboratory grouping metadata only.

- Archive SHA-256: `98fe3a30981a222dd4518fbcc3dddd45d5c0ce9b03ef6dc6fe5cf7a04cfbff5e`; master seed: 1729.
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

**Severity comparability gate unresolved:** bias median standardized change is 1, versus 0.0691 for gradual gain (14.5×). The bias operation induces 1844 feature values beyond training ranges. These prespecified interventions are numerically valid, but a future classifier could exploit magnitude instead of temporal cause. No severity was adjusted using model outcomes; obtain approval before training or revising this design.

`episode_diagnostics.json` contains family/severity counts by split and family counts by batch. `episode_manifest.csv` records source and grouping metadata for auditing; those columns must never be passed as model features.

## Unresolved scientific limitations

- Source order is artificial; a ramp versus an abrupt step is an injected signature. Equal active duration removes a trivial duration cue but cannot make intent causally identifiable. A gradual attacker can mimic synthetic drift.
- All interventions edit extracted feature blocks rather than raw sensor waveforms. Numerical validity does not establish physical plausibility or security realism.
- Background is unmodified, not verified healthy. The generator does not infer natural drift or real cyberattack labels.
- Batches 7–10 are exploratory held-out evaluation because their audit coverage and score distributions were inspected before protocol freeze. No model was trained or evaluated here.
- Model-input shortcut tests, operating-threshold selection, and statistical uncertainty checks remain for later approved stages.
