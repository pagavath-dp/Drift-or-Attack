# Autoencoder stage — synthetic intervention study

Background denotes unmodified input, not verified healthy hardware. Freeze and replay are exact-copy controls. Batches 7–10 are exploratory because their audit coverage and scores were inspected before protocol freeze.

## Fitting and selection

- Unmodified background episodes for final fit: 348 (B1–4: {'1': 6, '2': 135, '3': 192, '4': 15}).
- Source grouping by batch (gas/concentration groups and distinct source rows): {'1': {'gas_concentration_groups': 6, 'distinct_source_rows': 48}, '2': {'gas_concentration_groups': 46, 'distinct_source_rows': 1080}, '3': {'gas_concentration_groups': 50, 'distinct_source_rows': 1536}, '4': {'gas_concentration_groups': 7, 'distinct_source_rows': 120}}.
- Early-stop fit/holdout: 280/68 source-grouped episodes; selected epoch 77 from 79 run.
- Batch 5–6 matched-primary binary macro-F1 selected threshold: 0.178459808; higher threshold wins ties. The same threshold is used for stress and exploratory test.
- Selected forest refit reproduced the saved matched-primary validation confusion exactly before gate evaluation.
- Architecture 768→64→16→64→768; Huber reconstruction, Adam 0.001, hidden-kernel L2 1e-4; seed 1729. Median/IQR and asinh normalization use permitted background fitting episodes only.

## Results

Confusion matrices below use true rows and predicted columns. Binary order: background, intervention. Three-class order: background, synthetic drift, synthetic attack.

| Condition | Split | AE binary macro-F1 | Combined macro-F1 | Forest macro-F1 | Combined background false alarms |
|---|---|---:|---:|---:|---:|
| matched_primary | validation | 0.5458 | 0.3950 | 0.4232 | 147/282 (52.1%) |
| matched_primary | test | 0.4949 | 0.3694 | 0.3873 | 752/909 (82.7%) |
| original_stress | validation | 0.5661 | 0.4057 | 0.4294 | 147/282 (52.1%) |
| original_stress | test | 0.4992 | 0.3759 | 0.3928 | 752/909 (82.7%) |

The combined pipeline did not improve three-class macro-F1 over the saved forest in either condition or split. On matched-primary exploratory test it increased background false alarms from 724/909 to 752/909.

### matched_primary

**validation:** AE binary matrix `[[135, 147], [212, 352]]`; combined matrix `[[135, 130, 17], [115, 150, 17], [97, 125, 60]]`. Denominator: 846 paired variants.

| Class | Combined precision | Combined recall | Support |
|---|---:|---:|---:|
| background | 0.389 | 0.479 | 282 |
| synthetic_drift | 0.370 | 0.532 | 282 |
| synthetic_attack | 0.638 | 0.213 | 282 |

| Directional rate | Numerator / denominator |
|---|---:|
| attack_false_alarm | 17/282 |
| any_intervention_false_alarm | 147/282 |
| drift_mistaken_for_attack | 17/282 |
| attack_mistaken_for_background_or_drift | 222/282 |

| Family/severity | Correct / episodes |
|---|---:|
| bias:gain_level=0.05 | 1/20 |
| bias:gain_level=0.1 | 0/18 |
| bias:gain_level=0.2 | 2/19 |
| freeze:severity=None | 19/56 |
| gradual_gain:severity=0.05 | 44/95 |
| gradual_gain:severity=0.1 | 53/94 |
| gradual_gain:severity=0.2 | 53/93 |
| noise:severity=0.25 | 2/18 |
| noise:severity=0.5 | 0/18 |
| noise:severity=1.0 | 5/20 |
| replay:severity=None | 28/56 |
| scaling:severity=0.05 | 1/19 |
| scaling:severity=0.1 | 2/20 |
| scaling:severity=0.2 | 0/18 |

**test:** AE binary matrix `[[157, 752], [264, 1554]]`; combined matrix `[[157, 646, 106], [143, 653, 113], [121, 513, 275]]`. Denominator: 2727 paired variants.

| Class | Combined precision | Combined recall | Support |
|---|---:|---:|---:|
| background | 0.373 | 0.173 | 909 |
| synthetic_drift | 0.360 | 0.718 | 909 |
| synthetic_attack | 0.557 | 0.303 | 909 |

| Directional rate | Numerator / denominator |
|---|---:|
| attack_false_alarm | 106/909 |
| any_intervention_false_alarm | 752/909 |
| drift_mistaken_for_attack | 113/909 |
| attack_mistaken_for_background_or_drift | 634/909 |

| Family/severity | Correct / episodes |
|---|---:|
| bias:gain_level=0.05 | 9/63 |
| bias:gain_level=0.1 | 15/60 |
| bias:gain_level=0.2 | 6/61 |
| freeze:severity=None | 83/181 |
| gradual_gain:severity=0.05 | 220/305 |
| gradual_gain:severity=0.1 | 214/302 |
| gradual_gain:severity=0.2 | 219/302 |
| noise:severity=0.25 | 8/59 |
| noise:severity=0.5 | 13/61 |
| noise:severity=1.0 | 17/61 |
| replay:severity=None | 99/181 |
| scaling:severity=0.05 | 10/61 |
| scaling:severity=0.1 | 7/61 |
| scaling:severity=0.2 | 8/60 |

### original_stress

**validation:** AE binary matrix `[[135, 147], [191, 373]]`; combined matrix `[[135, 130, 17], [115, 150, 17], [76, 140, 66]]`. Denominator: 846 paired variants.

| Class | Combined precision | Combined recall | Support |
|---|---:|---:|---:|
| background | 0.414 | 0.479 | 282 |
| synthetic_drift | 0.357 | 0.532 | 282 |
| synthetic_attack | 0.660 | 0.234 | 282 |

| Directional rate | Numerator / denominator |
|---|---:|
| attack_false_alarm | 17/282 |
| any_intervention_false_alarm | 147/282 |
| drift_mistaken_for_attack | 17/282 |
| attack_mistaken_for_background_or_drift | 216/282 |

| Family/severity | Correct / episodes |
|---|---:|
| bias:severity=0.5 | 3/20 |
| bias:severity=1.0 | 2/18 |
| bias:severity=2.0 | 4/19 |
| freeze:severity=None | 19/56 |
| gradual_gain:severity=0.05 | 44/95 |
| gradual_gain:severity=0.1 | 53/94 |
| gradual_gain:severity=0.2 | 53/93 |
| noise:severity=0.25 | 2/18 |
| noise:severity=0.5 | 0/18 |
| noise:severity=1.0 | 5/20 |
| replay:severity=None | 28/56 |
| scaling:severity=0.05 | 1/19 |
| scaling:severity=0.1 | 2/20 |
| scaling:severity=0.2 | 0/18 |

**test:** AE binary matrix `[[157, 752], [246, 1572]]`; combined matrix `[[157, 646, 106], [143, 653, 113], [103, 516, 290]]`. Denominator: 2727 paired variants.

| Class | Combined precision | Combined recall | Support |
|---|---:|---:|---:|
| background | 0.390 | 0.173 | 909 |
| synthetic_drift | 0.360 | 0.718 | 909 |
| synthetic_attack | 0.570 | 0.319 | 909 |

| Directional rate | Numerator / denominator |
|---|---:|
| attack_false_alarm | 106/909 |
| any_intervention_false_alarm | 752/909 |
| drift_mistaken_for_attack | 113/909 |
| attack_mistaken_for_background_or_drift | 619/909 |

| Family/severity | Correct / episodes |
|---|---:|
| bias:severity=0.5 | 13/63 |
| bias:severity=1.0 | 19/60 |
| bias:severity=2.0 | 13/61 |
| freeze:severity=None | 83/181 |
| gradual_gain:severity=0.05 | 220/305 |
| gradual_gain:severity=0.1 | 214/302 |
| gradual_gain:severity=0.2 | 219/302 |
| noise:severity=0.25 | 8/59 |
| noise:severity=0.5 | 13/61 |
| noise:severity=1.0 | 17/61 |
| replay:severity=None | 99/181 |
| scaling:severity=0.05 | 10/61 |
| scaling:severity=0.1 | 7/61 |
| scaling:severity=0.2 | 8/60 |

## Interpretation and limits

The JSON report also contains binary class precision/recall, four directional rates, and per-batch confusion matrices with denominators for both conditions. Related variants share source episodes, so episode variants and the two conditions are not independent samples. No uncertainty intervals are claimed here.

These are responses to assigned manipulations of extracted features in artificial sequences. They do not validate actual sensor-aging detection or real-world cyberattack detection. Gas/concentration and all generator metadata were excluded from model inputs. The finite matched-bias pool and exact-copy controls remain simulation signatures.
