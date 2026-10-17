# Anomaly-only and Random Forest results

These are results for artificial interventions on extracted UCI features. They do not establish actual sensor aging or real cyberattack detection. Batches 7–10 are exploratory because their audit distributions were examined before protocol freeze.

- Archive: `98fe3a30981a222dd4518fbcc3dddd45d5c0ce9b03ef6dc6fe5cf7a04cfbff5e`; calibration: `3dd25ca3ad1c66b9a826d70c7714f9e7e08190c84538caef985556b567dab156`.
- Inputs: 768 numeric values from pre_mean, active_minus_pre_mean, onset_step, active_slope, active_variation, lag_four_similarity. Laboratory and generator metadata are excluded.
- Preprocessing fits Batches 1–4; models fit Batches 2–4; threshold and Random Forest setting selection use Batches 5–6 only.
- Anomaly threshold: 0.341941; selected Random Forest settings: `{'n_estimators': 200, 'max_depth': 8, 'min_samples_leaf': 1, 'max_features': 'sqrt', 'random_state': 1729, 'n_jobs': 4}`.

| Condition | Split | Model | Macro-F1 | True-label denominators |
|---|---|---|---:|---|
| matched_primary | validation | anomaly_only | 0.5284 | background 282; intervention 564 |
| matched_primary | validation | random_forest | 0.4232 | background 282; synthetic_drift 282; synthetic_attack 282 |
| matched_primary | test | anomaly_only | 0.4588 | background 909; intervention 1818 |
| matched_primary | test | random_forest | 0.3873 | background 909; synthetic_drift 909; synthetic_attack 909 |
| original_stress | validation | anomaly_only | 0.5469 | background 282; intervention 564 |
| original_stress | validation | random_forest | 0.4294 | background 282; synthetic_drift 282; synthetic_attack 282 |
| original_stress | test | anomaly_only | 0.4605 | background 909; intervention 1818 |
| original_stress | test | random_forest | 0.3928 | background 909; synthetic_drift 909; synthetic_attack 909 |

The anomaly baseline is binary (`background` versus any synthetic intervention). Its binary macro-F1 is not directly comparable with the Random Forest's three-class macro-F1. Its directional drift/attack error rates are undefined.

## Precision, recall, and confusion matrices

Confusion rows are true labels; columns follow the label order printed in each row below.

| Condition | Split | Model | Class | Precision | Recall | Support |
|---|---|---|---|---:|---:|---:|
| matched_primary | validation | anomaly_only | background | 0.370 | 0.397 | 282 |
| matched_primary | validation | anomaly_only | intervention | 0.687 | 0.661 | 564 |
| matched_primary | validation | random_forest | background | 0.391 | 0.468 | 282 |
| matched_primary | validation | random_forest | synthetic_drift | 0.376 | 0.518 | 282 |
| matched_primary | validation | random_forest | synthetic_attack | 0.683 | 0.291 | 282 |
| matched_primary | test | anomaly_only | background | 0.373 | 0.086 | 909 |
| matched_primary | test | anomaly_only | intervention | 0.670 | 0.928 | 1818 |
| matched_primary | test | random_forest | background | 0.381 | 0.204 | 909 |
| matched_primary | test | random_forest | synthetic_drift | 0.362 | 0.680 | 909 |
| matched_primary | test | random_forest | synthetic_attack | 0.573 | 0.337 | 909 |
| original_stress | validation | anomaly_only | background | 0.396 | 0.397 | 282 |
| original_stress | validation | anomaly_only | intervention | 0.698 | 0.697 | 564 |
| original_stress | validation | random_forest | background | 0.398 | 0.468 | 282 |
| original_stress | validation | random_forest | synthetic_drift | 0.374 | 0.518 | 282 |
| original_stress | validation | random_forest | synthetic_attack | 0.694 | 0.305 | 282 |
| original_stress | test | anomaly_only | background | 0.388 | 0.086 | 909 |
| original_stress | test | anomaly_only | intervention | 0.671 | 0.932 | 1818 |
| original_stress | test | random_forest | background | 0.391 | 0.204 | 909 |
| original_stress | test | random_forest | synthetic_drift | 0.362 | 0.680 | 909 |
| original_stress | test | random_forest | synthetic_attack | 0.583 | 0.351 | 909 |

| Condition | Split | Model | Label order | Confusion rows |
|---|---|---|---|---|
| matched_primary | validation | anomaly_only | background, intervention | `[[112, 170], [191, 373]]` |
| matched_primary | validation | random_forest | background, synthetic_drift, synthetic_attack | `[[132, 131, 19], [117, 146, 19], [89, 111, 82]]` |
| matched_primary | test | anomaly_only | background, intervention | `[[78, 831], [131, 1687]]` |
| matched_primary | test | random_forest | background, synthetic_drift, synthetic_attack | `[[185, 613, 111], [174, 618, 117], [127, 476, 306]]` |
| original_stress | validation | anomaly_only | background, intervention | `[[112, 170], [171, 393]]` |
| original_stress | validation | random_forest | background, synthetic_drift, synthetic_attack | `[[132, 131, 19], [117, 146, 19], [83, 113, 86]]` |
| original_stress | test | anomaly_only | background, intervention | `[[78, 831], [123, 1695]]` |
| original_stress | test | random_forest | background, synthetic_drift, synthetic_attack | `[[185, 613, 111], [174, 618, 117], [114, 476, 319]]` |

## Three-class directional rates

| Condition | Split | Attack false alarm | Any-intervention false alarm | Drift → attack | Attack → background/drift |
|---|---|---:|---:|---:|---:|
| matched_primary | validation | 19/282 (6.7%) | 150/282 (53.2%) | 19/282 (6.7%) | 200/282 (70.9%) |
| matched_primary | test | 111/909 (12.2%) | 724/909 (79.6%) | 117/909 (12.9%) | 603/909 (66.3%) |
| original_stress | validation | 19/282 (6.7%) | 150/282 (53.2%) | 19/282 (6.7%) | 196/282 (69.5%) |
| original_stress | test | 111/909 (12.2%) | 724/909 (79.6%) | 117/909 (12.9%) | 590/909 (64.9%) |

The binary anomaly-only background false-alarm denominators are:

| Condition | Split | Background flagged as intervention | Drift detected | Attack detected |
|---|---|---:|---:|---:|
| matched_primary | validation | 170/282 (60.3%) | 180/282 (63.8%) | 193/282 (68.4%) |
| matched_primary | test | 831/909 (91.4%) | 834/909 (91.7%) | 853/909 (93.8%) |
| original_stress | validation | 170/282 (60.3%) | 180/282 (63.8%) | 213/282 (75.5%) |
| original_stress | test | 831/909 (91.4%) | 834/909 (91.7%) | 861/909 (94.7%) |

## Per-batch performance

| Condition | Split | Batch | Model | Macro-F1 | Background false alarms / background |
|---|---|---:|---|---:|---:|
| matched_primary | validation | 5 | anomaly_only | 0.511 | 12/20 (60.0%) |
| matched_primary | validation | 6 | anomaly_only | 0.530 | 158/262 (60.3%) |
| matched_primary | validation | 5 | random_forest | 0.363 | 14/20 (70.0%) |
| matched_primary | validation | 6 | random_forest | 0.427 | 136/262 (51.9%) |
| matched_primary | test | 7 | anomaly_only | 0.469 | 350/391 (89.5%) |
| matched_primary | test | 8 | anomaly_only | 0.512 | 23/31 (74.2%) |
| matched_primary | test | 9 | anomaly_only | 0.528 | 26/55 (47.3%) |
| matched_primary | test | 10 | anomaly_only | 0.400 | 432/432 (100.0%) |
| matched_primary | test | 7 | random_forest | 0.403 | 279/391 (71.4%) |
| matched_primary | test | 8 | random_forest | 0.391 | 22/31 (71.0%) |
| matched_primary | test | 9 | random_forest | 0.421 | 35/55 (63.6%) |
| matched_primary | test | 10 | random_forest | 0.356 | 388/432 (89.8%) |
| original_stress | validation | 5 | anomaly_only | 0.537 | 12/20 (60.0%) |
| original_stress | validation | 6 | anomaly_only | 0.548 | 158/262 (60.3%) |
| original_stress | validation | 5 | random_forest | 0.387 | 14/20 (70.0%) |
| original_stress | validation | 6 | random_forest | 0.432 | 136/262 (51.9%) |
| original_stress | test | 7 | anomaly_only | 0.472 | 350/391 (89.5%) |
| original_stress | test | 8 | anomaly_only | 0.520 | 23/31 (74.2%) |
| original_stress | test | 9 | anomaly_only | 0.528 | 26/55 (47.3%) |
| original_stress | test | 10 | anomaly_only | 0.400 | 432/432 (100.0%) |
| original_stress | test | 7 | random_forest | 0.411 | 279/391 (71.4%) |
| original_stress | test | 8 | random_forest | 0.418 | 22/31 (71.0%) |
| original_stress | test | 9 | random_forest | 0.421 | 35/55 (63.6%) |
| original_stress | test | 10 | random_forest | 0.359 | 388/432 (89.8%) |

## Exploratory test recall by intervention and severity

| Condition | Model | Family and severity | Correct / episodes | Recall |
|---|---|---|---:|---:|
| matched_primary | anomaly_only | bias:gain_level=0.05 | 60/63 | 0.952 |
| matched_primary | anomaly_only | bias:gain_level=0.1 | 58/60 | 0.967 |
| matched_primary | anomaly_only | bias:gain_level=0.2 | 57/61 | 0.934 |
| matched_primary | anomaly_only | freeze:severity=None | 164/181 | 0.906 |
| matched_primary | anomaly_only | gradual_gain:severity=0.05 | 285/305 | 0.934 |
| matched_primary | anomaly_only | gradual_gain:severity=0.1 | 278/302 | 0.921 |
| matched_primary | anomaly_only | gradual_gain:severity=0.2 | 271/302 | 0.897 |
| matched_primary | anomaly_only | noise:severity=0.25 | 57/59 | 0.966 |
| matched_primary | anomaly_only | noise:severity=0.5 | 61/61 | 1.000 |
| matched_primary | anomaly_only | noise:severity=1.0 | 61/61 | 1.000 |
| matched_primary | anomaly_only | replay:severity=None | 167/181 | 0.923 |
| matched_primary | anomaly_only | scaling:severity=0.05 | 58/61 | 0.951 |
| matched_primary | anomaly_only | scaling:severity=0.1 | 57/61 | 0.934 |
| matched_primary | anomaly_only | scaling:severity=0.2 | 53/60 | 0.883 |
| matched_primary | random_forest | bias:gain_level=0.05 | 6/63 | 0.095 |
| matched_primary | random_forest | bias:gain_level=0.1 | 15/60 | 0.250 |
| matched_primary | random_forest | bias:gain_level=0.2 | 9/61 | 0.148 |
| matched_primary | random_forest | freeze:severity=None | 98/181 | 0.541 |
| matched_primary | random_forest | gradual_gain:severity=0.05 | 213/305 | 0.698 |
| matched_primary | random_forest | gradual_gain:severity=0.1 | 199/302 | 0.659 |
| matched_primary | random_forest | gradual_gain:severity=0.2 | 206/302 | 0.682 |
| matched_primary | random_forest | noise:severity=0.25 | 8/59 | 0.136 |
| matched_primary | random_forest | noise:severity=0.5 | 12/61 | 0.197 |
| matched_primary | random_forest | noise:severity=1.0 | 15/61 | 0.246 |
| matched_primary | random_forest | replay:severity=None | 117/181 | 0.646 |
| matched_primary | random_forest | scaling:severity=0.05 | 10/61 | 0.164 |
| matched_primary | random_forest | scaling:severity=0.1 | 8/61 | 0.131 |
| matched_primary | random_forest | scaling:severity=0.2 | 8/60 | 0.133 |
| original_stress | anomaly_only | bias:severity=0.5 | 63/63 | 1.000 |
| original_stress | anomaly_only | bias:severity=1.0 | 59/60 | 0.983 |
| original_stress | anomaly_only | bias:severity=2.0 | 61/61 | 1.000 |
| original_stress | anomaly_only | freeze:severity=None | 164/181 | 0.906 |
| original_stress | anomaly_only | gradual_gain:severity=0.05 | 285/305 | 0.934 |
| original_stress | anomaly_only | gradual_gain:severity=0.1 | 278/302 | 0.921 |
| original_stress | anomaly_only | gradual_gain:severity=0.2 | 271/302 | 0.897 |
| original_stress | anomaly_only | noise:severity=0.25 | 57/59 | 0.966 |
| original_stress | anomaly_only | noise:severity=0.5 | 61/61 | 1.000 |
| original_stress | anomaly_only | noise:severity=1.0 | 61/61 | 1.000 |
| original_stress | anomaly_only | replay:severity=None | 167/181 | 0.923 |
| original_stress | anomaly_only | scaling:severity=0.05 | 58/61 | 0.951 |
| original_stress | anomaly_only | scaling:severity=0.1 | 57/61 | 0.934 |
| original_stress | anomaly_only | scaling:severity=0.2 | 53/60 | 0.883 |
| original_stress | random_forest | bias:severity=0.5 | 13/63 | 0.206 |
| original_stress | random_forest | bias:severity=1.0 | 19/60 | 0.317 |
| original_stress | random_forest | bias:severity=2.0 | 11/61 | 0.180 |
| original_stress | random_forest | freeze:severity=None | 98/181 | 0.541 |
| original_stress | random_forest | gradual_gain:severity=0.05 | 213/305 | 0.698 |
| original_stress | random_forest | gradual_gain:severity=0.1 | 199/302 | 0.659 |
| original_stress | random_forest | gradual_gain:severity=0.2 | 206/302 | 0.682 |
| original_stress | random_forest | noise:severity=0.25 | 8/59 | 0.136 |
| original_stress | random_forest | noise:severity=0.5 | 12/61 | 0.197 |
| original_stress | random_forest | noise:severity=1.0 | 15/61 | 0.246 |
| original_stress | random_forest | replay:severity=None | 117/181 | 0.646 |
| original_stress | random_forest | scaling:severity=0.05 | 10/61 | 0.164 |
| original_stress | random_forest | scaling:severity=0.1 | 8/61 | 0.131 |
| original_stress | random_forest | scaling:severity=0.2 | 8/60 | 0.133 |

The exploratory primary test background recall is low, and both models raise many background alarms. These results do not support reliable deployment or attribution of drift versus attack.

`ml_baselines.json` contains all confusion matrices, per-class precision and recall, binary baseline rates, per-batch metrics, family/severity recall with denominators, and the full validation selection grid. `ml_preflight.json` contains finite-pool reuse and paired magnitude mismatch by split, family, and severity.

## Scientific limitations

- Exact finite-pool bias magnitudes repeat and may form a simulation signature.
- Static checks establish equal shape, source pairing and split isolation; they cannot prove the future model lacks all shortcuts.
- Episode order is artificial and interventions edit extracted features, not sensor waveforms.
- Batch 7-10 evaluation remains exploratory because audit distributions were previously inspected.
- The same primary-trained models and primary-selected threshold are applied to the original-stress condition; stress results are a separate sensitivity analysis.
- Episode order is seeded and artificial. No real-time detection-delay claim is made.
