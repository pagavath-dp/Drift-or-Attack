import math
import subprocess
import sys
import unittest
from dataclasses import replace

from drift_attack.dataset import Measurement
from drift_attack.episodes import (ACTIVE_POSITIONS, ATTACK_FAMILIES, BIAS_LEVELS,
                                   changed_positions, construct_episodes,
                                   fit_generator_scales, fit_magnitude_calibration,
                                   generate_variants, matched_bias_magnitude,
                                   standardized_change)


def source(batch, line, gas=1, concentration=10.0):
    # Distinct vectors in every batch and row, with nonzero training spread.
    base = batch * 10000 + line * 17
    return Measurement(batch, line, gas, concentration,
                       tuple(float(base + j + (line % 3) * j / 10) for j in range(128)))


def fixture():
    return {batch: [source(batch, line) for line in range(1, 17)] for batch in range(1, 11)}


class EpisodeIsolationTests(unittest.TestCase):
    def test_no_source_reuse_and_partition_preservation(self):
        episodes, coverage = construct_episodes(fixture())
        self.assertEqual(len(episodes), 20)
        self.assertTrue(all(coverage[b]["episodes"] == 2 for b in range(1, 11)))
        ids = [source_id for episode in episodes for source_id in episode.source_ids]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertTrue(all(all(source_id[0] == episode.batch for source_id in episode.source_ids)
                            for episode in episodes))

    def test_exact_feature_duplicate_across_partitions_is_rejected(self):
        batches = fixture()
        duplicate = batches[2][0]
        batches[7][0] = replace(batches[7][0], features=duplicate.features)
        with self.assertRaisesRegex(ValueError, "exact feature-vector duplicate"):
            construct_episodes(batches)

    def test_generator_scales_do_not_depend_on_validation_or_test(self):
        batches = fixture()
        original = fit_generator_scales(batches)
        batches[7] = [replace(row, features=tuple(value * 1e6 for value in row.features))
                      for row in batches[7]]
        self.assertEqual(fit_generator_scales(batches), original)

    def test_episode_order_and_variants_are_deterministic(self):
        episodes1, _ = construct_episodes(fixture())
        episodes2, _ = construct_episodes(fixture())
        self.assertEqual(episodes1, episodes2)
        episode = next(episode for episode in episodes1 if episode.batch == 2)
        scales = fit_generator_scales(fixture())
        self.assertEqual(generate_variants(episode, scales), generate_variants(episode, scales))


class PairedVariantTests(unittest.TestCase):
    def setUp(self):
        batches = fixture()
        self.episode = next(episode for episode in construct_episodes(batches)[0] if episode.batch == 2)
        self.scales = fit_generator_scales(batches)

    def test_all_families_keep_pairing_order_duration_and_sensor_scope(self):
        for index, family in enumerate(ATTACK_FAMILIES):
            episode = replace(self.episode, rotation_index=index)
            background, drift, attack = generate_variants(episode, self.scales)
            self.assertEqual(attack.family, family)
            self.assertEqual({variant.label for variant in (background, drift, attack)},
                             {"background", "synthetic_drift", "synthetic_attack"})
            self.assertEqual({variant.source_ids for variant in (background, drift, attack)},
                             {episode.source_ids})
            self.assertEqual({variant.split for variant in (background, drift, attack)},
                             {episode.split})
            self.assertEqual({variant.sensor for variant in (background, drift, attack)},
                             {background.sensor})
            self.assertEqual(changed_positions(episode, background), ())
            self.assertEqual(changed_positions(episode, drift), ACTIVE_POSITIONS)
            self.assertEqual(changed_positions(episode, attack), ACTIVE_POSITIONS)
            self.assertTrue(all(len(variant.rows) == 8 for variant in (background, drift, attack)))
            for variant in (drift, attack):
                self.assertTrue(all(math.isfinite(value) for row in variant.rows for value in row))
                for t in range(8):
                    for j in range(128):
                        if t not in ACTIVE_POSITIONS or not 8 * variant.sensor <= j < 8 * variant.sensor + 8:
                            self.assertEqual(variant.rows[t][j], episode.rows[t][j])

    def test_freeze_and_replay_copy_only_prior_source_values(self):
        freeze = generate_variants(replace(self.episode, rotation_index=3), self.scales)[2]
        replay = generate_variants(replace(self.episode, rotation_index=4), self.scales)[2]
        block = range(8 * freeze.sensor, 8 * freeze.sensor + 8)
        for t in ACTIVE_POSITIONS:
            for j in block:
                self.assertEqual(freeze.rows[t][j], self.episode.rows[4][j])
                self.assertEqual(replay.rows[t][j], self.episode.rows[t - 4][j])


class MagnitudeCalibrationTests(unittest.TestCase):
    def setUp(self):
        self.batches = fixture()
        self.episodes, _ = construct_episodes(self.batches)
        self.scales = fit_generator_scales(self.batches)
        self.calibration = fit_magnitude_calibration(self.episodes, self.scales)

    def test_only_training_episodes_enter_finite_positive_pools(self):
        self.assertEqual(len(self.calibration.source_episodes), 6)
        self.assertEqual({record[1] for record in self.calibration.source_episodes}, {2, 3, 4})
        self.assertTrue(all(source_id[0] == batch for _, batch, ids, _ in
                            self.calibration.source_episodes for source_id in ids))
        for pool in self.calibration.pools:
            self.assertEqual(len(pool), 6)
            self.assertTrue(all(math.isfinite(value) and value > 0 for value in pool))

    def test_validation_and_test_changes_cannot_recalibrate(self):
        changed = fixture()
        for batch in range(5, 11):
            changed[batch] = [replace(row, features=tuple(value * 1e6 for value in row.features))
                              for row in changed[batch]]
        other_episodes, _ = construct_episodes(changed)
        other_scales = fit_generator_scales(changed)
        other = fit_magnitude_calibration(other_episodes, other_scales)
        self.assertEqual(other_scales, self.scales)
        self.assertEqual(other, self.calibration)

    def test_pool_values_equal_training_drift_effects(self):
        by_id = {item.episode_id: item for item in self.episodes if item.split == "train"}
        for episode_id, batch, source_ids, magnitudes in self.calibration.source_episodes:
            episode = by_id[episode_id]
            self.assertEqual(batch, episode.batch)
            self.assertEqual(source_ids, episode.source_ids)
            for level, magnitude in enumerate(magnitudes):
                drift = generate_variants(replace(episode, rotation_index=level), self.scales)[1]
                self.assertAlmostEqual(magnitude, standardized_change(episode, drift, self.scales))
        for level, pool in enumerate(self.calibration.pools):
            self.assertEqual(pool, tuple(sorted(record[3][level] for record in
                                                self.calibration.source_episodes)))

    def test_matched_bias_magnitude_and_original_grid(self):
        self.assertEqual(BIAS_LEVELS, (0.5, 1.0, 2.0))
        episode = next(item for item in self.episodes if item.batch == 2)
        for level in range(3):
            rotation = next(index for index in range(15) if index % 5 == 0 and index % 3 == level)
            bias_episode = replace(episode, rotation_index=rotation)
            primary = generate_variants(bias_episode, self.scales,
                                        bias_condition="matched_primary", calibration=self.calibration)
            stress = generate_variants(bias_episode, self.scales,
                                       bias_condition="original_stress")
            background, drift, bias = primary
            self.assertEqual(background.rows, bias_episode.rows)
            self.assertEqual(drift, stress[1])
            self.assertEqual(bias.severity, matched_bias_magnitude(self.calibration, bias_episode, level))
            self.assertAlmostEqual(standardized_change(bias_episode, bias, self.scales), bias.severity,
                                   delta=1e-7 * max(1, bias.severity))
            self.assertEqual(stress[2].severity, BIAS_LEVELS[level])
            self.assertEqual(changed_positions(bias_episode, bias), ACTIVE_POSITIONS)
            self.assertEqual({item.split for item in primary}, {bias_episode.split})
            self.assertEqual({item.source_ids for item in primary}, {bias_episode.source_ids})

    def test_nonbias_attacks_unchanged_between_conditions(self):
        episode = next(item for item in self.episodes if item.batch == 5)
        for family_index in range(1, 5):
            selected = replace(episode, rotation_index=family_index)
            primary = generate_variants(selected, self.scales,
                                        bias_condition="matched_primary", calibration=self.calibration)
            stress = generate_variants(selected, self.scales, bias_condition="original_stress")
            self.assertEqual(primary, stress)

    def test_validation_and_test_bias_draw_from_frozen_training_pool(self):
        for split in ("validation", "test"):
            episode = next(item for item in self.episodes if item.split == split)
            selected = replace(episode, rotation_index=0)
            _, _, attack = generate_variants(selected, self.scales,
                                             bias_condition="matched_primary", calibration=self.calibration)
            self.assertIn(attack.severity, self.calibration.pools[0])
            self.assertEqual(attack.source_ids, selected.source_ids)
            self.assertEqual(attack.split, split)

    def test_invalid_or_mismatched_calibration_is_rejected(self):
        episode = replace(next(item for item in self.episodes if item.batch == 2), rotation_index=0)
        with self.assertRaisesRegex(ValueError, "requires calibration"):
            generate_variants(episode, self.scales, bias_condition="matched_primary")
        with self.assertRaisesRegex(ValueError, "requires calibration"):
            generate_variants(episode, tuple(value * 2 for value in self.scales),
                              bias_condition="matched_primary", calibration=self.calibration)
        tampered = replace(self.calibration, pools=((), *self.calibration.pools[1:]))
        with self.assertRaisesRegex(ValueError, "digest mismatch"):
            matched_bias_magnitude(tampered, episode, 0)

    def test_digest_and_draw_repeat_across_processes(self):
        episode = replace(next(item for item in self.episodes if item.batch == 2), rotation_index=0)
        expected = f"{self.calibration.digest} {matched_bias_magnitude(self.calibration, episode, 0):.17g}"
        script = ("import sys; sys.path.insert(0, 'tests'); "
                  "from test_episodes import fixture; "
                  "from drift_attack.episodes import construct_episodes, fit_generator_scales, "
                  "fit_magnitude_calibration, matched_bias_magnitude; "
                  "b=fixture(); e=construct_episodes(b)[0]; "
                  "c=fit_magnitude_calibration(e,fit_generator_scales(b)); "
                  "x=next(x for x in e if x.batch==2); "
                  "print(c.digest, format(matched_bias_magnitude(c,x,0),'.17g'))")
        for _ in range(2):
            self.assertEqual(subprocess.check_output([sys.executable, "-c", script], text=True).strip(),
                             expected)
