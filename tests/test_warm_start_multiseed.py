import io
import os
import sys
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from analysis_tools.relabilityanalysis import ReliabilitySuite, WarmStartMultiseedAnalyzer, WarmStartSeedFactory


def _make_cfg():
    return SimpleNamespace(
        sequence=SimpleNamespace(
            min_stage_1_burn=100.0,
            min_stage_2_burn=300.0,
            min_throttle=0.2,
        )
    )


def _make_guess():
    x_hist = np.array(
        [
            [0.0, 10.0, 20.0],
            [1.0, 11.0, 21.0],
            [2.0, 12.0, 22.0],
            [100.0, 110.0, 120.0],
            [200.0, 210.0, 220.0],
            [300.0, 310.0, 320.0],
            [2000.0, 1800.0, 1600.0],
        ],
        dtype=float,
    )
    direction = np.array(
        [
            [1.0, 0.8, 0.6],
            [0.0, 0.4, 0.8],
            [0.0, 0.4, 0.0],
        ],
        dtype=float,
    )
    direction = WarmStartSeedFactory.normalize_direction_history(direction)
    return {
        "T1": 120.0,
        "T2": 15.0,
        "T3": 360.0,
        "X1": x_hist.copy(),
        "X2": x_hist.copy(),
        "X3": x_hist.copy(),
        "TH1": np.array([0.4, 0.7, 0.9], dtype=float),
        "TH3": np.array([0.5, 0.8, 0.95], dtype=float),
        "TD1": direction.copy(),
        "TD3": direction.copy(),
    }


class WarmStartSeedFactoryTests(unittest.TestCase):
    def setUp(self):
        self.factory = WarmStartSeedFactory()
        self.cfg = _make_cfg()
        self.base_guess = _make_guess()

    def test_trial_variants_keep_nominal_then_structured_then_random(self):
        rng = np.random.default_rng(123)

        variants = self.factory.build_trial_variants(self.base_guess, self.cfg, rng, 7)

        self.assertEqual(
            [variant.label for variant in variants],
            ["nominal", "time_minus", "time_plus", "state_minus", "state_plus", "rand_05", "rand_06"],
        )
        self.assertIsNot(variants[0].guess, self.base_guess)
        np.testing.assert_allclose(variants[0].guess["X1"], self.base_guess["X1"])

    def test_trial_variants_enforce_minimum_five_and_default_to_ten_shape(self):
        rng = np.random.default_rng(456)

        variants_min = self.factory.build_trial_variants(self.base_guess, self.cfg, rng, 3)
        self.assertEqual(
            [variant.label for variant in variants_min],
            ["nominal", "time_minus", "time_plus", "state_minus", "state_plus"],
        )

        rng = np.random.default_rng(456)
        variants_ten = self.factory.build_trial_variants(self.base_guess, self.cfg, rng, 10)
        self.assertEqual(
            [variant.label for variant in variants_ten],
            [
                "nominal",
                "time_minus",
                "time_plus",
                "state_minus",
                "state_plus",
                "rand_05",
                "rand_06",
                "rand_07",
                "rand_08",
                "rand_09",
            ],
        )

    def test_randomized_variant_preserves_boundary_state_and_normalizes_controls(self):
        rng = np.random.default_rng(321)

        variant = self.factory.build_randomized_variant(self.base_guess, self.cfg, rng, "rand_05")

        for key in ("X1", "X2", "X3"):
            np.testing.assert_allclose(variant.guess[key][:, 0], self.base_guess[key][:, 0])
            self.assertTrue(np.all(np.diff(variant.guess[key][6, :]) <= 1.0e-12))
            self.assertTrue(np.all(variant.guess[key][6, :] >= 1.0e3))
        for key in ("TD1", "TD3"):
            norms = np.linalg.norm(variant.guess[key], axis=0)
            np.testing.assert_allclose(norms, np.ones_like(norms), atol=1.0e-10)
        for key in ("TH1", "TH3"):
            self.assertTrue(np.all(variant.guess[key] >= self.cfg.sequence.min_throttle))
            self.assertTrue(np.all(variant.guess[key] <= 1.0))
        self.assertGreaterEqual(variant.guess["T1"], self.cfg.sequence.min_stage_1_burn)
        self.assertGreaterEqual(variant.guess["T3"], self.cfg.sequence.min_stage_2_burn)
        self.assertTrue(np.isnan(variant.params.as_dict()["direction_sigma_deg"]))
        self.assertAlmostEqual(variant.params.as_dict()["direction_component_sigma"], 0.01)

    def test_seed_display_labels_are_paper_friendly(self):
        labels = ["nominal", "time_minus", "time_plus", "state_minus", "state_plus", "rand_05", "rand_06"]

        display = WarmStartMultiseedAnalyzer._seed_display_labels(labels)
        breaks = WarmStartMultiseedAnalyzer._seed_group_break_indices(labels)

        self.assertEqual(
            display,
            ["Nominal", "Timing -", "Timing +", "Lateral -", "Lateral +", "Random 1", "Random 2"],
        )
        self.assertEqual(breaks, [1, 5])

    def test_structured_variants_report_explicit_t2_scale(self):
        variants = self.factory.build_structured_variants(self.base_guess, self.cfg)

        self.assertEqual(variants[0].params.as_dict()["t2_scale"], 0.995)
        self.assertEqual(variants[1].params.as_dict()["t2_scale"], 1.005)
        self.assertEqual(variants[2].params.as_dict()["t2_scale"], 1.0)
        self.assertEqual(variants[3].params.as_dict()["t2_scale"], 1.0)

    def test_apply_throttle_bias_uses_available_headroom_and_floor_margin(self):
        throttle = np.array([0.4, 0.6, 0.8], dtype=float)

        plus = self.factory.apply_throttle_bias(throttle, self.cfg.sequence.min_throttle, 0.5)
        minus = self.factory.apply_throttle_bias(throttle, self.cfg.sequence.min_throttle, -0.5)

        np.testing.assert_allclose(plus, np.array([0.4, 0.7, 0.9]))
        np.testing.assert_allclose(minus, np.array([0.4, 0.5, 0.5]))

    def test_structured_seed_defaults_are_orthogonal_under_replay(self):
        time_minus = next(spec for spec in self.factory.structured_specs if spec.label == "time_minus")
        time_plus = next(spec for spec in self.factory.structured_specs if spec.label == "time_plus")
        state_minus = next(spec for spec in self.factory.structured_specs if spec.label == "state_minus")
        state_plus = next(spec for spec in self.factory.structured_specs if spec.label == "state_plus")

        self.assertEqual(time_minus.throttle_sigma, 0.0)
        self.assertEqual(time_plus.throttle_sigma, 0.0)
        self.assertEqual(state_minus.throttle_sigma, 0.0)
        self.assertEqual(state_plus.throttle_sigma, 0.0)
        self.assertEqual(time_minus.direction_axis, (0.0, 0.0, 1.0))
        self.assertEqual(time_plus.direction_axis, (0.0, 0.0, 1.0))
        self.assertEqual(state_minus.direction_axis, (0.0, 0.0, 1.0))
        self.assertEqual(state_plus.direction_axis, (0.0, 0.0, 1.0))

    def test_state_rebuilder_callback_replaces_state_histories(self):
        rng = np.random.default_rng(321)
        sentinel = np.full_like(self.base_guess["X1"], 42.0)

        def rebuilder(guess):
            rebuilt = dict(guess)
            rebuilt["X1"] = sentinel.copy()
            rebuilt["X2"] = sentinel.copy()
            rebuilt["X3"] = sentinel.copy()
            return rebuilt

        variant = self.factory.build_randomized_variant(
            self.base_guess,
            self.cfg,
            rng,
            "rand_05",
            state_rebuilder=rebuilder,
        )

        for key in ("X1", "X2", "X3"):
            np.testing.assert_allclose(variant.guess[key], sentinel)
        for key in ("TD1", "TD3"):
            norms = np.linalg.norm(variant.guess[key], axis=0)
            np.testing.assert_allclose(norms, np.ones_like(norms), atol=1.0e-10)


class WarmStartMultiseedDelegationTests(unittest.TestCase):
    def test_reliability_suite_delegates_to_new_analyzer(self):
        with TemporaryDirectory() as tmpdir:
            with redirect_stdout(io.StringIO()):
                suite = ReliabilitySuite(
                    output_dir=Path(tmpdir),
                    save_figures=False,
                    show_plots=False,
                    analysis_profile="course_core",
                )

            with patch("analysis_tools.relabilityanalysis.WarmStartMultiseedAnalyzer") as analyzer_cls:
                suite.analyze_randomized_multistart(n_trials=9)

            analyzer_cls.assert_called_once()
            analyzer_cls.return_value.analyze.assert_called_once_with(n_trials=9)

    def test_reliability_suite_default_multiseed_count_is_ten(self):
        with TemporaryDirectory() as tmpdir:
            with redirect_stdout(io.StringIO()):
                suite = ReliabilitySuite(
                    output_dir=Path(tmpdir),
                    save_figures=False,
                    show_plots=False,
                    analysis_profile="course_core",
                )

            with patch("analysis_tools.relabilityanalysis.WarmStartMultiseedAnalyzer") as analyzer_cls:
                suite.analyze_randomized_multistart()

            analyzer_cls.assert_called_once()
            analyzer_cls.return_value.analyze.assert_called_once_with(n_trials=10)


class WarmStartMultiseedLoggingTests(unittest.TestCase):
    def test_trial_output_is_tee_written_into_logs_folder(self):
        with TemporaryDirectory() as tmpdir:
            suite = SimpleNamespace(output_dir=Path(tmpdir))
            analyzer = WarmStartMultiseedAnalyzer(suite, lambda *_args, **_kwargs: {})
            log_path = analyzer._trial_log_path(0, "time+plus/test")

            console = io.StringIO()
            with redirect_stdout(console):
                with analyzer._tee_trial_output(log_path):
                    print("trial line")

            self.assertEqual(log_path.parent, Path(tmpdir) / "logs")
            self.assertTrue(log_path.exists())
            self.assertIn("trial line", console.getvalue())
            self.assertIn("trial line", log_path.read_text(encoding="utf-8"))

    def test_trial_output_logs_low_level_stdout_and_stderr_writes(self):
        stdout_fd = getattr(sys.__stdout__, "fileno", None)
        stderr_fd = getattr(sys.__stderr__, "fileno", None)
        if stdout_fd is None or stderr_fd is None:
            self.skipTest("Native stdout/stderr file descriptors are unavailable.")

        with TemporaryDirectory() as tmpdir:
            suite = SimpleNamespace(output_dir=Path(tmpdir))
            analyzer = WarmStartMultiseedAnalyzer(suite, lambda *_args, **_kwargs: {})
            log_path = analyzer._trial_log_path(1, "native-fd")

            with analyzer._tee_trial_output(log_path):
                print("python line")
                os.write(sys.__stdout__.fileno(), b"native stdout line\n")
                os.write(sys.__stderr__.fileno(), b"native stderr line\n")

            log_text = log_path.read_text(encoding="utf-8")
            self.assertIn("python line", log_text)
            self.assertIn("native stdout line", log_text)
            self.assertIn("native stderr line", log_text)

    def test_multiple_trial_logs_are_created_sequentially(self):
        with TemporaryDirectory() as tmpdir:
            suite = SimpleNamespace(output_dir=Path(tmpdir))
            analyzer = WarmStartMultiseedAnalyzer(suite, lambda *_args, **_kwargs: {})

            labels = ["nominal", "time_minus", "time_plus", "state_minus", "state_plus", "rand_05"]
            for idx, label in enumerate(labels):
                with analyzer._tee_trial_output(analyzer._trial_log_path(idx, label)):
                    print(f"log for {label}")

            created = sorted(p.name for p in (Path(tmpdir) / "logs").glob("*.log"))
            self.assertEqual(
                created,
                [
                    "01_nominal.log",
                    "02_time_minus.log",
                    "03_time_plus.log",
                    "04_state_minus.log",
                    "05_state_plus.log",
                    "06_rand_05.log",
                ],
            )

    def test_insertion_zoom_window_uses_terminal_segment_of_profiles(self):
        suite = SimpleNamespace(output_dir=Path("tmp/test_logs"))
        analyzer = WarmStartMultiseedAnalyzer(suite, lambda *_args, **_kwargs: {})
        profiles = [
            {
                "progress": np.array([0.0, 0.5, 0.85, 0.95, 1.0]),
                "downrange_km": np.array([0.0, 100.0, 180.0, 195.0, 200.0]),
                "ellipsoidal_altitude_km": np.array([0.0, 50.0, 160.0, 190.0, 200.0]),
            },
            {
                "progress": np.array([0.0, 0.5, 0.90, 0.97, 1.0]),
                "downrange_km": np.array([0.0, 100.0, 181.0, 196.0, 201.0]),
                "ellipsoidal_altitude_km": np.array([0.0, 50.0, 161.0, 191.0, 201.0]),
            },
        ]

        window = analyzer._compute_insertion_zoom_window(profiles)

        self.assertIsNotNone(window)
        (x_lo, x_hi), (y_lo, y_hi) = window
        self.assertLess(x_lo, 180.0)
        self.assertGreater(x_hi, 201.0)
        self.assertLess(y_lo, 160.0)
        self.assertGreater(y_hi, 201.0)

    def test_planned_labels_match_nominal_structured_then_random_order(self):
        suite = SimpleNamespace(output_dir=Path("tmp/test_logs"))
        analyzer = WarmStartMultiseedAnalyzer(suite, lambda *_args, **_kwargs: {})

        self.assertEqual(
            analyzer._planned_labels(10),
            [
                "nominal",
                "time_minus",
                "time_plus",
                "state_minus",
                "state_plus",
                "rand_05",
                "rand_06",
                "rand_07",
                "rand_08",
                "rand_09",
            ],
        )


class WarmStartMultiseedHelperTests(unittest.TestCase):
    def test_solver_return_status_classification_preserves_failure_modes(self):
        suite = SimpleNamespace(output_dir=Path("tmp/test_logs"))
        analyzer = WarmStartMultiseedAnalyzer(suite, lambda *_args, **_kwargs: {})

        self.assertEqual(analyzer._classify_solver_return_status("Solve_Succeeded"), "SOLVE_OK")
        self.assertEqual(analyzer._classify_solver_return_status("Restoration_Failed"), "RESTORATION_FAIL")
        self.assertEqual(analyzer._classify_solver_return_status("Maximum_Iterations_Exceeded"), "MAX_ITER")
        self.assertEqual(analyzer._classify_solver_return_status("NonIpopt_Exception_Thrown"), "INTERRUPTED")
        self.assertEqual(analyzer._classify_solver_return_status(""), "SOLVE_FAIL")

    def test_terminal_criteria_ratio_uses_worst_normalized_component(self):
        suite = SimpleNamespace(output_dir=Path("tmp/test_logs"))
        analyzer = WarmStartMultiseedAnalyzer(suite, lambda *_args, **_kwargs: {})

        ratio = analyzer._terminal_criteria_ratio(
            {
                "alt_err_m": 2500.0,
                "vel_err_m_s": -120.0,
                "radial_velocity_m_s": 1.0,
                "inclination_err_deg": 0.1,
            }
        )

        self.assertAlmostEqual(ratio, 1.2)

    def test_realized_seed_metrics_report_scales_and_zero_delta_for_nominal(self):
        suite = SimpleNamespace(output_dir=Path("tmp/test_logs"))
        analyzer = WarmStartMultiseedAnalyzer(suite, lambda *_args, **_kwargs: {})
        guess = _make_guess()

        metrics = analyzer._compute_realized_seed_metrics(guess, guess)

        self.assertAlmostEqual(metrics["realized_t1_scale"], 1.0)
        self.assertAlmostEqual(metrics["realized_t2_scale"], 1.0)
        self.assertAlmostEqual(metrics["realized_t3_scale"], 1.0)
        self.assertAlmostEqual(metrics["realized_position_rms_m"], 0.0)
        self.assertAlmostEqual(metrics["realized_velocity_rms_m_s"], 0.0)
        self.assertAlmostEqual(metrics["realized_mass_rms_kg"], 0.0)
        self.assertAlmostEqual(metrics["realized_throttle_rms_abs"], 0.0)
        self.assertAlmostEqual(metrics["realized_direction_rms_deg"], 0.0)

    def test_configured_seed_metadata_zeroes_state_perturbations_when_rebuilt(self):
        suite = SimpleNamespace(output_dir=Path("tmp/test_logs"))
        analyzer = WarmStartMultiseedAnalyzer(suite, lambda *_args, **_kwargs: {})
        params = {
            "pos_sigma": 0.01,
            "vel_sigma": 0.02,
            "mass_sigma": 0.03,
            "throttle_sigma": 0.04,
            "direction_sigma_deg": -0.35,
            "direction_component_sigma": np.nan,
        }

        metadata = analyzer._configured_seed_metadata("state_plus", params, rebuild_states=True)

        self.assertEqual(metadata["seed_family"], "structured")
        self.assertEqual(metadata["state_rebuild_applied"], 1)
        self.assertEqual(metadata["configured_position_perturbation"], 0.0)
        self.assertEqual(metadata["configured_velocity_perturbation"], 0.0)
        self.assertEqual(metadata["configured_mass_perturbation"], 0.0)
        self.assertAlmostEqual(metadata["configured_direction_perturbation_deg"], -0.35)
        self.assertAlmostEqual(metadata["configured_direction_rms_deg_est"], 0.35)

    def test_nominal_configured_seed_metadata_does_not_fake_state_rebuild(self):
        suite = SimpleNamespace(output_dir=Path("tmp/test_logs"))
        analyzer = WarmStartMultiseedAnalyzer(suite, lambda *_args, **_kwargs: {})
        params = {
            "pos_sigma": 0.0,
            "vel_sigma": 0.0,
            "mass_sigma": 0.0,
            "throttle_sigma": 0.0,
            "direction_sigma_deg": 0.0,
            "direction_component_sigma": 0.0,
        }

        metadata = analyzer._configured_seed_metadata("nominal", params, rebuild_states=False)

        self.assertEqual(metadata["seed_family"], "nominal")
        self.assertEqual(metadata["perturbation_semantics"], "none")
        self.assertEqual(metadata["state_rebuild_applied"], 0)
        self.assertEqual(metadata["configured_direction_rms_deg_est"], 0.0)

    def test_configured_seed_metadata_reports_random_direction_component_sigma(self):
        suite = SimpleNamespace(output_dir=Path("tmp/test_logs"))
        analyzer = WarmStartMultiseedAnalyzer(suite, lambda *_args, **_kwargs: {})
        params = {
            "pos_sigma": 0.01,
            "vel_sigma": 0.01,
            "mass_sigma": 0.01,
            "throttle_sigma": 0.01,
            "direction_sigma_deg": np.nan,
            "direction_component_sigma": 0.01,
        }

        metadata = analyzer._configured_seed_metadata("rand_05", params, rebuild_states=True)

        self.assertEqual(metadata["seed_family"], "randomized")
        self.assertTrue(np.isnan(metadata["configured_direction_perturbation_deg"]))
        self.assertAlmostEqual(metadata["configured_direction_component_sigma"], 0.01)
        self.assertAlmostEqual(
            metadata["configured_direction_rms_deg_est"],
            analyzer._direction_component_sigma_to_rms_deg(0.01),
        )

    def test_spread_candidate_indices_only_include_non_reference_raw_passes(self):
        suite = SimpleNamespace(output_dir=Path("tmp/test_logs"))
        analyzer = WarmStartMultiseedAnalyzer(suite, lambda *_args, **_kwargs: {})

        indices = analyzer._spread_candidate_indices(0, [0, 2, 4])

        self.assertEqual(indices, {2, 4})

    def test_runtime_summary_medians_split_non_raw_from_true_failures(self):
        suite = SimpleNamespace(output_dir=Path("tmp/test_logs"))
        analyzer = WarmStartMultiseedAnalyzer(suite, lambda *_args, **_kwargs: {})
        rows = [
            {"runtime_s": 10.0, "mission_success": 1, "mission_success_slack": 1},
            {"runtime_s": 20.0, "mission_success": 0, "mission_success_slack": 1},
            {"runtime_s": 40.0, "mission_success": 0, "mission_success_slack": 0},
        ]

        medians = analyzer._runtime_summary_medians(rows)

        self.assertAlmostEqual(medians["median_success_runtime_s"], 10.0)
        self.assertAlmostEqual(medians["median_non_raw_runtime_s"], 30.0)
        self.assertAlmostEqual(medians["median_failure_runtime_s"], 40.0)


if __name__ == "__main__":
    unittest.main()
