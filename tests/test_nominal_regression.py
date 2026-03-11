import copy
import io
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np

from orbit_launch.config import EARTH_CONFIG, StarshipBlock2
from orbit_launch.environment import Environment
from orbit_launch.main import solve_optimal_trajectory
from analysis_tools.relabilityanalysis import (
    ReliabilitySuite,
    _run_interval_replay_phase_worker,
    _ms_refine_peak_time,
    evaluate_terminal_state,
)
from orbit_launch.simulation import run_simulation
from orbit_launch.vehicle import Vehicle


class NominalRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = copy.deepcopy(StarshipBlock2)
        cls.env_cfg = copy.deepcopy(EARTH_CONFIG)
        with redirect_stdout(io.StringIO()):
            cls.env = Environment(cls.env_cfg)
            cls.veh = Vehicle(cls.cfg, cls.env)
            cls.opt_res = solve_optimal_trajectory(cls.cfg, cls.veh, cls.env, print_level=0)
            cls.sim_res = run_simulation(cls.opt_res, cls.veh, cls.cfg, rtol=1e-9, atol=1e-12)
        cls.suite = ReliabilitySuite(
            output_dir=Path("tmp/unittest_reliability"),
            save_figures=False,
            show_plots=False,
            analysis_profile="course_core",
        )

    def test_nominal_terminal_state_and_raw_path_are_still_valid(self):
        self.assertTrue(self.opt_res["success"])

        terminal = evaluate_terminal_state(self.sim_res, self.cfg, self.env_cfg)
        traj = self.suite._trajectory_diagnostics(self.sim_res, self.veh, self.cfg)
        path = self.suite._evaluate_path_compliance(traj, self.cfg, q_slack_pa=0.0, g_slack=0.0)

        self.assertTrue(terminal["strict_ok"])
        self.assertTrue(path["path_ok"])
        self.assertLess(abs(float(terminal["spherical_height_err_m"])), 10.0)
        self.assertLessEqual(float(traj["max_q_pa"]), float(self.cfg.max_q_limit))
        self.assertLessEqual(float(traj["max_g"]), float(self.cfg.max_g_load))

    def test_first_boost_interval_replay_audit_stays_small(self):
        n1 = int(self.opt_res["U1"].shape[1])
        dt1 = float(self.opt_res["T1"]) / float(n1)
        payload = {
            "phase_name": "boost",
            "X": self.opt_res["X1"][:, :2],
            "U": self.opt_res["U1"][:, :1],
            "T": dt1,
            "t_start": 0.0,
            "stage_mode": "boost",
            "cfg": self.cfg,
            "env_cfg": self.env_cfg,
        }

        out = _run_interval_replay_phase_worker(payload)

        self.assertEqual(int(out["solver_ok"][0]), 1)
        self.assertLess(float(out["position_error"][0]), 1e-2)
        self.assertLess(float(out["relative_error"][0]), 1e-5)
        self.assertLessEqual(float(out["max_q_pa"][0]), float(self.cfg.max_q_limit))
        self.assertLessEqual(float(out["max_g"][0]), float(self.cfg.max_g_load))


class IntervalReplayAuditLogicTests(unittest.TestCase):
    def test_interval_replay_audit_warns_on_dense_path_limit_violation(self):
        with TemporaryDirectory() as tmpdir:
            suite = ReliabilitySuite(
                output_dir=Path(tmpdir),
                save_figures=False,
                show_plots=False,
                analysis_profile="course_core",
            )
            opt_res = {
                "T1": 1.0,
                "X1": np.zeros((7, 2), dtype=float),
                "U1": np.zeros((4, 1), dtype=float),
                "T3": 1.0,
                "X3": np.zeros((7, 2), dtype=float),
                "U3": np.zeros((4, 1), dtype=float),
            }
            worker_results = {
                "boost": {
                    "phase_name": "boost",
                    "position_error": np.array([1.0e-7], dtype=float),
                    "velocity_error": np.array([1.0e-8], dtype=float),
                    "mass_error": np.array([0.0], dtype=float),
                    "relative_error": np.array([1.0e-9], dtype=float),
                    "max_q_pa": np.array([suite.base_config.max_q_limit + 1.0], dtype=float),
                    "max_g": np.array([suite.base_config.max_g_load - 0.1], dtype=float),
                    "solver_ok": np.array([1], dtype=int),
                },
                "ship": {
                    "phase_name": "ship",
                    "position_error": np.array([1.0e-7], dtype=float),
                    "velocity_error": np.array([1.0e-8], dtype=float),
                    "mass_error": np.array([0.0], dtype=float),
                    "relative_error": np.array([1.0e-9], dtype=float),
                    "max_q_pa": np.array([suite.base_config.max_q_limit - 1.0], dtype=float),
                    "max_g": np.array([suite.base_config.max_g_load - 0.1], dtype=float),
                    "solver_ok": np.array([1], dtype=int),
                },
            }

            def fake_worker(payload):
                return worker_results[str(payload["phase_name"])]

            stdout = io.StringIO()
            with patch.object(suite, "_get_baseline_opt_res", return_value=opt_res):
                with patch("analysis_tools.relabilityanalysis._run_interval_replay_phase_worker", side_effect=fake_worker):
                    with redirect_stdout(stdout):
                        suite.analyze_interval_replay_audit(max_workers=1)

            report = stdout.getvalue()
            self.assertIn("Dense replay Q violations:          1", report)
            self.assertIn("WARN: Interval replay audit found", report)


class PeakRefinementTests(unittest.TestCase):
    def test_reliability_peak_refinement_does_not_overshoot_local_samples(self):
        t = np.array([132.34608764363904, 132.36728549306147, 132.37107089890034], dtype=float)
        y = np.array([3.9680277378095146, 3.968996179007993, 3.9197887624787414], dtype=float)

        t_peak, y_peak = ReliabilitySuite._refine_peak_time(t, y, 1)

        self.assertAlmostEqual(t_peak, float(t[1]), places=12)
        self.assertAlmostEqual(y_peak, float(y[1]), places=12)
        self.assertLessEqual(y_peak, float(np.max(y)))

    def test_worker_peak_refinement_matches_main_reliability_logic(self):
        t = np.array([132.34608764363904, 132.36728549306147, 132.37107089890034], dtype=float)
        y = np.array([3.9680277378095146, 3.968996179007993, 3.9197887624787414], dtype=float)

        t_peak, y_peak = _ms_refine_peak_time(t, y, 1)

        self.assertAlmostEqual(t_peak, float(t[1]), places=12)
        self.assertAlmostEqual(y_peak, float(y[1]), places=12)
        self.assertLessEqual(y_peak, float(np.max(y)))


if __name__ == "__main__":
    unittest.main()
