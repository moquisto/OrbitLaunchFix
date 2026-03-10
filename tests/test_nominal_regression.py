import copy
import io
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np

from orbit_launch.config import EARTH_CONFIG, StarshipBlock2
from orbit_launch.environment import Environment
from orbit_launch.main import solve_optimal_trajectory
from analysis_tools.relabilityanalysis import (
    ReliabilitySuite,
    _run_interval_replay_phase_worker,
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


if __name__ == "__main__":
    unittest.main()
