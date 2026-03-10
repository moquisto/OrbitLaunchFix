import copy
import unittest

import numpy as np

from orbit_launch.config import EARTH_CONFIG, StarshipBlock2
from analysis_tools.relabilityanalysis import evaluate_terminal_state
from orbit_launch.trajectory_metrics import (
    circular_target_speed_m_s,
    ellipsoidal_altitude_m,
    spherical_altitude_m,
    target_orbit_radius_m,
)


class TrajectoryMetricTests(unittest.TestCase):
    def test_equatorial_altitude_metrics_match_target_height(self):
        env_cfg = copy.deepcopy(EARTH_CONFIG)
        target_alt = 420_000.0
        position = np.array([env_cfg.earth_radius_equator + target_alt, 0.0, 0.0], dtype=float)

        self.assertAlmostEqual(spherical_altitude_m(position, env_cfg), target_alt, places=6)
        self.assertAlmostEqual(ellipsoidal_altitude_m(position, env_cfg), target_alt, places=6)

    def test_terminal_state_reports_explicit_height_metrics(self):
        cfg = copy.deepcopy(StarshipBlock2)
        env_cfg = copy.deepcopy(EARTH_CONFIG)
        cfg.target_inclination = 0.0

        target_alt = float(cfg.target_altitude)
        radius = target_orbit_radius_m(target_alt, env_cfg)
        speed = circular_target_speed_m_s(target_alt, env_cfg)
        final_mass = cfg.stage_2.dry_mass + cfg.payload_mass + 1_000.0
        state = np.array([radius, 0.0, 0.0, 0.0, speed, 0.0, final_mass], dtype=float)
        sim_res = {
            "t": np.array([0.0], dtype=float),
            "y": state.reshape(7, 1),
            "u": np.zeros((4, 1), dtype=float),
        }

        metrics = evaluate_terminal_state(sim_res, cfg, env_cfg)

        self.assertTrue(metrics["terminal_valid"])
        self.assertTrue(metrics["strict_ok"])
        self.assertAlmostEqual(metrics["target_orbit_radius_m"], radius, places=6)
        self.assertAlmostEqual(metrics["spherical_height_m"], target_alt, places=6)
        self.assertAlmostEqual(metrics["ellipsoidal_altitude_m"], target_alt, places=6)
        self.assertAlmostEqual(metrics["spherical_height_err_m"], 0.0, places=6)
        self.assertAlmostEqual(metrics["vel_err_m_s"], 0.0, places=6)


if __name__ == "__main__":
    unittest.main()
