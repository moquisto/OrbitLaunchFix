# unittest.py

import unittest
import numpy as np
from oldconfig import EnvConfig
from oldenvironment import Environment, EnvironmentalState

class TestEnvironment(unittest.TestCase):

    def setUp(self):
        """Set up a test environment before each test."""
        self.config = EnvConfig()
        self.environment = Environment(self.config)

    def test_initialization(self):
        """Test if the Environment class initializes correctly."""
        self.assertIsInstance(self.environment, Environment)
        self.assertEqual(self.environment.config, self.config)
        
        # Check lookup table properties
        expected_points = int(self.config.atmosphere_max_alt / self.config.atmosphere_step) + 1
        self.assertEqual(self.environment.grid_altitudes.shape, (expected_points,))
        self.assertEqual(self.environment.table_rho.shape, (expected_points,))
        self.assertEqual(self.environment.table_temp.shape, (expected_points,))
        self.assertEqual(self.environment.table_pressure.shape, (expected_points,))
        
        # Check if initial values are reasonable (sea level)
        self.assertAlmostEqual(self.environment.grid_altitudes[0], 0.0, places=5)
        self.assertAlmostEqual(self.environment.table_rho[0], 1.225, places=3) # Standard sea level density
        self.assertAlmostEqual(self.environment.table_temp[0], 288.15, places=2) # Standard sea level temp
        self.assertAlmostEqual(self.environment.table_pressure[0], 101325.0, places=0) # Standard sea level pressure

    def test_get_state_at_sea_level_equator(self):
        """Test get_state at the equator at sea level."""
        r_eci = np.array([self.config.earth_radius_equator, 0, 0])
        state = self.environment.get_state(0, r_eci)
        
        self.assertIsInstance(state, EnvironmentalState)
        
        # Check atmospheric properties (should be close to standard sea level)
        self.assertAlmostEqual(state.density, 1.225, places=3)
        self.assertAlmostEqual(state.temperature, 288.15, places=2)
        self.assertAlmostEqual(state.pressure, 101325.0, places=0)
        self.assertAlmostEqual(state.speed_of_sound, 340.297, places=3)

        # Check wind (v_wind = omega x r) -> should be in +y direction
        expected_wind_y = self.config.earth_omega_vector[2] * self.config.earth_radius_equator
        self.assertAlmostEqual(state.wind_velocity[0], 0)
        self.assertAlmostEqual(state.wind_velocity[1], expected_wind_y, places=2)
        self.assertAlmostEqual(state.wind_velocity[2], 0)

        # Check gravity (should be mostly in -x direction, with small J2 effect)
        g_central = -self.config.earth_mu / self.config.earth_radius_equator**2
        self.assertGreater(abs(state.gravity_vector[0]), abs(g_central)) # J2 adds to gravity at equator
        self.assertAlmostEqual(state.gravity_vector[1], 0, places=5)
        self.assertAlmostEqual(state.gravity_vector[2], 0, places=5)


    def test_get_state_in_vacuum(self):
        """Test get_state in a vacuum (above max_altitude)."""
        alt = self.config.atmosphere_max_alt + 10000 # 10km above the limit
        r_eci = np.array([self.config.earth_radius_equator + alt, 0, 0])
        state = self.environment.get_state(0, r_eci)

        self.assertEqual(state.density, 0.0)
        self.assertEqual(state.pressure, 0.0)
        self.assertEqual(state.speed_of_sound, 0.0)
        np.testing.assert_array_equal(state.wind_velocity, np.array([0,0,0]))
        
        # Gravity should still exist
        self.assertNotEqual(state.gravity_vector[0], 0)


    def test_gravity_central_only(self):
        """Test the gravity calculation with J2 perturbation disabled."""
        # Temporarily disable J2
        self.config.use_j2_perturbation = False
        self.environment = Environment(self.config) # Re-initialize
        
        r_eci = np.array([self.config.earth_radius_equator, 0, 0])
        r_norm = np.linalg.norm(r_eci)
        
        g_vec = self.environment._calculate_gravity(r_eci, r_norm)
        
        expected_g = -self.config.earth_mu / r_norm**2
        expected_g_vec = np.array([expected_g, 0, 0])
        
        np.testing.assert_allclose(g_vec, expected_g_vec, rtol=1e-7)

        # Reset config for other tests
        self.config.use_j2_perturbation = True

    def test_gravity_with_j2_at_pole(self):
        """Test J2 perturbation effect at the North Pole."""
        # At the pole, the J2 perturbation opposes central gravity, making you "lighter"
        r_eci = np.array([0, 0, self.config.earth_radius_equator]) # Approx radius at pole
        state = self.environment.get_state(0, r_eci)
        
        g_central_mag = self.config.earth_mu / np.linalg.norm(r_eci)**2
        g_total_mag = np.linalg.norm(state.gravity_vector)
        
        # Total gravity at pole should be slightly less than central gravity
        self.assertLess(g_total_mag, g_central_mag)
        
        # Check direction
        self.assertAlmostEqual(state.gravity_vector[0], 0, places=5)
        self.assertAlmostEqual(state.gravity_vector[1], 0, places=5)
        self.assertLess(state.gravity_vector[2], 0)

    def test_geodetic_altitude(self):
        """Test the geodetic altitude calculation is reasonable."""
        flattening = 1.0 / 298.257
        # At the pole, the surface radius is smaller
        polar_radius = self.config.earth_radius_equator * (1.0 - flattening)
        
        # Position at the pole, on the surface
        r_eci_pole = np.array([0, 0, polar_radius])
        
        # The internal altitude calculation should be close to zero here
        # We can't access altitude directly, but we can check the state
        state_pole = self.environment.get_state(0, r_eci_pole)
        
        # Density should be sea-level density because altitude is ~0
        self.assertAlmostEqual(state_pole.density, 1.225, places=3)
        
    def test_j2_perturbation_exact_values(self):
        """
        Quantitative test for J2 perturbation.
        Verifies the calculated vector against the analytical formula at a known point (45 deg lat).
        """
        # Setup a specific point: 45 degrees latitude, distance R_earth
        # We use 45 degrees because it makes sin^2(lat) = 0.5, which exercises the math well.
        R = self.config.earth_radius_equator
        r_eci = np.array([R/np.sqrt(2), 0, R/np.sqrt(2)])
        r_norm = R
        
        # Calculate expected J2 acceleration manually
        mu = self.config.earth_mu
        J2 = self.config.j2_constant
        x, y, z = r_eci
        
        # (z/r)^2 = 0.5
        z_r_sq = 0.5
        
        # Factor from J2 formula: -1.5 * J2 * mu * R^2 / r^5
        # Since r=R here, this simplifies to: -1.5 * J2 * mu / R^3
        factor = -1.5 * J2 * mu / (R**3)
        
        # Calculate components based on the formula: factor * component * (term)
        expected_ax = factor * x * (1 - 5 * z_r_sq) 
        expected_ay = factor * y * (1 - 5 * z_r_sq) 
        expected_az = factor * z * (3 - 5 * z_r_sq) 
        
        expected_j2_vec = np.array([expected_ax, expected_ay, expected_az])
        
        # Get actual gravity from environment and isolate J2 part by subtracting central gravity
        state = self.environment.get_state(0, r_eci)
        g_central = -mu / (r_norm**3) * r_eci
        g_j2_actual = state.gravity_vector - g_central
        
        np.testing.assert_allclose(g_j2_actual, expected_j2_vec, rtol=1e-9, err_msg="J2 vector math is incorrect")

    def test_atmosphere_interpolation(self):
        """
        Test that atmospheric properties are interpolated correctly between grid points.
        This ensures we aren't just snapping to the nearest 5m step.
        """
        # Choose an altitude between grid points (step is 5.0m)
        # 1002.5 meters
        alt_mid = 1002.5
        r_eci = np.array([self.config.earth_radius_equator + alt_mid, 0, 0])
        
        state_mid = self.environment.get_state(0, r_eci)
        
        # Get grid points around it
        r_low = np.array([self.config.earth_radius_equator + 1000.0, 0, 0])
        r_high = np.array([self.config.earth_radius_equator + 1005.0, 0, 0])
        
        state_low = self.environment.get_state(0, r_low)
        state_high = self.environment.get_state(0, r_high)
        
        # Check Density Interpolation
        # Should be roughly the average (linear interpolation)
        expected_rho = (state_low.density + state_high.density) / 2.0
        self.assertAlmostEqual(state_mid.density, expected_rho, places=5)

    def test_underground_robustness(self):
        """
        Test that the environment handles positions below the surface.
        It should clamp to surface values rather than crashing or extrapolating.
        """
        # Position 100m underground
        r_eci = np.array([self.config.earth_radius_equator - 100.0, 0, 0])
        
        # Should not raise error
        state = self.environment.get_state(0, r_eci)
        
        # Should be clamped to sea level values (index 0)
        self.assertAlmostEqual(state.density, 1.225, places=3)
        self.assertAlmostEqual(state.pressure, 101325.0, places=0)

    def test_wind_variation_with_altitude(self):
        """Test that wind velocity increases with altitude (co-rotation)."""
        # Surface
        r_surf = np.array([self.config.earth_radius_equator, 0, 0])
        state_surf = self.environment.get_state(0, r_surf)
        v_wind_surf = np.linalg.norm(state_surf.wind_velocity)
        
        # High Altitude (e.g. 100km)
        r_high = np.array([self.config.earth_radius_equator + 100_000.0, 0, 0])
        state_high = self.environment.get_state(0, r_high)
        v_wind_high = np.linalg.norm(state_high.wind_velocity)
        
        # Wind speed = omega * r. Higher r = Higher speed.
        self.assertGreater(v_wind_high, v_wind_surf)

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
